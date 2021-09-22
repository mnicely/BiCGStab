/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <typeinfo>  // for usage of C++ typeid

#include "cublas_v2.h"
#include "cusparse.h"
#include <cuda_runtime.h>
//  #include "helper_cusolver.h"
#include "mmio.h"

#include "mmio_wrapper.h"

//  #include "helper_cuda.h"

// profiling the code
#define TIME_INDIVIDUAL_LIBRARY_CALLS

#define DBICGSTAB_MAX_ULP_ERR 100
#define DBICGSTAB_EPS 1.E-14f  // 9e-2

#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto        status = static_cast<cudaError_t>( call );                                                         \
        auto        format = "ERROR: CUDA_RT call \"%s\" in line %d of file %s failed with code (%d).\n";              \
        auto        size   = std::snprintf( nullptr, 0, format, #call, __LINE__, __FILE__, status );                   \
        std::string output( size + 1, '\0' );                                                                          \
        std::sprintf( &output[0], format, #call, __LINE__, __FILE__, status );                                         \
        if ( status != cudaSuccess )                                                                                   \
            throw std::runtime_error( output );                                                                        \
    }
#endif  // CUDA_RT_CALL

#ifndef CUBLAS_CALL
#define CUBLAS_CALL( call )                                                                                            \
    {                                                                                                                  \
        auto        status = static_cast<cublasStatus_t>( call );                                                      \
        auto        format = "ERROR: CULBAS call \"%s\" in line %d of file %s failed with code (%d).\n";               \
        auto        size   = std::snprintf( nullptr, 0, format, #call, __LINE__, __FILE__, status );                   \
        std::string output( size + 1, '\0' );                                                                          \
        std::sprintf( &output[0], format, #call, __LINE__, __FILE__, status );                                         \
        if ( status != CUBLAS_STATUS_SUCCESS )                                                                         \
            throw std::runtime_error( output );                                                                        \
    }
#endif  // CUBLAS_CALL

#ifndef CUSPARSE_CALL
#define CUSPARSE_CALL( call )                                                                                          \
    {                                                                                                                  \
        auto        status = static_cast<cusparseStatus_t>( call );                                                    \
        auto        format = "ERROR: CUSPARSE call \"%s\" in line %d of file %s failed with code (%d).\n";             \
        auto        size   = std::snprintf( nullptr, 0, format, #call, __LINE__, __FILE__, status );                   \
        std::string output( size + 1, '\0' );                                                                          \
        std::sprintf( &output[0], format, #call, __LINE__, __FILE__, status );                                         \
        if ( status != CUSPARSE_STATUS_SUCCESS )                                                                       \
            throw std::runtime_error( output );                                                                        \
    }
#endif  // CUSPARSE_CALL

#define CLEANUP( )                                                                                                     \
    do {                                                                                                               \
        if ( x )                                                                                                       \
            free( x );                                                                                                 \
        if ( f )                                                                                                       \
            free( f );                                                                                                 \
        if ( r )                                                                                                       \
            free( r );                                                                                                 \
        if ( rw )                                                                                                      \
            free( rw );                                                                                                \
        if ( p )                                                                                                       \
            free( p );                                                                                                 \
        if ( pw )                                                                                                      \
            free( pw );                                                                                                \
        if ( s )                                                                                                       \
            free( s );                                                                                                 \
        if ( t )                                                                                                       \
            free( t );                                                                                                 \
        if ( v )                                                                                                       \
            free( v );                                                                                                 \
        if ( tx )                                                                                                      \
            free( tx );                                                                                                \
        if ( Aval )                                                                                                    \
            free( Aval );                                                                                              \
        if ( AcolsIndex )                                                                                              \
            free( AcolsIndex );                                                                                        \
        if ( ArowsIndex )                                                                                              \
            free( ArowsIndex );                                                                                        \
        if ( Mval )                                                                                                    \
            free( Mval );                                                                                              \
        if ( devPtrX )                                                                                                 \
            CUDA_RT_CALL( cudaFree( devPtrX ) );                                                                       \
        if ( devPtrF )                                                                                                 \
            CUDA_RT_CALL( cudaFree( devPtrF ) );                                                                       \
        if ( devPtrR )                                                                                                 \
            CUDA_RT_CALL( cudaFree( devPtrR ) );                                                                       \
        if ( devPtrRW )                                                                                                \
            CUDA_RT_CALL( cudaFree( devPtrRW ) );                                                                      \
        if ( devPtrP )                                                                                                 \
            CUDA_RT_CALL( cudaFree( devPtrP ) );                                                                       \
        if ( devPtrS )                                                                                                 \
            CUDA_RT_CALL( cudaFree( devPtrS ) );                                                                       \
        if ( devPtrT )                                                                                                 \
            CUDA_RT_CALL( cudaFree( devPtrT ) );                                                                       \
        if ( devPtrV )                                                                                                 \
            CUDA_RT_CALL( cudaFree( devPtrV ) );                                                                       \
        if ( devPtrAval )                                                                                              \
            CUDA_RT_CALL( cudaFree( devPtrAval ) );                                                                    \
        if ( devPtrAcolsIndex )                                                                                        \
            CUDA_RT_CALL( cudaFree( devPtrAcolsIndex ) );                                                              \
        if ( devPtrArowsIndex )                                                                                        \
            CUDA_RT_CALL( cudaFree( devPtrArowsIndex ) );                                                              \
        if ( devPtrMval )                                                                                              \
            CUDA_RT_CALL( cudaFree( devPtrMval ) );                                                                    \
        if ( stream )                                                                                                  \
            CUDA_RT_CALL( cudaStreamDestroy( stream ) );                                                               \
        if ( cublasHandle )                                                                                            \
            CUBLAS_CALL( cublasDestroy( cublasHandle ) );                                                              \
        if ( cusparseHandle )                                                                                          \
            CUSPARSE_CALL( cusparseDestroy( cusparseHandle ) );                                                        \
        fflush( stdout );                                                                                              \
    } while ( 0 )

#ifndef STRCPY
#define STRCPY( sFilePath, nLength, sPath ) strcpy( sFilePath, sPath )
#endif

#ifndef FOPEN
#define FOPEN( fHandle, filename, mode ) ( fHandle = fopen( filename, mode ) )
#endif

// PULLED from helper_cuda.h
inline char *sdkFindFilePath( const char *filename, const char *executable_path ) {
    // <executable_name> defines a variable that is replaced with the name of the
    // executable

    const char *searchPath[] = {
        "./",   // same dir
        "../",  // upper dir
    };

    // Extract the executable name
    std::string executable_name;

    if ( executable_path != 0 ) {
        executable_name = std::string( executable_path );

        // Linux & OSX path delimiter
        size_t delimiter_pos = executable_name.find_last_of( '/' );
        executable_name.erase( 0, delimiter_pos + 1 );
    }

    // Loop over all search paths and return the first hit
    for ( unsigned int i = 0; i < sizeof( searchPath ) / sizeof( char * ); ++i ) {
        std::string path( searchPath[i] );
        size_t      executable_name_pos = path.find( "<executable_name>" );

        // If there is executable_name variable in the searchPath
        // replace it with the value
        if ( executable_name_pos != std::string::npos ) {
            if ( executable_path != 0 ) {
                path.replace( executable_name_pos, strlen( "<executable_name>" ), executable_name );
            } else {
                // Skip this path entry if no executable argument is given
                continue;
            }
        }

        // Test if the file exists
        path.append( filename );
        FILE *fp;
        FOPEN( fp, path.c_str( ), "rb" );

        if ( fp != NULL ) {
            fclose( fp );
            // File found
            // returning an allocated array here for backwards compatibility reasons
            char *file_path = reinterpret_cast<char *>( malloc( path.length( ) + 1 ) );
            STRCPY( file_path, path.length( ) + 1, path.c_str( ) );
            return file_path;
        }

        if ( fp ) {
            fclose( fp );
        }
    }

    // File not found
    return 0;
}

// #include <stddef.h>
// #include <sys/resource.h>
// #include <sys/sysctl.h>
// #include <sys/time.h>
// #include <sys/types.h>
// double second( void ) {
//     struct timeval tv;
//     gettimeofday( &tv, NULL );
//     return ( double )tv.tv_sec + ( double )tv.tv_usec / 1000000.0;
// }

static void gpu_pbicgstab(
    cublasHandle_t             cublasHandle,
    cusparseHandle_t           cusparseHandle,
    int                        m,
    int                        n,
    int                        nnz,
    const cusparseSpMatDescr_t descra, /* the coefficient matrix in CSR format */
    double *                   a,
    const int *                ia,
    const int *                ja,
    const cusparseSpMatDescr_t descrm, /* the preconditioner in CSR format, lower & upper triangular factor */
    double *                   vm,
    const int *                im,
    const int *                jm,
    const cusparseMatDescr_t   descra_uli,
    cusparseDnVecDescr_t       vecX,
    cusparseDnVecDescr_t       vecY,
    cusparseSpSVDescr_t        spsvDescr_l,
    cusparseSpSVDescr_t        spsvDescr_u,
    cusparseFillMode_t         fillmode,
    cusparseDiagType_t         diagtype,
    csrilu02Info_t             info_M,
    //    csrsv2Info_t             info_l,
    //    csrsv2Info_t             info_u, /* the analysis of the lower and upper triangular parts */
    void *      dBuffer,
    void *      dBuffer_ilu,
    double *    f,
    double *    r,
    double *    rw,
    double *    p,
    double *    pw,
    double *    s,
    double *    t,
    double *    v,
    double *    x,
    int         maxit,
    double      tol,
    double      ttt_sv,
    cudaEvent_t ttm,
    cudaEvent_t ttm2,
    float       ttm_ttm2,
    cudaEvent_t ttl,
    cudaEvent_t ttl2,
    float       ttl_ttl2,
    cudaEvent_t ttu,
    cudaEvent_t ttu2,
    float       ttu_ttu2 ) {

    double rho        = 0.0;
    double rhop       = 0.0;
    double beta       = 0.0;
    double alpha      = 0.0;
    double negalpha   = 0.0;
    double omega      = 0.0;
    double negomega   = 0.0;
    double temp       = 0.0;
    double temp2      = 0.0;
    double nrmr       = 0.0;
    double nrmr0      = 0.0;
    rho               = 0.0;
    const double zero = 0.0;
    const double one  = 1.0;
    double       mone = -1.0;
    int          i    = 0;
    int          j    = 0;
    // double       ttl, ttl2, ttu, ttu2, ttm, ttm2;
    // double ttt_mv = 0.0;

// WARNING: Analysis is done outside of the function (and the time taken by it is passed to the function in variable
// ttt_sv)

// compute initial residual r0=b-Ax0 (using initial guess in x)
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
    // CUDA_RT_CALL( cudaDeviceSynchronize( ) );
    // ttm = second( );
    double ttt_mv = 0.0;
    CUDA_RT_CALL( cudaEventRecord( ttm ) );
#endif

    void * dBuffer_mv    = NULL;
    size_t bufferSize_mv = 0;

    // CUSPARSE_CALL( cusparseDcsrmv(
    //     cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, descra, a, ia, ja, x, &zero, r ) );

    CUSPARSE_CALL( cusparseCreateDnVec( &vecX, n, x, CUDA_R_64F ) )
    CUSPARSE_CALL( cusparseCreateDnVec( &vecY, n, r, CUDA_R_64F ) )

    // allocate an external buffer if needed
    CUSPARSE_CALL( cusparseSpMV_bufferSize( cusparseHandle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &one,
                                            descra,
                                            vecX,
                                            &zero,
                                            vecY,
                                            CUDA_R_64F,
                                            CUSPARSE_MV_ALG_DEFAULT,
                                            &bufferSize_mv ) )

    CUDA_RT_CALL( cudaMalloc( &dBuffer_mv, bufferSize_mv ) )

    // execute SpMV
    CUSPARSE_CALL( cusparseSpMV( cusparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &one,
                                 descra,
                                 vecX,
                                 &zero,
                                 vecY,
                                 CUDA_R_64F,
                                 CUSPARSE_MV_ALG_DEFAULT,
                                 dBuffer_mv ) )

#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
    // CUDA_RT_CALL( cudaDeviceSynchronize( ) );
    CUDA_RT_CALL( cudaEventRecord( ttm2 ) );
    CUDA_RT_CALL( cudaEventSynchronize( ttm2 ) );
    CUDA_RT_CALL( cudaEventElapsedTime( &ttm_ttm2, ttm, ttm2 ) );
    // ttm2 = second( );
    ttt_mv += ( ttm_ttm2 * 1e-3 );
    printf( "matvec %f (s)\n", ttm_ttm2 * 1e-3 );
#endif
    // 	 CUBLAS_CALL(cublasDscal(cublasHandle, n, &mone, r, 1));
    // 	 CUBLAS_CALL(cublasDaxpy(cublasHandle, n, &one, f, 1, r, 1));
    // copy residual r into r^{\hat} and p
    // 	 CUBLAS_CALL(cublasDcopy(cublasHandle, n, r, 1, rw, 1));
    // 	 CUBLAS_CALL(cublasDcopy(cublasHandle, n, r, 1, p, 1));
    // 	 CUBLAS_CALL(cublasDnrm2(cublasHandle, n, r, 1, &nrmr0));

    CUBLAS_CALL( cublasScalEx( cublasHandle, n, &mone, CUDA_R_64F, r, CUDA_R_64F, 1, CUDA_R_64F ) );
    CUBLAS_CALL( cublasAxpyEx( cublasHandle, n, &one, CUDA_R_64F, f, CUDA_R_64F, 1, r, CUDA_R_64F, 1, CUDA_R_64F ) );
    // copy residual r into r^{\hat} and p
    CUBLAS_CALL( cublasDcopy( cublasHandle, n, r, 1, rw, 1 ) );
    CUBLAS_CALL( cublasDcopy( cublasHandle, n, r, 1, p, 1 ) );
    CUBLAS_CALL( cublasDnrm2( cublasHandle, n, r, 1, &nrmr0 ) );
    printf( "gpu, init residual:norm %20.16f\n", nrmr0 );

    for ( i = 0; i < maxit; ) {
        rhop = rho;
        // 		 CUBLAS_CALL(cublasDdot(cublasHandle, n, rw, 1, r, 1, &rho));
        CUBLAS_CALL(
            cublasDotEx( cublasHandle, n, rw, CUDA_R_64F, 1, r, CUDA_R_64F, 1, &rho, CUDA_R_64F, CUDA_R_64F ) );

        if ( i > 0 ) {
            beta     = ( rho / rhop ) * ( alpha / omega );
            negomega = -omega;
            // 			 CUBLAS_CALL(cublasDaxpy(cublasHandle,n, &negomega, v, 1, p, 1));
            // 			 CUBLAS_CALL(cublasDscal(cublasHandle,n, &beta, p, 1));
            // 			 CUBLAS_CALL(cublasDaxpy(cublasHandle,n, &one, r, 1, p, 1));
            CUBLAS_CALL( cublasAxpyEx(
                cublasHandle, n, &negomega, CUDA_R_64F, v, CUDA_R_64F, 1, p, CUDA_R_64F, 1, CUDA_R_64F ) );
            CUBLAS_CALL( cublasScalEx( cublasHandle, n, &beta, CUDA_R_64F, p, CUDA_R_64F, 1, CUDA_R_64F ) );
            CUBLAS_CALL(
                cublasAxpyEx( cublasHandle, n, &one, CUDA_R_64F, r, CUDA_R_64F, 1, p, CUDA_R_64F, 1, CUDA_R_64F ) );
        }
        // preconditioning step (lower and upper triangular solve)
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
        // CUDA_RT_CALL( cudaDeviceSynchronize( ) );
        // ttl = second( );
        CUDA_RT_CALL( cudaEventRecord( ttl ) );
#endif
        // CUSPARSE_CALL( cusparseSetMatFillMode( descrm, CUSPARSE_FILL_MODE_LOWER ) );
        // CUSPARSE_CALL( cusparseSetMatDiagType( descrm, CUSPARSE_DIAG_TYPE_UNIT ) );
        // // 		 CUSPARSE_CALL(cusparseDcsrsv_solve(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,n,&one,
        // // descrm,vm,im,jm,info_l,p,t));
        // CUSPARSE_CALL( cusparseDcsrsv2_solve( cusparseHandle,
        //                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
        //                                       n,
        //                                       nnz,
        //                                       &one,
        //                                       descrm,
        //                                       vm,
        //                                       im,
        //                                       jm,
        //                                       info_l,
        //                                       p,
        //                                       t,
        //                                       CUSPARSE_SOLVE_POLICY_NO_LEVEL,
        //                                       dBuffer_l ) );

        CUSPARSE_CALL( cusparseCreateDnVec( &vecX, n, p, CUDA_R_64F ) )
        CUSPARSE_CALL( cusparseCreateDnVec( &vecY, n, t, CUDA_R_64F ) )

        // execute SpSV Lower
        CUSPARSE_CALL( cusparseSpSV_solve( cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one,
                                           descrm,
                                           vecX,
                                           vecY,
                                           CUDA_R_64F,
                                           CUSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescr_l ) )

#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
        // CUDA_RT_CALL( cudaDeviceSynchronize( ) );
        CUDA_RT_CALL( cudaEventRecord( ttl2 ) );
        CUDA_RT_CALL( cudaEventSynchronize( ttl2 ) );
        CUDA_RT_CALL( cudaEventElapsedTime( &ttl_ttl2, ttl, ttl2 ) );

        // ttl2 = second( );
        // ttu  = second( );
        CUDA_RT_CALL( cudaEventRecord( ttu ) );

#endif
        // CUSPARSE_CALL( cusparseSetMatFillMode( descrm, CUSPARSE_FILL_MODE_UPPER ) );
        // CUSPARSE_CALL( cusparseSetMatDiagType( descrm, CUSPARSE_DIAG_TYPE_NON_UNIT ) );
        // // 		 CUSPARSE_CALL(cusparseDcsrsv_solve(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,n,&one,
        // // descrm,vm,im,jm,info_u,t,pw));
        // CUSPARSE_CALL( cusparseDcsrsv2_solve( cusparseHandle,
        //                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
        //                                       n,
        //                                       nnz,
        //                                       &one,
        //                                       descrm,
        //                                       vm,
        //                                       im,
        //                                       jm,
        //                                       info_u,
        //                                       t,
        //                                       pw,
        //                                       CUSPARSE_SOLVE_POLICY_NO_LEVEL,
        //                                       dBuffer_u ) );

        CUSPARSE_CALL( cusparseCreateDnVec( &vecX, n, t, CUDA_R_64F ) )
        CUSPARSE_CALL( cusparseCreateDnVec( &vecY, n, pw, CUDA_R_64F ) )

        // execute SpSV Lower
        CUSPARSE_CALL( cusparseSpSV_solve( cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one,
                                           descrm,
                                           vecX,
                                           vecY,
                                           CUDA_R_64F,
                                           CUSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescr_u ) )

#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
        // CUDA_RT_CALL( cudaDeviceSynchronize( ) );
        // ttu2 = second( );
        CUDA_RT_CALL( cudaEventRecord( ttu2 ) );
        CUDA_RT_CALL( cudaEventSynchronize( ttu2 ) );
        CUDA_RT_CALL( cudaEventElapsedTime( &ttu_ttu2, ttu, ttu2 ) );
        ttt_sv += ( ttl_ttl2 * 1e-3 ) + ( ttu_ttu2 * 1e-3 );
        printf( "solve lower %f (s), upper %f (s) \n", ttl_ttl2 * 1e-3, ttu_ttu2 * 1e-3 );
#endif

        // matrix-vector multiplication
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
        // CUDA_RT_CALL( cudaDeviceSynchronize( ) );
        // ttm = second( );
        CUDA_RT_CALL( cudaEventRecord( ttm ) );
#endif

        // 		 CUSPARSE_CALL(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one,
        // descra, a, ia, ja, pw, &zero, v));

#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
        // CUDA_RT_CALL( cudaDeviceSynchronize( ) );
        // ttm2 = second( );
        CUDA_RT_CALL( cudaEventRecord( ttm2 ) );
        CUDA_RT_CALL( cudaEventSynchronize( ttm2 ) );
        CUDA_RT_CALL( cudaEventElapsedTime( &ttm_ttm2, ttm, ttm2 ) );
        ttt_mv += ( ttm_ttm2 * 1e-3 );
        printf( "matvec %f (s)\n", ttm_ttm2 * 1e-3 );
#endif

        // 		 CUBLAS_CALL(cublasDdot(cublasHandle,n, rw, 1, v, 1,&temp));
        CUBLAS_CALL(
            cublasDotEx( cublasHandle, n, rw, CUDA_R_64F, 1, v, CUDA_R_64F, 1, &temp, CUDA_R_64F, CUDA_R_64F ) );
        alpha    = rho / temp;
        negalpha = -( alpha );
        // 		 CUBLAS_CALL(cublasDaxpy(cublasHandle,n, &negalpha, v, 1, r, 1));

        // 		 CUBLAS_CALL(cublasDaxpy(cublasHandle,n, &alpha,        pw, 1, x, 1));
        // 		 CUBLAS_CALL(cublasDnrm2(cublasHandle, n, r, 1, &nrmr));
        CUBLAS_CALL(
            cublasAxpyEx( cublasHandle, n, &negalpha, CUDA_R_64F, v, CUDA_R_64F, 1, r, CUDA_R_64F, 1, CUDA_R_64F ) );
        CUBLAS_CALL(
            cublasAxpyEx( cublasHandle, n, &alpha, CUDA_R_64F, pw, CUDA_R_64F, 1, x, CUDA_R_64F, 1, CUDA_R_64F ) );
        CUBLAS_CALL( cublasDnrm2( cublasHandle, n, r, 1, &nrmr ) );

        if ( nrmr < tol * nrmr0 ) {
            j = 5;
            break;
        }

        // preconditioning step (lower and upper triangular solve)
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
        // CUDA_RT_CALL( cudaDeviceSynchronize( ) );
        // ttl = second( );
        CUDA_RT_CALL( cudaEventRecord( ttl ) );
#endif
        // CUSPARSE_CALL( cusparseSetMatFillMode( descrm, CUSPARSE_FILL_MODE_LOWER ) );
        // CUSPARSE_CALL( cusparseSetMatDiagType( descrm, CUSPARSE_DIAG_TYPE_UNIT ) );
        // // 		 CUSPARSE_CALL(cusparseDcsrsv_solve(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,n,&one,
        // // descrm,vm,im,jm,info_l,r,t));
        // CUSPARSE_CALL( cusparseDcsrsv2_solve( cusparseHandle,
        //                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
        //                                       n,
        //                                       nnz,
        //                                       &one,
        //                                       descrm,
        //                                       vm,
        //                                       im,
        //                                       jm,
        //                                       info_l,
        //                                       r,
        //                                       t,
        //                                       CUSPARSE_SOLVE_POLICY_NO_LEVEL,
        //                                       dBuffer_l ) );
        CUSPARSE_CALL( cusparseCreateDnVec( &vecX, n, r, CUDA_R_64F ) )
        CUSPARSE_CALL( cusparseCreateDnVec( &vecY, n, t, CUDA_R_64F ) )

        // execute SpSV Lower
        CUSPARSE_CALL( cusparseSpSV_solve( cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one,
                                           descrm,
                                           vecX,
                                           vecY,
                                           CUDA_R_64F,
                                           CUSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescr_l ) )

#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
        // CUDA_RT_CALL( cudaDeviceSynchronize( ) );
        // ttl2 = second( );
        CUDA_RT_CALL( cudaEventRecord( ttl2 ) );
        CUDA_RT_CALL( cudaEventSynchronize( ttl2 ) );
        CUDA_RT_CALL( cudaEventElapsedTime( &ttl_ttl2, ttl, ttl2 ) );
        // ttu  = second( );
        CUDA_RT_CALL( cudaEventRecord( ttu ) );
#endif
        // CUSPARSE_CALL( cusparseSetMatFillMode( descrm, CUSPARSE_FILL_MODE_UPPER ) );
        // CUSPARSE_CALL( cusparseSetMatDiagType( descrm, CUSPARSE_DIAG_TYPE_NON_UNIT ) );
        // // 		 CUSPARSE_CALL(cusparseDcsrsv_solve(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,n,
        // // &one,descrm,vm,im,jm,info_u,t,s));
        // CUSPARSE_CALL( cusparseDcsrsv2_solve( cusparseHandle,
        //                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
        //                                       n,
        //                                       nnz,
        //                                       &one,
        //                                       descrm,
        //                                       vm,
        //                                       im,
        //                                       jm,
        //                                       info_u,
        //                                       t,
        //                                       s,
        //                                       CUSPARSE_SOLVE_POLICY_NO_LEVEL,
        //                                       dBuffer_u ) );

        CUSPARSE_CALL( cusparseCreateDnVec( &vecX, n, t, CUDA_R_64F ) )
        CUSPARSE_CALL( cusparseCreateDnVec( &vecY, n, s, CUDA_R_64F ) )

        // execute SpSV Lower
        CUSPARSE_CALL( cusparseSpSV_solve( cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one,
                                           descrm,
                                           vecX,
                                           vecY,
                                           CUDA_R_64F,
                                           CUSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescr_u ) )
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
        // CUDA_RT_CALL( cudaDeviceSynchronize( ) );
        // ttu2 = second( );
        CUDA_RT_CALL( cudaEventRecord( ttu2 ) );
        CUDA_RT_CALL( cudaEventSynchronize( ttu2 ) );
        CUDA_RT_CALL( cudaEventElapsedTime( &ttu_ttu2, ttu, ttu2 ) );
        ttt_sv += ( ttl_ttl2 * 1e-3 ) + ( ttu_ttu2 * 1e-3 );
        printf( "solve lower %f (s), upper %f (s) \n", ttl_ttl2 * 1e-3, ttu_ttu2 * 1e-3 );
#endif
        // matrix-vector multiplication
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
        // CUDA_RT_CALL( cudaDeviceSynchronize( ) );
        // ttm = second( );
        CUDA_RT_CALL( cudaEventRecord( ttm ) );
#endif

        // 		 CUSPARSE_CALL(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one,
        // descra, a, ia, ja, s, &zero, t));
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
        // CUDA_RT_CALL( cudaDeviceSynchronize( ) );
        // ttm2 = second( );
        CUDA_RT_CALL( cudaEventRecord( ttm2 ) );
        CUDA_RT_CALL( cudaEventSynchronize( ttm2 ) );
        CUDA_RT_CALL( cudaEventElapsedTime( &ttm_ttm2, ttm, ttm2 ) );
        ttt_mv += ( ttm_ttm2 * 1e-3 );
        printf( "matvec %f (s)\n", ttm_ttm2 * 1e-3 );
#endif

        // 		 CUBLAS_CALL(cublasDdot(cublasHandle,n, t, 1, r, 1,&temp));
        // 		 CUBLAS_CALL(cublasDdot(cublasHandle,n, t, 1, t, 1,&temp2));
        CUBLAS_CALL(
            cublasDotEx( cublasHandle, n, t, CUDA_R_64F, 1, r, CUDA_R_64F, 1, &temp, CUDA_R_64F, CUDA_R_64F ) );
        CUBLAS_CALL(
            cublasDotEx( cublasHandle, n, t, CUDA_R_64F, 1, t, CUDA_R_64F, 1, &temp2, CUDA_R_64F, CUDA_R_64F ) );

        omega    = temp / temp2;
        negomega = -( omega );
        // 		 CUBLAS_CALL(cublasDaxpy(cublasHandle,n, &omega, s, 1, x, 1));
        // 		 CUBLAS_CALL(cublasDaxpy(cublasHandle,n, &negomega, t, 1, r, 1));

        // 		 CUBLAS_CALL(cublasDnrm2(cublasHandle,n, r, 1,&nrmr));

        CUBLAS_CALL(
            cublasAxpyEx( cublasHandle, n, &omega, CUDA_R_64F, s, CUDA_R_64F, 1, x, CUDA_R_64F, 1, CUDA_R_64F ) );
        CUBLAS_CALL(
            cublasAxpyEx( cublasHandle, n, &negomega, CUDA_R_64F, t, CUDA_R_64F, 1, r, CUDA_R_64F, 1, CUDA_R_64F ) );
        CUBLAS_CALL( cublasDnrm2( cublasHandle, n, r, 1, &nrmr ) );

        if ( nrmr < tol * nrmr0 ) {
            i++;
            j = 0;
            break;
        }
        i++;
    }

#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
    printf( "gpu total solve time %f (s), matvec time %f (s)\n", ttt_sv, ttt_mv );
#endif
}

int test_bicgstab( char *      matrix_filename,
                   char *      coloring_filename,
                   const char *element_type,
                   int         symmetrize,
                   int         debug,
                   double      damping,
                   int         maxit,
                   double      tol,
                   float       err,
                   float       eps ) {

    cublasHandle_t   cublasHandle   = NULL;
    cusparseHandle_t cusparseHandle = NULL;
    // cusparseMatDescr_t descra         = NULL;
    // cusparseMatDescr_t descrm         = NULL;
    cusparseSpMatDescr_t descra      = NULL;
    cusparseSpMatDescr_t descrm      = NULL;
    cusparseMatDescr_t   descra_uli  = NULL;
    cusparseDnVecDescr_t vecX        = NULL;
    cusparseDnVecDescr_t vecY        = NULL;
    cusparseSpSVDescr_t  spsvDescr_l = NULL;
    cusparseSpSVDescr_t  spsvDescr_u = NULL;
    cusparseFillMode_t   fillmode;
    cusparseDiagType_t   diagtype;
    cudaStream_t         stream = NULL;
    csrilu02Info_t       info_M = NULL;
    // csrsv2Info_t       info_l         = NULL;
    // csrsv2Info_t       info_u         = NULL;
    // cusparseStatus_t status1, status2, status3;
    double *devPtrAval       = nullptr;
    int *   devPtrAcolsIndex = nullptr;
    int *   devPtrArowsIndex = nullptr;
    double *devPtrMval       = nullptr;
    int *   devPtrMcolsIndex = nullptr;
    int *   devPtrMrowsIndex = nullptr;
    double *devPtrX          = nullptr;
    double *devPtrF          = nullptr;
    double *devPtrR          = nullptr;
    double *devPtrRW         = nullptr;
    double *devPtrP          = nullptr;
    double *devPtrPW         = nullptr;
    double *devPtrS          = nullptr;
    double *devPtrT          = nullptr;
    double *devPtrV          = nullptr;
    double *Aval             = nullptr;
    int *   AcolsIndex       = nullptr;
    int *   ArowsIndex       = nullptr;
    double *Mval             = nullptr;
    // int *   MrowsIndex       = nullptr;
    // int *   McolsIndex       = nullptr;
    double *x  = nullptr;
    double *tx = nullptr;
    double *f  = nullptr;
    double *r  = nullptr;
    double *rw = nullptr;
    double *p  = nullptr;
    double *pw = nullptr;
    double *s  = nullptr;
    double *t  = nullptr;
    double *v  = nullptr;
    int     matrixM;
    int     matrixN;
    int     matrixSizeAval;
    int     matrixSizeAcolsIndex;
    int     matrixSizeArowsIndex;
    int     mSizeAval;
    // int     mSizeAcolsIndex;
    // int     mSizeArowsIndex;
    int arraySizeX;
    int arraySizeF;
    int arraySizeR;
    int arraySizeRW;
    int arraySizeP;
    int arraySizePW;
    int arraySizeS;
    int arraySizeT;
    int arraySizeV;
    int nnz;
    int mNNZ;
    // long long           flops;
    // double              start, stop;
    int num_iterations;
    // int                 nbrTests;
    int count;
    int base;
    int mbase;
    // cusparseOperation_t trans;
    double alpha;
    double ttt_sv = 0.0;

    printf( "Testing %cbicgstab\n", *element_type );

    alpha = damping;
    // trans = CUSPARSE_OPERATION_NON_TRANSPOSE;

    /* load the coefficient matrix */
    if ( loadMMSparseMatrix( matrix_filename,
                             *element_type,
                             true,
                             &matrixM,
                             &matrixN,
                             &nnz,
                             &Aval,
                             &ArowsIndex,
                             &AcolsIndex,
                             symmetrize ) ) {
        CLEANUP( );
        fprintf( stderr, "!!!! cusparseLoadMMSparseMatrix FAILED\n" );
        return EXIT_FAILURE;
    }

    matrixSizeAval       = nnz;
    matrixSizeAcolsIndex = matrixSizeAval;
    matrixSizeArowsIndex = matrixM + 1;
    base                 = ArowsIndex[0];
    if ( matrixM != matrixN ) {
        fprintf( stderr, "!!!! matrix MUST be square, error: m=%d != n=%d\n", matrixM, matrixN );
        return EXIT_FAILURE;
    }
    printf( "^^^^ M=%d, N=%d, nnz=%d\n", matrixM, matrixN, nnz );

    /* set some extra parameters for lower triangular factor */
    mNNZ      = ArowsIndex[matrixM] - ArowsIndex[0];
    mSizeAval = mNNZ;
    // mSizeAcolsIndex = mSizeAval;
    // mSizeArowsIndex = matrixM + 1;
    mbase = ArowsIndex[0];

    /* compressed sparse row */
    arraySizeX  = matrixN;
    arraySizeF  = matrixM;
    arraySizeR  = matrixM;
    arraySizeRW = matrixM;
    arraySizeP  = matrixN;
    arraySizePW = matrixN;
    arraySizeS  = matrixM;
    arraySizeT  = matrixM;
    arraySizeV  = matrixM;

    CUBLAS_CALL( cublasCreate( &cublasHandle ) );
    CUSPARSE_CALL( cusparseCreate( &cusparseHandle ) )

    /* create three matrix descriptors */
    // status1 = cusparseCreateMatDescr( &descra );
    // status2 = cusparseCreateMatDescr( &descrm );
    // if ( ( status1 != CUSPARSE_STATUS_SUCCESS ) || ( status2 != CUSPARSE_STATUS_SUCCESS ) ) {
    //     fprintf( stderr, "!!!! CUSPARSE cusparseCreateMatDescr (coefficient matrix or preconditioner) error\n" );
    //     return EXIT_FAILURE;
    // }

    /* allocate device memory for csr matrix and vectors */
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &devPtrX ), sizeof( devPtrX[0] ) * arraySizeX ) );
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &devPtrF ), sizeof( devPtrF[0] ) * arraySizeF ) );
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &devPtrR ), sizeof( devPtrR[0] ) * arraySizeR ) );
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &devPtrRW ), sizeof( devPtrRW[0] ) * arraySizeRW ) );
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &devPtrP ), sizeof( devPtrP[0] ) * arraySizeP ) );
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &devPtrPW ), sizeof( devPtrPW[0] ) * arraySizePW ) );
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &devPtrS ), sizeof( devPtrS[0] ) * arraySizeS ) );
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &devPtrT ), sizeof( devPtrT[0] ) * arraySizeT ) );
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &devPtrV ), sizeof( devPtrV[0] ) * arraySizeV ) );
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &devPtrAval ), sizeof( devPtrAval[0] ) * matrixSizeAval ) );
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &devPtrAcolsIndex ),
                              sizeof( devPtrAcolsIndex[0] ) * matrixSizeAcolsIndex ) );
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &devPtrArowsIndex ),
                              sizeof( devPtrArowsIndex[0] ) * matrixSizeArowsIndex ) );
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &devPtrMval ), sizeof( devPtrMval[0] ) * mSizeAval ) );

    /* allocate host memory for  vectors */
    x    = ( double * )malloc( arraySizeX * sizeof( x[0] ) );
    f    = ( double * )malloc( arraySizeF * sizeof( f[0] ) );
    r    = ( double * )malloc( arraySizeR * sizeof( r[0] ) );
    rw   = ( double * )malloc( arraySizeRW * sizeof( rw[0] ) );
    p    = ( double * )malloc( arraySizeP * sizeof( p[0] ) );
    pw   = ( double * )malloc( arraySizePW * sizeof( pw[0] ) );
    s    = ( double * )malloc( arraySizeS * sizeof( s[0] ) );
    t    = ( double * )malloc( arraySizeT * sizeof( t[0] ) );
    v    = ( double * )malloc( arraySizeV * sizeof( v[0] ) );
    tx   = ( double * )malloc( arraySizeX * sizeof( tx[0] ) );
    Mval = ( double * )malloc( matrixSizeAval * sizeof( Mval[0] ) );
    if ( ( !Aval ) || ( !AcolsIndex ) || ( !ArowsIndex ) || ( !Mval ) || ( !x ) || ( !f ) || ( !r ) || ( !rw ) ||
         ( !p ) || ( !pw ) || ( !s ) || ( !t ) || ( !v ) || ( !tx ) ) {
        CLEANUP( );
        fprintf( stderr, "!!!! memory allocation error\n" );
        return EXIT_FAILURE;
    }
    /* use streams */
    int useStream = 0;
    if ( useStream ) {

        CUDA_RT_CALL( cudaStreamCreate( &stream ) );
        CUBLAS_CALL( cublasSetStream( cublasHandle, stream ) );
        CUSPARSE_CALL( cusparseSetStream( cusparseHandle, stream ) );
    }

    /* clean memory */
    CUDA_RT_CALL( cudaMemset( reinterpret_cast<void *>( devPtrX ), 0, sizeof( devPtrX[0] ) * arraySizeX ) );
    CUDA_RT_CALL( cudaMemset( reinterpret_cast<void *>( devPtrF ), 0, sizeof( devPtrF[0] ) * arraySizeF ) );
    CUDA_RT_CALL( cudaMemset( reinterpret_cast<void *>( devPtrR ), 0, sizeof( devPtrR[0] ) * arraySizeR ) );
    CUDA_RT_CALL( cudaMemset( reinterpret_cast<void *>( devPtrRW ), 0, sizeof( devPtrRW[0] ) * arraySizeRW ) );
    CUDA_RT_CALL( cudaMemset( reinterpret_cast<void *>( devPtrP ), 0, sizeof( devPtrP[0] ) * arraySizeP ) );
    CUDA_RT_CALL( cudaMemset( reinterpret_cast<void *>( devPtrPW ), 0, sizeof( devPtrPW[0] ) * arraySizePW ) );
    CUDA_RT_CALL( cudaMemset( reinterpret_cast<void *>( devPtrS ), 0, sizeof( devPtrS[0] ) * arraySizeS ) );
    CUDA_RT_CALL( cudaMemset( reinterpret_cast<void *>( devPtrT ), 0, sizeof( devPtrT[0] ) * arraySizeT ) );
    CUDA_RT_CALL( cudaMemset( reinterpret_cast<void *>( devPtrV ), 0, sizeof( devPtrV[0] ) * arraySizeV ) );
    CUDA_RT_CALL( cudaMemset( reinterpret_cast<void *>( devPtrAval ), 0, sizeof( devPtrAval[0] ) * matrixSizeAval ) );
    CUDA_RT_CALL( cudaMemset(
        reinterpret_cast<void *>( devPtrAcolsIndex ), 0, sizeof( devPtrAcolsIndex[0] ) * matrixSizeAcolsIndex ) );
    CUDA_RT_CALL( cudaMemset(
        reinterpret_cast<void *>( devPtrArowsIndex ), 0, sizeof( devPtrArowsIndex[0] ) * matrixSizeArowsIndex ) );
    CUDA_RT_CALL( cudaMemset( reinterpret_cast<void *>( devPtrMval ), 0, sizeof( devPtrMval[0] ) * mSizeAval ) );

    memset( x, 0, arraySizeX * sizeof( x[0] ) );
    memset( f, 0, arraySizeF * sizeof( f[0] ) );
    memset( r, 0, arraySizeR * sizeof( r[0] ) );
    memset( rw, 0, arraySizeRW * sizeof( rw[0] ) );
    memset( p, 0, arraySizeP * sizeof( p[0] ) );
    memset( pw, 0, arraySizePW * sizeof( pw[0] ) );
    memset( s, 0, arraySizeS * sizeof( s[0] ) );
    memset( t, 0, arraySizeT * sizeof( t[0] ) );
    memset( v, 0, arraySizeV * sizeof( v[0] ) );
    memset( tx, 0, arraySizeX * sizeof( tx[0] ) );

    // Timing
    cudaEvent_t start_matrix_copy { nullptr };
    cudaEvent_t stop_matrix_copy { nullptr };
    cudaEvent_t ttm { nullptr };
    cudaEvent_t ttm2 { nullptr };
    float       ttm_ttm2 {};
    cudaEvent_t ttl { nullptr };
    cudaEvent_t ttl2 { nullptr };
    float       ttl_ttl2 {};
    cudaEvent_t ttu { nullptr };
    cudaEvent_t ttu2 { nullptr };
    float       ttu_ttu2 {};
    cudaEvent_t start_ilu { nullptr };
    cudaEvent_t stop_ilu { nullptr };
    cudaEvent_t start_event { nullptr };
    cudaEvent_t stop_event { nullptr };
    float       elapsed_events_ms {};

    CUDA_RT_CALL( cudaEventCreate( &start_matrix_copy, cudaEventBlockingSync ) );
    CUDA_RT_CALL( cudaEventCreate( &stop_matrix_copy, cudaEventBlockingSync ) );
    CUDA_RT_CALL( cudaEventCreate( &ttm, cudaEventBlockingSync ) );
    CUDA_RT_CALL( cudaEventCreate( &ttm2, cudaEventBlockingSync ) );
    CUDA_RT_CALL( cudaEventCreate( &ttl, cudaEventBlockingSync ) );
    CUDA_RT_CALL( cudaEventCreate( &ttl2, cudaEventBlockingSync ) );
    CUDA_RT_CALL( cudaEventCreate( &ttu, cudaEventBlockingSync ) );
    CUDA_RT_CALL( cudaEventCreate( &ttu2, cudaEventBlockingSync ) );
    CUDA_RT_CALL( cudaEventCreate( &start_ilu, cudaEventBlockingSync ) );
    CUDA_RT_CALL( cudaEventCreate( &stop_ilu, cudaEventBlockingSync ) );
    CUDA_RT_CALL( cudaEventCreate( &start_event, cudaEventBlockingSync ) );
    CUDA_RT_CALL( cudaEventCreate( &stop_event, cudaEventBlockingSync ) );

    // Moved here from ILU section to prevent runtime error
    devPtrMrowsIndex = devPtrArowsIndex;
    devPtrMcolsIndex = devPtrAcolsIndex;

    /* create the test matrix and vectors on the host */
    if ( base ) {
        CUSPARSE_CALL( cusparseCreateCsr( &descra,
                                          matrixM,
                                          matrixN,
                                          nnz,
                                          devPtrArowsIndex,
                                          devPtrAcolsIndex,
                                          devPtrAval,
                                          CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_BASE_ONE,
                                          CUDA_R_64F ) )
    } else {
        CUSPARSE_CALL( cusparseCreateCsr( &descra,
                                          matrixM,
                                          matrixN,
                                          nnz,
                                          devPtrArowsIndex,
                                          devPtrAcolsIndex,
                                          devPtrAval,
                                          CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_BASE_ZERO,
                                          CUDA_R_64F ) )
    }
    if ( mbase ) {
        CUSPARSE_CALL( cusparseCreateCsr( &descrm,
                                          matrixM,
                                          matrixN,
                                          nnz,
                                          devPtrMrowsIndex,
                                          devPtrMcolsIndex,
                                          devPtrMval,
                                          CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_BASE_ONE,
                                          CUDA_R_64F ) )
    } else {
        CUSPARSE_CALL( cusparseCreateCsr( &descrm,
                                          matrixM,
                                          matrixN,
                                          nnz,
                                          devPtrMrowsIndex,
                                          devPtrMcolsIndex,
                                          devPtrMval,
                                          CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_BASE_ZERO,
                                          CUDA_R_64F ) )
    }

    // compute the right-hand-side f=A*e, where e=[1, ..., 1]'
    for ( int i = 0; i < arraySizeP; i++ ) {
        p[i] = 1.0;
    }

    /* copy the csr matrix and vectors into device memory */
    CUDA_RT_CALL( cudaEventRecord( start_matrix_copy ) );

    CUDA_RT_CALL(
        cudaMemcpy( devPtrAval, Aval, ( size_t )( matrixSizeAval * sizeof( Aval[0] ) ), cudaMemcpyHostToDevice ) );
    CUDA_RT_CALL( cudaMemcpy( devPtrAcolsIndex,
                              AcolsIndex,
                              ( size_t )( matrixSizeAcolsIndex * sizeof( AcolsIndex[0] ) ),
                              cudaMemcpyHostToDevice ) );
    CUDA_RT_CALL( cudaMemcpy( devPtrArowsIndex,
                              ArowsIndex,
                              ( size_t )( matrixSizeArowsIndex * sizeof( ArowsIndex[0] ) ),
                              cudaMemcpyHostToDevice ) );
    CUDA_RT_CALL( cudaMemcpy(
        devPtrMval, devPtrAval, ( size_t )( matrixSizeAval * sizeof( devPtrMval[0] ) ), cudaMemcpyDeviceToDevice ) );

    CUDA_RT_CALL( cudaEventRecord( stop_matrix_copy ) );
    CUDA_RT_CALL( cudaEventSynchronize( stop_matrix_copy ) );
    CUDA_RT_CALL( cudaEventElapsedTime( &elapsed_events_ms, start_matrix_copy, stop_matrix_copy ) );

    fprintf( stdout, "Copy matrix from CPU to GPU, time(s) = %f\n", elapsed_events_ms * 1e-3 );

    CUDA_RT_CALL( cudaMemcpy( devPtrX, x, ( size_t )( arraySizeX * sizeof( devPtrX[0] ) ), cudaMemcpyHostToDevice ) );
    CUDA_RT_CALL( cudaMemcpy( devPtrF, f, ( size_t )( arraySizeF * sizeof( devPtrF[0] ) ), cudaMemcpyHostToDevice ) );
    CUDA_RT_CALL( cudaMemcpy( devPtrR, r, ( size_t )( arraySizeR * sizeof( devPtrR[0] ) ), cudaMemcpyHostToDevice ) );
    CUDA_RT_CALL(
        cudaMemcpy( devPtrRW, rw, ( size_t )( arraySizeRW * sizeof( devPtrRW[0] ) ), cudaMemcpyHostToDevice ) );
    CUDA_RT_CALL( cudaMemcpy( devPtrP, p, ( size_t )( arraySizeP * sizeof( devPtrP[0] ) ), cudaMemcpyHostToDevice ) );
    CUDA_RT_CALL(
        cudaMemcpy( devPtrPW, pw, ( size_t )( arraySizePW * sizeof( devPtrPW[0] ) ), cudaMemcpyHostToDevice ) );
    CUDA_RT_CALL( cudaMemcpy( devPtrS, s, ( size_t )( arraySizeS * sizeof( devPtrS[0] ) ), cudaMemcpyHostToDevice ) );
    CUDA_RT_CALL( cudaMemcpy( devPtrT, t, ( size_t )( arraySizeT * sizeof( devPtrT[0] ) ), cudaMemcpyHostToDevice ) );
    CUDA_RT_CALL( cudaMemcpy( devPtrV, v, ( size_t )( arraySizeV * sizeof( devPtrV[0] ) ), cudaMemcpyHostToDevice ) );

    /* --- GPU --- */
    /* create the analysis info (for lower and upper triangular factors) */
    size_t bufferSize_l = 0;
    size_t bufferSize_u = 0;

    void * dBuffer    = NULL;
    size_t bufferSize = 0;

    /* Calculate LOWER buffersize */
    // Create dense vector X_u
    CUSPARSE_CALL( cusparseCreateDnVec( &vecX, matrixM, devPtrP, CUDA_R_64F ) );

    // Create dense vector Y_u
    CUSPARSE_CALL( cusparseCreateDnVec( &vecY, matrixM, devPtrT, CUDA_R_64F ) );

    // Create opaque data structure, that holds analysis data between calls.
    CUSPARSE_CALL( cusparseSpSV_createDescr( &spsvDescr_l ) );

    // Specify Lower|Upper fill mode.
    fillmode = CUSPARSE_FILL_MODE_LOWER;
    CUSPARSE_CALL( cusparseSpMatSetAttribute( descrm, CUSPARSE_SPMAT_FILL_MODE, &fillmode, sizeof( fillmode ) ) );
    // Specify Unit|Non-Unit diagonal type.
    diagtype = CUSPARSE_DIAG_TYPE_UNIT;
    CUSPARSE_CALL( cusparseSpMatSetAttribute( descrm, CUSPARSE_SPMAT_DIAG_TYPE, &diagtype, sizeof( diagtype ) ) );

    // allocate an external buffer for analysis
    CUSPARSE_CALL( cusparseSpSV_bufferSize( cusparseHandle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha,
                                            descrm,
                                            vecX,
                                            vecY,
                                            CUDA_R_64F,
                                            CUSPARSE_SPSV_ALG_DEFAULT,
                                            spsvDescr_l,
                                            &bufferSize_l ) );

    /* Calculate UPPER buffersize */
    // Create dense vector X_u
    CUSPARSE_CALL( cusparseCreateDnVec( &vecX, matrixM, devPtrT, CUDA_R_64F ) );

    // Create dense vector Y_u
    CUSPARSE_CALL( cusparseCreateDnVec( &vecY, matrixM, devPtrPW, CUDA_R_64F ) );

    // Create opaque data structure, that holds analysis data between calls.
    CUSPARSE_CALL( cusparseSpSV_createDescr( &spsvDescr_u ) );

    // Specify Lower|Upper fill mode.
    fillmode = CUSPARSE_FILL_MODE_UPPER;
    CUSPARSE_CALL( cusparseSpMatSetAttribute( descrm, CUSPARSE_SPMAT_FILL_MODE, &fillmode, sizeof( fillmode ) ) );
    // Specify Unit|Non-Unit diagonal type.
    diagtype = CUSPARSE_DIAG_TYPE_NON_UNIT;
    CUSPARSE_CALL( cusparseSpMatSetAttribute( descrm, CUSPARSE_SPMAT_DIAG_TYPE, &diagtype, sizeof( diagtype ) ) );

    // allocate an external buffer for analysis
    CUSPARSE_CALL( cusparseSpSV_bufferSize( cusparseHandle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha,
                                            descrm,
                                            vecX,
                                            vecY,
                                            CUDA_R_64F,
                                            CUSPARSE_SPSV_ALG_DEFAULT,
                                            spsvDescr_u,
                                            &bufferSize_u ) );

    /* Allocate max buffersize */
    bufferSize = max( bufferSize_l, bufferSize_u );
    CUDA_RT_CALL( cudaMalloc( &dBuffer, bufferSize ) );

    /* analyse the lower and upper triangular factors */
    // double ttl = second( );
    CUDA_RT_CALL( cudaEventRecord( ttl ) );

    // CUSPARSE_CALL( cusparseSetMatFillMode( descrm, CUSPARSE_FILL_MODE_LOWER ) );
    // CUSPARSE_CALL( cusparseSetMatDiagType( descrm, CUSPARSE_DIAG_TYPE_UNIT ) );
    // checkCudaErrors( cusparseDcsrsv_analysis( cusparseHandle,
    //                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                                           matrixM,
    //                                           nnz,
    //                                           descrm,
    //                                           devPtrAval,
    //                                           devPtrArowsIndex,
    //                                           devPtrAcolsIndex,
    //                                           info_l ) );

    CUSPARSE_CALL( cusparseCreateDnVec( &vecX, matrixM, devPtrP, CUDA_R_64F ) );
    CUSPARSE_CALL( cusparseCreateDnVec( &vecY, matrixM, devPtrT, CUDA_R_64F ) );
    CUSPARSE_CALL( cusparseSpSV_analysis( cusparseHandle,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha,
                                          descrm,
                                          vecX,
                                          vecY,
                                          CUDA_R_64F,
                                          CUSPARSE_SPSV_ALG_DEFAULT,
                                          spsvDescr_l,
                                          dBuffer ) );

    // CUDA_RT_CALL( cudaDeviceSynchronize( ) );
    // double ttl2 = second( );
    CUDA_RT_CALL( cudaEventRecord( ttl2 ) );
    CUDA_RT_CALL( cudaEventSynchronize( ttl2 ) );
    // CUDA_RT_CALL( cudaEventElapsedTime( &elapsed_events_ms, start_event, stop_event ) );

    // double ttu = second( );
    CUDA_RT_CALL( cudaEventRecord( ttu ) );

    // CUSPARSE_CALL( cusparseSetMatFillMode( descrm, CUSPARSE_FILL_MODE_UPPER ) );
    // CUSPARSE_CALL( cusparseSetMatDiagType( descrm, CUSPARSE_DIAG_TYPE_NON_UNIT ) );
    // CUSPARSE_CALL( cusparseDcsrsv_analysis( cusparseHandle,
    //                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                                         matrixM,
    //                                         nnz,
    //                                         descrm,
    //                                         devPtrAval,
    //                                         devPtrArowsIndex,
    //                                         devPtrAcolsIndex,
    //                                         info_u ) );

    CUSPARSE_CALL( cusparseCreateDnVec( &vecX, matrixM, devPtrT, CUDA_R_64F ) );
    CUSPARSE_CALL( cusparseCreateDnVec( &vecY, matrixM, devPtrPW, CUDA_R_64F ) );
    CUSPARSE_CALL( cusparseSpSV_analysis( cusparseHandle,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha,
                                          descrm,
                                          vecX,
                                          vecY,
                                          CUDA_R_64F,
                                          CUSPARSE_SPSV_ALG_DEFAULT,
                                          spsvDescr_u,
                                          dBuffer ) );

    // CUDA_RT_CALL( cudaDeviceSynchronize( ) );
    // double ttu2 = second( );

    CUDA_RT_CALL( cudaEventRecord( ttu2 ) );
    CUDA_RT_CALL( cudaEventSynchronize( ttu2 ) );
    CUDA_RT_CALL( cudaEventElapsedTime( &ttl_ttl2, ttl, ttl2 ) );
    CUDA_RT_CALL( cudaEventElapsedTime( &ttu_ttu2, ttu, ttu2 ) );

    ttt_sv += ( ttl_ttl2 * 1e-3 ) + ( ttu_ttu2 * 1e-3 );
    printf( "analysis lower %f (s), upper %f (s) \n", ttl_ttl2 * 1e-3, ttu_ttu2 * 1e-3 );

    /* compute the lower and upper triangular factors using CUSPARSE csrilu0 routine (on the GPU) */
    // double start_ilu, stop_ilu;
    printf( "CUSPARSE csrilu0\n" );
    // start_ilu = second( );
    CUDA_RT_CALL( cudaEventRecord( start_ilu ) );
    // devPtrMrowsIndex = devPtrArowsIndex;
    // devPtrMcolsIndex = devPtrAcolsIndex;

    void *dBuffer_ilu    = NULL;
    int   bufferSize_ilu = 0;

    // int m = matrixM + matrixN;

    CUSPARSE_CALL( cusparseCreateCsrilu02Info( &info_M ) );
    CUSPARSE_CALL( cusparseCreateMatDescr( &descra_uli ) );
    CUSPARSE_CALL( cusparseSetMatType( descra_uli, CUSPARSE_MATRIX_TYPE_GENERAL ) );
    if ( base ) {
        CUSPARSE_CALL( cusparseSetMatIndexBase( descra_uli, CUSPARSE_INDEX_BASE_ONE ) );
    } else {
        // CUSPARSE_CALL( cusparseSetMatIndexBase( descra_uli, CUSPARSE_INDEX_BASE_ZERO ) );
    }

    CUSPARSE_CALL( cusparseDcsrilu02_bufferSize( cusparseHandle,
                                                 matrixM,
                                                 nnz,
                                                 descra_uli,
                                                 devPtrMval,
                                                 devPtrArowsIndex,
                                                 devPtrAcolsIndex,
                                                 info_M,
                                                 &bufferSize_ilu ) );

    CUDA_RT_CALL( cudaMalloc( &dBuffer_ilu, bufferSize_ilu ) );

    CUSPARSE_CALL( cusparseDcsrilu02_analysis( cusparseHandle,
                                               matrixM,
                                               nnz,
                                               descra_uli,
                                               devPtrMval,
                                               devPtrArowsIndex,
                                               devPtrAcolsIndex,
                                               info_M,
                                               CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                               dBuffer_ilu ) );

    // FIXME -- BUG?
    CUSPARSE_CALL( cusparseDcsrilu02( cusparseHandle,
                                      matrixM,
                                      nnz,
                                      descra_uli,
                                      devPtrMval,
                                      devPtrArowsIndex,
                                      devPtrAcolsIndex,
                                      info_M,
                                      CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                      dBuffer_ilu ) );

    // OLD
    // checkCudaErrors( cusparseDcsrilu0( cusparseHandle,
    //                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                                        matrixM,
    //                                        descra,
    //                                        devPtrMval,
    //                                        devPtrArowsIndex,
    //                                        devPtrAcolsIndex,
    //                                        info_l ) );

    // CUDA_RT_CALL( cudaDeviceSynchronize( ) );
    // stop_ilu = second( );
    CUDA_RT_CALL( cudaEventRecord( stop_ilu ) );
    CUDA_RT_CALL( cudaEventSynchronize( stop_ilu ) );
    CUDA_RT_CALL( cudaEventElapsedTime( &elapsed_events_ms, start_ilu, stop_ilu ) );

    fprintf( stdout, "time(s) = %10.8f \n", elapsed_events_ms * 1e-3 );

    /* run the test */
    // Note that multiple iterations don't provide correct results
    // Because of inplace writes.
    num_iterations = 1;  // 10;
    CUDA_RT_CALL( cudaEventRecord( start_event ) );
    for ( count = 0; count < num_iterations; count++ ) {

        gpu_pbicgstab( cublasHandle,
                       cusparseHandle,
                       matrixM,
                       matrixN,
                       nnz,
                       descra,
                       devPtrAval,
                       devPtrArowsIndex,
                       devPtrAcolsIndex,
                       descrm,
                       devPtrMval,
                       devPtrMrowsIndex,
                       devPtrMcolsIndex,
                       descra_uli,
                       vecX,
                       vecY,
                       spsvDescr_l,
                       spsvDescr_u,
                       fillmode,
                       diagtype,
                       info_M,
                       //    info_l,
                       //    info_u,
                       dBuffer,
                       dBuffer_ilu,
                       devPtrF,
                       devPtrR,
                       devPtrRW,
                       devPtrP,
                       devPtrPW,
                       devPtrS,
                       devPtrT,
                       devPtrV,
                       devPtrX,
                       maxit,
                       tol,
                       ttt_sv,
                       ttm,
                       ttm2,
                       ttm_ttm2,
                       ttl,
                       ttl2,
                       ttl_ttl2,
                       ttu,
                       ttu2,
                       ttu_ttu2 );

        CUDA_RT_CALL( cudaDeviceSynchronize( ) );
    }
    // stop = second( ) / num_iterations;
    CUDA_RT_CALL( cudaEventRecord( stop_event ) );
    CUDA_RT_CALL( cudaEventSynchronize( stop_event ) );
    CUDA_RT_CALL( cudaEventElapsedTime( &elapsed_events_ms, start_event, stop_event ) );

    fprintf( stdout, "Average bicgstab time(s) = %10.8f \n", ( elapsed_events_ms / num_iterations ) * 1e-3 );

    /* copy the result into host memory */
    CUDA_RT_CALL( cudaMemcpy( tx, devPtrX, ( size_t )( arraySizeX * sizeof( tx[0] ) ), cudaMemcpyDeviceToHost ) );

    /* destroy the analysis info (for lower and upper triangular factors) */
    // CUSPARSE_CALL( cusparseDestroyCsrsv2Info( info_l ) );
    // CUSPARSE_CALL( cusparseDestroyCsrsv2Info( info_u ) );
    CUSPARSE_CALL( cusparseDestroyCsrilu02Info( info_M ) );

    // CUSPARSE_CALL( cusparseDestroyMatDescr( descra ) );
    // CUSPARSE_CALL( cusparseDestroyMatDescr( descrm ) );
    CUSPARSE_CALL( cusparseDestroyMatDescr( descra_uli ) );

    CUSPARSE_CALL( cusparseDestroy( cusparseHandle ) );
    CUBLAS_CALL( cublasDestroy( cublasHandle ) );

    // CUDA_RT_CALL( cudaStreamDestroy( stream ) );

    CUDA_RT_CALL( cudaEventDestroy( start_matrix_copy ) );
    CUDA_RT_CALL( cudaEventDestroy( stop_matrix_copy ) );
    CUDA_RT_CALL( cudaEventDestroy( ttl ) );
    CUDA_RT_CALL( cudaEventDestroy( ttl2 ) );
    CUDA_RT_CALL( cudaEventDestroy( ttu ) );
    CUDA_RT_CALL( cudaEventDestroy( ttu2 ) );
    CUDA_RT_CALL( cudaEventDestroy( ttm ) );
    CUDA_RT_CALL( cudaEventDestroy( ttm2 ) );
    CUDA_RT_CALL( cudaEventDestroy( start_ilu ) );
    CUDA_RT_CALL( cudaEventDestroy( stop_ilu ) );
    CUDA_RT_CALL( cudaEventDestroy( start_event ) );
    CUDA_RT_CALL( cudaEventDestroy( stop_event ) );

    return EXIT_SUCCESS;
}

int main( int argc, char *argv[] ) {
    int   status            = EXIT_FAILURE;
    char *matrix_filename   = NULL;
    char *coloring_filename = NULL;

    int    symmetrize = 0;
    int    debug      = 0;
    int    maxit      = 2000;  // 5; //2000; //1000;  //50; //5; //50; //100; //500; //10000;
    double tol = 0.0000001;    // 0.000001; //0.00001; //0.00000001; //0.0001; //0.001; //0.00000001; //0.1; //0.001;
                               // //0.00000001;
    double damping = 0.75;

    /* WARNING: it is assumed that the matrices are stores in Matrix Market format */
    printf( "WARNING: it is assumed that the matrices are stores in Matrix Market format with double as elementtype\n "
            "Usage: ./BiCGStab -F[matrix.mtx] [-E] [-D]\n" );

    printf( "Starting [%s]\n", argv[0] );
    int i         = 0;
    int temp_argc = argc;
    while ( argc ) {
        if ( *argv[i] == '-' ) {
            switch ( *( argv[i] + 1 ) ) {
            case 'F':
                matrix_filename = argv[i] + 2;
                break;
            case 'E':
                symmetrize = 1;
                break;
            case 'D':
                debug = 1;
                break;
            case 'C':
                coloring_filename = argv[i] + 2;
                break;
            default:
                fprintf( stderr, "Unknown switch '-%s'\n", argv[i] + 1 );
                return status;
            }
        }
        argc--;
        i++;
    }

    argc = temp_argc;

    // Use default input file
    if ( matrix_filename == NULL ) {
        printf( "argv[0] = %s\n", argv[0] );
        matrix_filename = sdkFindFilePath( "gr_900_900_cfg.mtx", argv[0] );

        if ( matrix_filename != NULL ) {
            printf( "Using default input file [%s]\n", matrix_filename );
        } else {
            printf( "Could not find input file = %s\n", matrix_filename );
            return EXIT_FAILURE;
        }
    } else {
        printf( "Using input file [%s]\n", matrix_filename );
    }

    //  findCudaDevice(argc, (const char **)argv);

    status = test_bicgstab( matrix_filename,
                            coloring_filename,
                            "d",
                            symmetrize,
                            debug,
                            damping,
                            maxit,
                            tol,
                            DBICGSTAB_MAX_ULP_ERR,
                            DBICGSTAB_EPS );

    return status;
}
