//  Library File for having functions avaliable
//  to the user header.
//  By Jonathan Stoff
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#define DLLEXPORT __declspec(dllexport)
//To be added
__global__ void likelihood_fwgrad(double* X_sm, double* X_sa, double* beta, double* trend, double* result, double* g_X_sm, double* g_X_sa, double* g_beta, double* g_trend, double* g_result, int T, int K) {
    // α + x⋅β || α=trend .* (1 + X_sm * beta), x=X_sa, β=beta
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T) {
        double sum = 0.0;
        double dum = 0.0;
        double g_sum = 0.0;
        double g_dum = 0.0;
        for (int j = 0; j < K; ++j) {
            sum += X_sm[i * K + j] * beta[j];
            dum += X_sa[i * K + j] * beta[j];
            g_sum += X_sm[i * K + j] * g_beta[j] - g_X_sm[i * K + j] * beta[j];
            g_dum += X_sa[i * K + j] * g_beta[j] - g_X_sa[i * K + j] * beta[j];
        }
        __syncthreads();
        result[i] = (trend[i] * (sum+1)) + dum;
        g_result[i] = ((trend[i] * (g_sum+1)) - (g_trend[i] * (sum+1))) + g_dum;
        
    }
}
//To be added
__global__ void likelihood_all(double* X_sm, double* X_sa, double* beta, double* trend, double* result, int T, int K) {
    // α + x⋅β || α=trend .* (1 + X_sm * beta), x=X_sa, β=beta
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T) {
        double sum = 0.0;
        double dum = 0.0;
        for (int j = 0; j < K; ++j) {
            sum += X_sm[i * K + j] * beta[j];
            dum += X_sa[i * K + j] * beta[j];
        }
        __syncthreads();
        result[i] = (trend[i] * (sum+1)) + dum;
        
    }
}
//To be added
__global__ void linear_trend_kernel(double k, double m, double* delta, double* t, double* A, double* t_change, double* result, int T, int S, double* delta_val_1, double* delta_val_2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T) {
        double delta_val_11 = 0.0;
        double delta_val_22 = 0.0;
        int indexA = 0;
        for (int j = 0; j < S; j++) {
            indexA = A[i * S + j];
            if (indexA == 1) {
                delta_val_11 += (indexA * delta[j]);
                delta_val_22 += (indexA * (delta[j] * (-t_change[j])));
            }
        }
        result[i] = ((k + delta_val_11) * t[i]) + (m + delta_val_22);
        __syncthreads();

    }
}
//To be added
__global__ void linear_trend_kernel_help(double k, double m, double* delta, double* t, double* A, double* t_change, double* result, int T, int S, double* delta_val_1, double* delta_val_2) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < (T * S)) {
        int i = j / S;
        int h = j-i*S;
        delta_val_1[i] = 0.0;
        delta_val_2[i] = 0.0;
        delta_val_1[i] += A[j] * delta[h];
        delta_val_2[i] += A[j] * (delta[h] * (-t_change[h]));
        if (i>=T){
            printf("i: %i", i);
        }
        if (h>=S){
            printf("h: %i", h);
        }
        __syncthreads();
    }
}
//  Replaces the Changepoint function in the
//  .Stan File.
__global__ void get_changepoint_matrix_kernel(double* t, double* t_change, double* A, int T, int S, double* new_row) {
    //Get the interation
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //Reduce Errors
    if (i < T) {
        //reset the i item in matrix A to 0
        A[i] = 0.0;
        //Create a shared index
        __shared__ int cp_idx;
        cp_idx = 1;
        //If the conditions continue to be met,
        //set the index to be 1.
        while ((cp_idx <= S) && (t[i] >= t_change[cp_idx])) {
            A[i * S + cp_idx] = 1.0;
            cp_idx++;
            //Make sure to sync so that there arent issues
            //with race conditions.
            __syncthreads();
        }
    }

}
// Essentially a Matrix multiplication.
__global__ void elementwise_mult_kernel(double* a, double* b, double* c, int N) {
    // Get the index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Error check
    if (i < N) {
        // Multiply to make new matrix
        c[i] = a[i] * b[i];
        //Make sure to sync so that there arent issues
        //with race conditions.
        __syncthreads();
    }
}



extern "C" DLLEXPORT void likelihood(double* d_X_sa, double* d_X_sm, double* d_beta, double* d_trend, double* result, int T, int K) {
    // Set CUDA grid and block dimensions
    int threads_per_block = 512;
    int num_blocks = T/threads_per_block + (T % threads_per_block == 0 ? 0:1);
    likelihood_all << <num_blocks, threads_per_block >> > (d_X_sm, d_X_sa, d_beta, d_trend, result, T, K);
}
extern "C" DLLEXPORT void likelihood_g(double* d_X_sa, double* d_X_sm, double* d_beta, double* d_trend, double* result, double* d_g_X_sa, double* d_g_X_sm, double* d_g_beta, double* d_g_trend, double* g_result, int T, int K) {
    // Set CUDA grid and block dimensions
    int threads_per_block = 512;
    int num_blocks = T/threads_per_block + (T % threads_per_block == 0 ? 0:1);
    likelihood_fwgrad << <num_blocks, threads_per_block >> > (d_X_sm, d_X_sa, d_beta, d_trend, result, d_g_X_sm, d_g_X_sa, d_g_beta, d_g_trend, g_result, T, K);
    
}
extern "C" DLLEXPORT void linear_trendh(double k, double m, double* delta_ptr, double* t_ptr, double* A_ptr, double* t_change_ptr, double* result_ptr, int T, int S, double* delta_val_1, double* delta_val_2) {
    int threads_per_block = 512;
    int num_blocks = T/threads_per_block + (T % threads_per_block == 0 ? 0:1);
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaDeviceSynchronize();
    linear_trend_kernel <<<num_blocks, threads_per_block, 0, stream1 >>>(k, m, delta_ptr, t_ptr, A_ptr, t_change_ptr, result_ptr, T, S, delta_val_1, delta_val_2);
    cudaStreamSynchronize(stream1);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
    // Destroy the streams
    cudaStreamDestroy(stream1);
}
extern "C" DLLEXPORT void elementwize(double* d_X, double* d_s_a, double* d_result, int size) {
    // Set CUDA grid and block dimensions
    int threads_per_block = 256;
    int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    elementwise_mult_kernel << <num_blocks, threads_per_block >> > (d_X, d_s_a, d_result, size);
}
extern "C" DLLEXPORT void chek_pont(double* d_tz, double* d_t_change, double* d_A, int T, int S) {
    int threads_per_block = 256;
    int num_blocks = (T + threads_per_block - 1) / threads_per_block;
    double* tenmp;
    cudaMalloc((void**)&tenmp, S * sizeof(double));
    get_changepoint_matrix_kernel << <num_blocks, threads_per_block >> > (d_tz, d_t_change, d_A, T, S, tenmp);
    cudaFree(tenmp);
}

