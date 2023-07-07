#include <stan/model/model_header.hpp>
#include <Eigen/core>
#include <stan/math.hpp>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <windows.h>
#include <fstream>
#include <string>
#include <iomanip>
#include <stan/math/rev/fun.hpp>
#include <mutex>
#include <processthreadsapi.h>
#define DLLIMPORT __declspec(dllimport)
DLLIMPORT void likelihood(double* d_X_sa, double* d_X_sm, double* d_beta, double* d_trend, double* result, int num_rows, int num_cols);
DLLIMPORT void linear_trendh(double k, double m, double* delta_ptr, double* t_ptr, double* A_ptr, double* t_change_ptr, double* result_ptr, int T, int S, double* delta_val_1, double* delta_val_2);
DLLIMPORT void elementwize(double* d_X, double* d_s_a, double* d_result, int size);
DLLIMPORT void chek_pont(double* d_tz, double* d_t_change, double* d_A, int T, int S);
#ifndef STAN_CUDA_DLL
#define STAN_CUDA_DLL
  HMODULE cudaDll = LoadLibrary("cuda_func_help.dll");
#endif  
int CSV_NUM = 0;
Eigen::Map<Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, 1>, 0, Eigen::Stride<0, 0>> modeld_g(const Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, 1>& beta, const Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, Eigen::Dynamic>& X_sa, const Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, Eigen::Dynamic>& X_sm, const Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, 1>& trend) {
  // Summary of equation: α + x⋅β || α = (trend .* (1 + X_sm * beta)) || x = X_sa || β = beta
  void (*likelihood_g)(double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, int, int) = (void (*)(double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, int, int))GetProcAddress(cudaDll, "likelihood_g");
  const int T = X_sa.rows();
  const int K = beta.rows();
  // Allocate device memory
  double* d_X_sa; //size = T*K
  double* d_X_sm; //size = T*K
  double* d_beta; //size = K
  double* d_trend; //size = T
  double* result_ptr;
  double* d_g_X_sa; //size = T*K
  double* d_g_X_sm; //size = T*K
  double* d_g_beta; //size = K
  double* d_g_trend; //size = T
  double* g_result_ptr;
  cudaMalloc((void**)&d_X_sa, T * K * sizeof(double));
  cudaMalloc((void**)&d_X_sm, T * K * sizeof(double));
  cudaMalloc((void**)&d_beta, K * sizeof(double));
  cudaMalloc((void**)&d_trend, T * sizeof(double));
  cudaMalloc((void**)&result_ptr, T * sizeof(double));
  cudaMalloc((void**)&d_g_X_sa, T * K * sizeof(double));
  cudaMalloc((void**)&d_g_X_sm, T * K * sizeof(double));
  cudaMalloc((void**)&d_g_beta, K * sizeof(double));
  cudaMalloc((void**)&d_g_trend, T * sizeof(double));
  cudaMalloc((void**)&g_result_ptr, T * sizeof(double));

  // Copy data from host to device
  double* X_sm_arr = new double [T * K];
  double* X_sa_arr = new double [T * K];
  double* beta_arr = new double [K];
  double* trend_arr = new double [T];
  double* g_X_sm_arr = new double [T * K];
  double* g_X_sa_arr = new double [T * K];
  double* g_beta_arr = new double [K];
  double* g_trend_arr = new double [T];
  for (int i = 0; i < T; i++) {
    for (int j = 0; j < K; j++) {
      X_sa_arr[i * K + j] = X_sa(i, j).val();
      X_sm_arr[i * K + j] = X_sm(i, j).val();
      g_X_sa_arr[i * K + j] = X_sa(i, j).adj();
      g_X_sm_arr[i * K + j] = X_sm(i, j).adj();
    }
    trend_arr[i] = trend(i, 0).val();
    g_trend_arr[i] = trend(i, 0).adj();
  }
  for (int i = 0; i < K; i++) {
    beta_arr[i] = beta(i, 0).val();
    g_beta_arr[i] = beta(i, 0).adj();
  }
  // Copy data from host to device
  cudaMemcpy(d_X_sa, X_sa_arr, T * K * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_X_sm, X_sm_arr, T * K * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_beta, beta_arr, K * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_trend, trend_arr, T * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_g_X_sa, g_X_sa_arr, T * K * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_g_X_sm, g_X_sm_arr, T * K * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_g_beta, g_beta_arr, K * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_g_trend, g_trend_arr, T * sizeof(double), cudaMemcpyHostToDevice);

  // Launch the likelihood kernel
  likelihood_g(d_X_sa, d_X_sm, d_beta, d_trend, result_ptr, d_g_X_sa, d_g_X_sm, d_g_beta, d_g_trend, g_result_ptr, T, K);
  cudaDeviceSynchronize();
  // Copy the results back to the host.
  double* temp = new double [T];
  cudaMemcpy(temp, result_ptr, T * sizeof(double), cudaMemcpyDeviceToHost);
  double* g_temp = new double [T];
  cudaMemcpy(g_temp, g_result_ptr, T * sizeof(double), cudaMemcpyDeviceToHost);
  stan::math::var_value<double>* result_arrayd = new stan::math::var_value<double> [T];
  stan::math::var_value<double> res;
  for (int i = 0; i < T; i++){
    res = temp[i];
    res.adj() = g_temp[i];
    result_arrayd[i] = res;
  }
  Eigen::Map<Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, 1>, 0, Eigen::Stride<0, 0>> to_normal = Eigen::Map<Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, 1>, 0, Eigen::Stride<0, 0>>(result_arrayd, T, 1);
  cudaFree(0);
  cudaFree(d_X_sa);
  cudaFree(d_X_sm);
  cudaFree(d_beta);
  cudaFree(d_trend);
  delete[] X_sa_arr;
  delete[] X_sm_arr;
  delete[] beta_arr;
  delete[] trend_arr;
  return to_normal;
}
Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>, 0, Eigen::Stride<0, 0>> modeld(const Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, 1>& beta, const Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, Eigen::Dynamic>& X_sa, const Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, Eigen::Dynamic>& X_sm, const Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, 1>& trend) {
  // Summary of equation: α + x⋅β || α = (trend .* (1 + X_sm * beta)) || x = X_sa || β = beta
  void (*likelihood)(double*, double*, double*, double*, double*, int, int) = (void (*)(double*, double*, double*, double*, double*, int, int))GetProcAddress(cudaDll, "likelihood");
  const int T = X_sa.rows();
  const int K = beta.rows();
  // Allocate device memory
  double* d_X_sa; //size = T*K
  double* d_X_sm; //size = T*K
  double* d_beta; //size = K
  double* d_trend; //size = T
  double* result_ptr;
  
  cudaMalloc((void**)&d_X_sa, T * K * sizeof(double));
  cudaMalloc((void**)&d_X_sm, T * K * sizeof(double));
  cudaMalloc((void**)&d_beta, K * sizeof(double));
  cudaMalloc((void**)&d_trend, T * sizeof(double));
  cudaMalloc((void**)&result_ptr, T * sizeof(double));

  // Copy data from host to device
  double* X_sm_arr = new double [T * K];
  double* X_sa_arr = new double [T * K];
  double* beta_arr = new double [K];
  double* trend_arr = new double [T];

  for (int i = 0; i < T; i++) {
    for (int j = 0; j < K; j++) {
      X_sa_arr[i * K + j] = X_sa(i, j).val();
      X_sm_arr[i * K + j] = X_sm(i, j).val();
    }
    trend_arr[i] = trend(i, 0).val();
  }
  for (int i = 0; i < K; i++) {
    beta_arr[i] = beta(i, 0).val();
  }
  // Copy data from host to device
  cudaMemcpy(d_X_sa, X_sa_arr, T * K * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_X_sm, X_sm_arr, T * K * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_beta, beta_arr, K * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_trend, trend_arr, T * sizeof(double), cudaMemcpyHostToDevice);
  // Launch the likelihood kernel
  likelihood(d_X_sa, d_X_sm, d_beta, d_trend, result_ptr, T, K);
  cudaDeviceSynchronize();
  // Copy the results back to the host.
  double* temp = new double [T];
  cudaMemcpy(temp, result_ptr, T * sizeof(double), cudaMemcpyDeviceToHost);
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>, 0, Eigen::Stride<0, 0>> to_normal = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>, 0, Eigen::Stride<0, 0>>(temp, T, 1);
  cudaFree(0);
  cudaFree(d_X_sa);
  cudaFree(d_X_sm);
  cudaFree(d_beta);
  cudaFree(d_trend);
  delete[] X_sa_arr;
  delete[] X_sm_arr;
  delete[] beta_arr;
  delete[] trend_arr;
  return to_normal;
}
void linear_trendd(Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, 1> mine, Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, 1> stan){
    std::cout << "linear_trendd" << std::endl;
    const int T = stan.rows();
    for (int i = 0; i < T; i++) {
        if (mine(i, 0).val() != stan(i, 0).val()) {
            std::cout << "mine: " << mine(i,0).val() << " stan: " << stan(i,0).val() << std::endl;
        }
    }
    
    /*
    void (*linear_trendh)(double, double, double*, double*, double*, double*, double*, int, int, double*, double*) = (void (*)(double, double, double*, double*, double*, double*, double*, int, int, double*, double*))GetProcAddress(cudaDll, "linear_trendh");    
    //cudaFree(0);
    int T = t.rows(); //A rows
    int S = delta.rows();// A cols
    const int T_C = T;
    const int S_C = S;
    double* delta_ptr;
    double* t_ptr;
    double* A_ptr;
    double* t_change_ptr;
    double* result_ptr;
    double* d_v_1ptr;
    double* d_v_2ptr;
    cudaMalloc((void**)&delta_ptr, S * sizeof(double));
    cudaMalloc((void**)&t_ptr, T * sizeof(double));
    cudaMalloc((void**)&d_v_1ptr, T * sizeof(double));
    cudaMalloc((void**)&d_v_2ptr, T * sizeof(double));
    cudaMalloc((void**)&A_ptr, T * S * sizeof(double));
    cudaMalloc((void**)&t_change_ptr, S * sizeof(double));
    cudaMalloc((void**)&result_ptr, T * sizeof(double));
    double* t_arr = new double [T_C];
    double* t_c_arr = new double [S_C];
    double* A_arr = new double [T_C * S_C];;
    double* delt_arr = new double [S_C];
    double* result = new double [T_C];
    for (int j = 0; j < T_C; j++) {
        t_arr[j] = t(j, 0).val();
    }
    for (int i=0; i < T_C; i++) {
        for (int j = 0; j < S_C; j++) {
            A_arr[i * S_C + j] = A(i, j).val(); //i = rows, j = cols
        }
    }
    for (int j = 0; j < S_C; j++) {
        delt_arr[j] = delta(j, 0).val();
    }
    for (int j = 0; j < S_C; j++) {
        t_c_arr[j] = t_change(j, 0).val();
    }
    double k_val = k.val();
    double m_val = m.val();
    cudaMemcpy(delta_ptr, delt_arr, S * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(t_ptr, t_arr, T * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(A_ptr, A_arr, T * S * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(t_change_ptr, t_c_arr, S * sizeof(double), cudaMemcpyHostToDevice);
    //run kernel
    linear_trendh(k_val, m_val, delta_ptr, t_ptr, A_ptr, t_change_ptr, result_ptr, T, S, d_v_1ptr, d_v_2ptr);
    cudaDeviceSynchronize();
    // Copy the results back to the host.
    double* temp = new double [T_C];

    cudaMemcpy(temp, result_ptr, T * sizeof(double), cudaMemcpyDeviceToHost);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>, 0, Eigen::Stride<0, 0>> trendz = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>, 0, Eigen::Stride<0, 0>>(temp, T_C, 1);
    // Free device memory

    cudaFree(0);
    cudaFree(delta_ptr);
    cudaFree(t_ptr);
    cudaFree(A_ptr);
    cudaFree(t_change_ptr);
    cudaFree(result_ptr);
    delete[] t_arr;
    delete[] t_c_arr;
    delete[] A_arr;
    delete[] delt_arr;
    return trendz;*/
  
}
Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::Stride<0, 0>> Xmaker(Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, Eigen::Dynamic>& X, Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, 1>& s_a, const int& T) {
  void (*elementwize)(double*, double*, double*, int) = (void (*)(double*, double*, double*, int))GetProcAddress(cudaDll, "elementwize");
  const int size = X.size();
  const int T_C = T;
  const int K_C = X.cols();
  // Allocate device memory
  double* d_X;
  double* d_s_a;
  double* d_result;
  //std::cout << "Size needed in kb:" << ((T_C * K_C * sizeof(double) + K_C * sizeof(double) + T_C * K_C * sizeof(double)) / 1024) << std::endl;  
  cudaMalloc((void**)&d_X, T_C * K_C * sizeof(double));
  cudaMalloc((void**)&d_s_a, T_C * K_C * sizeof(double));
  cudaMalloc((void**)&d_result, T_C * K_C * sizeof(double));
  // Copy data from host to device
  double* X_arr = new double [T_C * K_C];
  double* s_a_arr = new double [T_C * K_C];
    // Initialize the array.
  for (int i = 0; i < T_C; i++) {
      for (int j = 0; j < K_C; j++) {
          X_arr[i * K_C + j] = X(i, j).val();
      }
  }
    // Initialize the array.
  for (int i = 0; i < T_C; i++) {
    for (int j = 0; j < K_C; j++) {
      s_a_arr[i * K_C + j] = s_a(j, 0).val();
    }
  }
  cudaMemcpy(d_X, X_arr, T_C * K_C * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_s_a, s_a_arr, K_C * sizeof(double), cudaMemcpyHostToDevice);
  // Launch the element-wise multiplication kernel
  elementwize(d_X, d_s_a, d_result, (T_C * K_C));
  
  // Map the device memory to Eigen::Matrix
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> x_x(T_C, K_C);
  cudaMemcpy(x_x.data(), d_result, (T_C * K_C), cudaMemcpyDeviceToHost);
  double* r_data = new double [T_C * K_C];
  //switch from rowwise to colwise because of eigen
  for (int i = 0; i < T_C; i++) {
    for (int j = 0; j < K_C; j++) {
      r_data[i * K_C + j] = x_x(i, j);
    }
  }
  // Copy the result back to the host
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::Stride<0, 0>> resultHost = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::Stride<0, 0>>(r_data, T_C, K_C);
  // Free device memory
  cudaFree(d_X);
  cudaFree(d_s_a);
  cudaFree(d_result);
  delete[] s_a_arr;
  delete[] X_arr;
  return resultHost;
}
Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::Stride<0, 0>> get_changepoint_matrixx(Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, 1>& t, Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, 1>& t_change, const int& T, const int& S){
  void (*chek_pont)(double*, double*, double*, int, int) = (void (*)(double*, double*, double*, int, int))GetProcAddress(cudaDll, "chek_pont");  
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A(T, S);
  const int T_C = T;
  const int S_C = S;
  // Allocate device memory
  double* d_tz;
  double* d_t_change;
  double* d_A;
  //std::cout << "Size needed in kb:" << ((T_C * sizeof(double) + S_C * sizeof(double) + S_C * sizeof(double))/1024) << std::endl;
  cudaMalloc((void**)&d_tz, T_C * sizeof(double));
  cudaMalloc((void**)&d_t_change, S_C * sizeof(double));
  cudaMalloc((void**)&d_A, T_C * S_C * sizeof(double));
  //maybe can get rid of this
  double* t_arr = new double [T_C];
  double* t_c_arr = new double [S_C];
    // Initialize the array.
  for (int j = 0; j < T_C; j++) {
      t_arr[j] = t(j, 0).val();
  }
     // Initialize the array.
  for (int j = 0; j < S_C; j++) {
      t_c_arr[j] = t_change(j, 0).val();
  }
  // Copy data from host to device
  cudaMemcpy(d_tz, t_arr, T_C * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_t_change, t_c_arr, S_C * sizeof(double), cudaMemcpyHostToDevice);
  // Launch the kernel
  chek_pont(d_tz, d_t_change, d_A, T_C, S_C);
  double* temp = new double [T_C * S_C];

  // Copy result from device to host
  cudaMemcpy(temp, d_A, T_C * S_C * sizeof(double), cudaMemcpyDeviceToHost);
  
  double* col_wise_data = new double [T_C * S_C];
  //switch from rowwise to colwise because of eigen
  for (int i = 0; i < T_C; i++) {
    for (int j = 0; j < S_C; j++) {
      col_wise_data[j * T_C + i] = temp[i * S_C + j];
    }
  }
  // Map the device memory to Eigen::Matrix 
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::Stride<0, 0>> Ab = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::Stride<0, 0>>(col_wise_data, T_C, S_C);
  // Free device memory

  cudaFree(d_tz);
  cudaFree(d_t_change);
  cudaFree(d_A);
  delete[] t_arr;
  delete[] t_c_arr;
  delete[] temp;
  return Ab;
}

namespace prophet_model_namespace {
  //Eigen::Map<Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, 1>, 0, Eigen::Stride<0, 0>> cuStan_model(const Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, 1>& beta, const Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, Eigen::Dynamic>& X_sa, const Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, Eigen::Dynamic>& X_sm, const Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, 1>& trend, std::ostream *pstream__ = nullptr) {
      //return Eigen::Map<Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, 1>, 0, Eigen::Stride<0, 0>>(beta.data(), beta.rows(), 1);
    //research auto diff, gradient, hessian, wrt, log prob, & var(new precomp_v_vari(f, x.vi_, dfdx_))
    //}
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>, 0, Eigen::Stride<0, 0>> cuStan_model(const Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, 1>& beta, const Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, Eigen::Dynamic>& X_sa, const Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, Eigen::Dynamic>& X_sm, const Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, 1>& trend, std::ostream *pstream__ = nullptr) {
    return modeld(beta, X_sa, X_sm, trend);
  }
  //template <typename T>
  //T cuStan_model(const Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, 1>& beta, const Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, Eigen::Dynamic>& X_sa, const Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, Eigen::Dynamic>& X_sm, const Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, 1>& trend, std::ostream *pstream__ = nullptr) {
   // if (std::is_same<T, Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>, 0, Eigen::Stride<0, 0>>>::value )
     // return modeld(beta, X_sa, X_sm, trend);
    //else
      //return modeld_g(beta, X_sa, X_sm, trend);
  //}
  void cuStan_lt(Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, 1> mine, Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, 1> stan, std::ostream *pstream__ = nullptr) {
      linear_trendd(mine, stan);
    }
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::Stride<0, 0>> cuStan_chp_mat(Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, 1>&& t, Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, 1>&& t_change, const int& T, const int& S, std::ostream *pstream__ = nullptr) {
      return get_changepoint_matrixx(t, t_change, T, S);
    }

  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::Stride<0, 0>> cuStan_xmake(Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, Eigen::Dynamic>&& X, Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, 1>&& s_a, const int& T, std::ostream *pstream__ = nullptr) {
      return Xmaker(X, s_a, T);
    }
}



