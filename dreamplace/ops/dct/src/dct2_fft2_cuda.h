/**
 * @file   dct2_fft2_cuda.h
 * @author Zixuan Jiang, Jiaqi Gu
 * @date   Apr 2019
 * @brief  All the transforms in this file are implemented based on 2D FFT.
 *      Each transfrom has three steps, 1) preprocess, 2) 2d fft or 2d ifft, 3)
 * postprocess.
 */

#ifndef DREAMPLACE_DCT2_FFT2_CUDA_H
#define DREAMPLACE_DCT2_FFT2_CUDA_H

#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

// dct2_fft2
void dct2_fft2_forward(at::Tensor x, at::Tensor expkM, at::Tensor expkN,
                       at::Tensor out, at::Tensor buf);
void dct2_fft2_scaled_forward(at::Tensor x, double scale, at::Tensor expkM,
                              at::Tensor expkN, at::Tensor out,
                              at::Tensor buf);

template <typename T>
void dct2dPreprocessCudaLauncher(const T *x, T *y, const int M, const int N,
                                 const T input_scale);

template <typename T>
void dct2dPostprocessCudaLauncher(const T *x, T *y, const int M, const int N,
                                  const T *__restrict__ expkM,
                                  const T *__restrict__ expkN);

// idct2_fft2
void idct2_fft2_forward(at::Tensor x, at::Tensor expkM, at::Tensor expkN,
                        at::Tensor out, at::Tensor buf);

template <typename T>
void idct2_fft2PreprocessCudaLauncher(const T *x, T *y, const int M,
                                      const int N, const T *__restrict__ expkM,
                                      const T *__restrict__ expkN);

template <typename T>
void idct2_fft2PostprocessCudaLauncher(const T *x, T *y, const int M,
                                       const int N,
                                       const T normalization_scale);

// idct_idxst
void idct_idxst_forward(at::Tensor x, at::Tensor expkM, at::Tensor expkN,
                        at::Tensor out, at::Tensor buf);
void idct_idxst_weighted_forward(at::Tensor x, at::Tensor weight,
                                 at::Tensor expkM, at::Tensor expkN,
                                 at::Tensor out, at::Tensor buf);

template <typename T>
void idct_idxstPreprocessCudaLauncher(
    const T *x, const T *weight, T *y, const int M, const int N,
    const T *__restrict__ expkM, const T *__restrict__ expkN);

template <typename T>
void idct_idxstPostprocessCudaLauncher(const T *x, T *y, const int M,
                                       const int N,
                                       const T normalization_scale);

// idxst_idct
void idxst_idct_forward(at::Tensor x, at::Tensor expkM, at::Tensor expkN,
                        at::Tensor out, at::Tensor buf);
void idxst_idct_weighted_forward(at::Tensor x, at::Tensor weight,
                                 at::Tensor expkM, at::Tensor expkN,
                                 at::Tensor out, at::Tensor buf);

template <typename T>
void idxst_idctPreprocessCudaLauncher(
    const T *x, const T *weight, T *y, const int M, const int N,
    const T *__restrict__ expkM, const T *__restrict__ expkN);

template <typename T>
void idxst_idctPostprocessCudaLauncher(const T *x, T *y, const int M,
                                       const int N,
                                       const T normalization_scale);

DREAMPLACE_END_NAMESPACE

#endif
