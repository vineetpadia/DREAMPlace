/**
 * @file   dct2_fft2_cuda.cpp
 * @author Zixuan Jiang, Jiaqi Gu
 * @date   Apr 2019
 * @brief  All the transforms in this file are implemented based on 2D FFT.
 *      Each transfrom has three steps, 1) preprocess, 2) 2d fft or 2d ifft, 3)
 * postprocess.
 */

#include "dct2_fft2_cuda.h"

DREAMPLACE_BEGIN_NAMESPACE

void dct2_fft2_forward(at::Tensor x, at::Tensor expkM, at::Tensor expkN,
                       at::Tensor out, at::Tensor buf) {
  CHECK_CUDA(x);
  CHECK_CUDA(expkM);
  CHECK_CUDA(expkN);
  CHECK_CUDA(out);
  CHECK_CUDA(buf);

  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(expkM);
  CHECK_CONTIGUOUS(expkN);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(buf);

  auto N = x.size(-1);
  auto M = x.numel() / N;

  DREAMPLACE_DISPATCH_FLOATING_TYPES(x, "dct2_fft2_forward", [&] {
    dct2dPreprocessCudaLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(x, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(out, scalar_t), M, N);

#if TORCH_VERSION_MAJOR > 1 || \
    (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 8)
    // The compatibility rfft wrapper materializes a real view of the complex
    // result with contiguous(), even though the caller already owns a buffer
    // with the required layout.  Write the FFT directly into that buffer.
    auto complex_buf = at::view_as_complex(buf);
    at::fft_rfft2_out(complex_buf, out, c10::nullopt, {-2, -1}, "backward");
#else
    buf = at::rfft(out, 2, false, true);
#endif

    dct2dPostprocessCudaLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(buf, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(out, scalar_t), M, N,
        DREAMPLACE_TENSOR_DATA_PTR(expkM, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(expkN, scalar_t));
  });
}

void idct2_fft2_forward(at::Tensor x, at::Tensor expkM, at::Tensor expkN,
                        at::Tensor out, at::Tensor buf) {
  CHECK_CUDA(x);
  CHECK_CUDA(expkM);
  CHECK_CUDA(expkN);
  CHECK_CUDA(out);
  CHECK_CUDA(buf);

  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(expkM);
  CHECK_CONTIGUOUS(expkN);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(buf);

  auto N = x.size(-1);
  auto M = x.numel() / N;

  DREAMPLACE_DISPATCH_FLOATING_TYPES(x, "idct2_fft2_forward", [&] {
    idct2_fft2PreprocessCudaLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(x, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(buf, scalar_t), M, N,
        DREAMPLACE_TENSOR_DATA_PTR(expkM, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(expkN, scalar_t));

    auto y = at::irfft(buf, 2, false, true, {{M, N}});

    idct2_fft2PostprocessCudaLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(y, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(out, scalar_t), M, N);
  });
}

void idct_idxst_forward_impl(at::Tensor x, at::Tensor weight,
                             at::Tensor expkM, at::Tensor expkN,
                             at::Tensor out, at::Tensor buf) {
  CHECK_CUDA(x);
  if (weight.defined()) {
    CHECK_CUDA(weight);
    CHECK_CONTIGUOUS(weight);
    TORCH_CHECK(weight.sizes() == x.sizes(),
                "weight must have the same shape as x");
    TORCH_CHECK(weight.scalar_type() == x.scalar_type(),
                "weight must have the same dtype as x");
  }
  CHECK_CUDA(expkM);
  CHECK_CUDA(expkN);
  CHECK_CUDA(out);
  CHECK_CUDA(buf);

  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(expkM);
  CHECK_CONTIGUOUS(expkN);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(buf);

  auto N = x.size(-1);
  auto M = x.numel() / N;

  DREAMPLACE_DISPATCH_FLOATING_TYPES(x, "idct_idxst_forward", [&] {
    idct_idxstPreprocessCudaLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(x, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(weight, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(buf, scalar_t), M, N,
        DREAMPLACE_TENSOR_DATA_PTR(expkM, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(expkN, scalar_t));

    auto y = at::irfft(buf, 2, false, true, {{M, N}});

    idct_idxstPostprocessCudaLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(y, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(out, scalar_t), M, N);
  });
}

void idct_idxst_forward(at::Tensor x, at::Tensor expkM, at::Tensor expkN,
                        at::Tensor out, at::Tensor buf) {
  idct_idxst_forward_impl(x, at::Tensor(), expkM, expkN, out, buf);
}

void idct_idxst_weighted_forward(at::Tensor x, at::Tensor weight,
                                 at::Tensor expkM, at::Tensor expkN,
                                 at::Tensor out, at::Tensor buf) {
  idct_idxst_forward_impl(x, weight, expkM, expkN, out, buf);
}

void idxst_idct_forward_impl(at::Tensor x, at::Tensor weight,
                             at::Tensor expkM, at::Tensor expkN,
                             at::Tensor out, at::Tensor buf) {
  CHECK_CUDA(x);
  if (weight.defined()) {
    CHECK_CUDA(weight);
    CHECK_CONTIGUOUS(weight);
    TORCH_CHECK(weight.sizes() == x.sizes(),
                "weight must have the same shape as x");
    TORCH_CHECK(weight.scalar_type() == x.scalar_type(),
                "weight must have the same dtype as x");
  }
  CHECK_CUDA(expkM);
  CHECK_CUDA(expkN);
  CHECK_CUDA(out);
  CHECK_CUDA(buf);

  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(expkM);
  CHECK_CONTIGUOUS(expkN);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(buf);

  auto N = x.size(-1);
  auto M = x.numel() / N;

  DREAMPLACE_DISPATCH_FLOATING_TYPES(x, "idxst_idct_forward", [&] {
    idxst_idctPreprocessCudaLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(x, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(weight, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(buf, scalar_t), M, N,
        DREAMPLACE_TENSOR_DATA_PTR(expkM, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(expkN, scalar_t));

    auto y = at::irfft(buf, 2, false, true, {{M, N}});

    idxst_idctPostprocessCudaLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(y, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(out, scalar_t), M, N);
  });
}

void idxst_idct_forward(at::Tensor x, at::Tensor expkM, at::Tensor expkN,
                        at::Tensor out, at::Tensor buf) {
  idxst_idct_forward_impl(x, at::Tensor(), expkM, expkN, out, buf);
}

void idxst_idct_weighted_forward(at::Tensor x, at::Tensor weight,
                                 at::Tensor expkM, at::Tensor expkN,
                                 at::Tensor out, at::Tensor buf) {
  idxst_idct_forward_impl(x, weight, expkM, expkN, out, buf);
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dct2_fft2", &DREAMPLACE_NAMESPACE::dct2_fft2_forward,
        "DCT2 FFT2D (CUDA)");
  m.def("idct2_fft2", &DREAMPLACE_NAMESPACE::idct2_fft2_forward,
        "IDCT2 FFT2D (CUDA)");
  m.def("idct_idxst", &DREAMPLACE_NAMESPACE::idct_idxst_forward,
        "IDCT IDXST FFT2D (CUDA)");
  m.def("idct_idxst_weighted",
        &DREAMPLACE_NAMESPACE::idct_idxst_weighted_forward,
        "Weighted IDCT IDXST FFT2D (CUDA)");
  m.def("idxst_idct", &DREAMPLACE_NAMESPACE::idxst_idct_forward,
        "IDXST IDCT FFT2D (CUDA)");
  m.def("idxst_idct_weighted",
        &DREAMPLACE_NAMESPACE::idxst_idct_weighted_forward,
        "Weighted IDXST IDCT FFT2D (CUDA)");
}
