#include <torch/extension.h>

void moe_grouped_mm_nt_xe20_mxfp4_w4a16(
    torch::Tensor& output,
    const torch::Tensor& activations,
    const torch::Tensor& packed_weights,
    const torch::Tensor& scales,
    const std::optional<torch::Tensor>& bias,
    const torch::Tensor& total_rows_for_experts,
    int64_t n_experts,
    int64_t activation_type,
    bool fuse_act,
    double gemm1_alpha,
    double gemm1_limit);

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def(
      "moe_grouped_mm_nt_xe20_mxfp4_w4a16(Tensor! output, Tensor activations, "
      "Tensor packed_weights, Tensor scales, Tensor? bias, "
      "Tensor total_rows_for_experts, int n_experts, int activation_type, "
      "bool fuse_act, float gemm1_alpha=1.702, float gemm1_limit=7.0) -> ()");
  m.impl(
      "moe_grouped_mm_nt_xe20_mxfp4_w4a16",
      torch::kXPU,
      &moe_grouped_mm_nt_xe20_mxfp4_w4a16);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
