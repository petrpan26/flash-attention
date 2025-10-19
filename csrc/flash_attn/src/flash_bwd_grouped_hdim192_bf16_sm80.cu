/*******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Grouped flash attention backward kernel instantiation for bf16, headdim 192, sm80+
 ******************************************************************************/

#include "flash_bwd_launch_template.h"

namespace FLASH_NAMESPACE {

template<>
void run_mha_bwd_grouped_<cutlass::bfloat16_t, 192, false>(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_grouped_hdim192<cutlass::bfloat16_t, false>(params, stream);
}

template<>
void run_mha_bwd_grouped_<cutlass::bfloat16_t, 192, true>(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_grouped_hdim192<cutlass::bfloat16_t, true>(params, stream);
}

} // namespace FLASH_NAMESPACE
