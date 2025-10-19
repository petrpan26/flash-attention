/*******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Grouped flash attention backward kernel instantiation for fp16, headdim 96, sm80+
 ******************************************************************************/

#include "flash_bwd_launch_template.h"

namespace FLASH_NAMESPACE {

template<>
void run_mha_bwd_grouped_<cutlass::half_t, 96, false>(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_grouped_hdim96<cutlass::half_t, false>(params, stream);
}

template<>
void run_mha_bwd_grouped_<cutlass::half_t, 96, true>(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_grouped_hdim96<cutlass::half_t, true>(params, stream);
}

} // namespace FLASH_NAMESPACE
