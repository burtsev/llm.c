#pragma once

#include "llmc/cuda_common.h"
#include "llmc/encoder.cuh" // for ParameterTensors, ActivationTensors

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int num_streams;       // Number of independent streams (N)
    int stream_size;       // d_stream = d_model / N
    int num_heads_stream;  // num_heads / N
} SplitStreamConfig;

// Forward pass for a single split-stream transformer block
void split_stream_transformer_forward(
    floatX* output,              // [B, T, d_model]
    const floatX* input,         // [B, T, d_model]
    const ParameterTensors* params, // model parameters (expect per-stream layout!)
    ActivationTensors* acts,     // activations (per-stream layout if needed)
    const SplitStreamConfig* config,
    int layer_id,
    int B, int T,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif
