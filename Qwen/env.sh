#!/usr/bin/env bash

: "${CONDA_PREFIX:?请先执行 conda activate msloc_qwen35}"

export CUDA_HOME="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/cuda-compat:${LD_LIBRARY_PATH:-}"
export FORCE_QWENVL_VIDEO_READER=torchcodec
export TORCH_CUDA_ARCH_LIST=8.0
export FLASH_ATTN_CUDA_ARCHS=80
export MAX_JOBS=8
