#!/usr/bin/env bash
# Bootstrap the BitDecoding-FP4 inference environment.
#
# Usage:
#   bash tools/bootstrap_env.sh
#   bash tools/bootstrap_env.sh _local/envs/bd_fp4
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(realpath "${REPO_ROOT}")"
export REPO_ROOT
cd "${REPO_ROOT}"

# Resolved so that a symlink to local disk is installed through its target:
# through the link, every file operation is a lookup on the repo's mount.
venv="$(realpath -m "${1:-_local/envs/bd_fp4}")"
export UV_LINK_MODE=copy

if ! command -v uv >/dev/null 2>&1; then
    echo "Install uv first: python3.12 -m pip install uv" >&2
    exit 1
fi

uv venv --managed-python --python 3.12 "${venv}"
py="${venv}/bin/python"
uv pip install --python "${py}" \
    torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu130
uv pip install --python "${py}" \
    "cmake>=3.26.1" ninja "packaging>=24.2" \
    "setuptools>=77.0.3,<81.0.0" setuptools-scm setuptools-rust maturin wheel jinja2
(
    cd third_party/vllm
    VLLM_USE_PRECOMPILED=1 uv pip install \
        --python "${py}" -e . --no-build-isolation \
        "quack-kernels==0.6.1" \
        --extra-index-url https://flashinfer.ai/whl/ \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --index-strategy unsafe-best-match
)
uv pip install --python "${py}" -e ".[eval]"
"${py}" -c \
    "import torch, vllm, flashinfer, nvfp4_decode_kernel; print('torch', torch.__version__, '| vllm', vllm.__version__, '| flashinfer', flashinfer.__version__)"
echo "inference environment ready: ${venv}"
echo "LiveCodeBench grading uses this same interpreter in a separate process."
echo "Put a LiveCodeBench checkout on PYTHONPATH (_local/LiveCodeBench); do not pip install it."
