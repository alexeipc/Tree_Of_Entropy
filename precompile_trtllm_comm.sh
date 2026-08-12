#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Precompile FlashInfer trtllm_comm on H200
# ============================================================

FLASHINFER_VERSION="0.6.6"
GPU_ARCH="90a"
BUILD_JOBS="${BUILD_JOBS:-8}"

CACHE_DIR="${HOME}/.cache/flashinfer/${FLASHINFER_VERSION}/${GPU_ARCH}/cached_ops/trtllm_comm"
LOG_FILE="${HOME}/trtllm_comm_build_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================"
echo "FlashInfer trtllm_comm precompile"
echo "Cache directory: ${CACHE_DIR}"
echo "Build jobs:      ${BUILD_JOBS}"
echo "Log file:        ${LOG_FILE}"
echo "============================================================"

# ------------------------------------------------------------
# 1. Kill stale recursive Ninja processes for this target
# ------------------------------------------------------------

echo
echo "[1/6] Killing stale trtllm_comm Ninja processes..."

pkill -9 -u "${USER}" -f "ninja.*trtllm_comm" 2>/dev/null || true

sleep 2

if pgrep -u "${USER}" -af "ninja.*trtllm_comm" >/dev/null; then
    echo "ERROR: Some trtllm_comm Ninja processes are still alive:"
    pgrep -u "${USER}" -af "ninja.*trtllm_comm"
    exit 1
fi

echo "No stale trtllm_comm Ninja processes remain."

# ------------------------------------------------------------
# 2. Find the native Ninja binary
# ------------------------------------------------------------

echo
echo "[2/6] Locating native Ninja binary..."

REAL_NINJA="$(
python - <<'PY'
import os
import sys

try:
    import ninja
except ImportError:
    sys.exit("Python package 'ninja' is not installed.")

binary = os.path.join(ninja.BIN_DIR, "ninja")

if not os.path.isfile(binary):
    sys.exit(f"Native Ninja binary not found: {binary}")

print(binary)
PY
)"

echo "Native Ninja: ${REAL_NINJA}"

if [[ ! -x "${REAL_NINJA}" ]]; then
    echo "ERROR: Ninja binary is not executable:"
    echo "  ${REAL_NINJA}"
    exit 1
fi

echo
file "${REAL_NINJA}"
"${REAL_NINJA}" --version

# Ensure subprocesses resolve ninja to the native binary.
export PATH="$(dirname "${REAL_NINJA}"):${PATH}"
hash -r

echo
echo "Ninja resolved through PATH:"
command -v ninja
file "$(command -v ninja)"

if [[ "$(readlink -f "$(command -v ninja)")" != "$(readlink -f "${REAL_NINJA}")" ]]; then
    echo "ERROR: PATH still resolves Ninja to the wrong executable."
    exit 1
fi

# ------------------------------------------------------------
# 3. Verify H200/CUDA environment
# ------------------------------------------------------------

echo
echo "[3/6] Checking GPU and CUDA environment..."

nvidia-smi --query-gpu=index,name,driver_version,memory.total \
    --format=csv,noheader || true

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(
            f"GPU {index}: {props.name}, "
            f"compute capability {props.major}.{props.minor}"
        )
PY

# ------------------------------------------------------------
# 4. Validate generated build directory
# ------------------------------------------------------------

echo
echo "[4/6] Checking build directory..."

if [[ ! -d "${CACHE_DIR}" ]]; then
    echo "ERROR: FlashInfer build directory does not exist:"
    echo "  ${CACHE_DIR}"
    echo
    echo "Run vLLM once so FlashInfer generates build.ninja, then rerun this script."
    exit 1
fi

if [[ ! -f "${CACHE_DIR}/build.ninja" ]]; then
    echo "ERROR: build.ninja does not exist:"
    echo "  ${CACHE_DIR}/build.ninja"
    exit 1
fi

echo "Found:"
echo "  ${CACHE_DIR}/build.ninja"

# Remove stale Ninja bookkeeping that may have been left by the
# recursively spawned wrapper. Do not delete generated source files.
rm -f \
    "${CACHE_DIR}/.ninja_lock" \
    "${CACHE_DIR}/.ninja_deps.tmp" \
    "${CACHE_DIR}/.ninja_log.tmp"

# ------------------------------------------------------------
# 5. Show what Ninja intends to build
# ------------------------------------------------------------

echo
echo "[5/6] Running Ninja dry run..."

"${REAL_NINJA}" \
    -C "${CACHE_DIR}" \
    -f "${CACHE_DIR}/build.ninja" \
    -n \
    -v

# ------------------------------------------------------------
# 6. Compile
# ------------------------------------------------------------

echo
echo "[6/6] Compiling trtllm_comm..."
echo

set +e

"${REAL_NINJA}" \
    -C "${CACHE_DIR}" \
    -f "${CACHE_DIR}/build.ninja" \
    -v \
    -j "${BUILD_JOBS}" \
    2>&1 | tee "${LOG_FILE}"

BUILD_STATUS=${PIPESTATUS[0]}

set -e

echo

if [[ "${BUILD_STATUS}" -ne 0 ]]; then
    echo "============================================================"
    echo "BUILD FAILED"
    echo "Exit code: ${BUILD_STATUS}"
    echo "Log: ${LOG_FILE}"
    echo "============================================================"
    exit "${BUILD_STATUS}"
fi

echo "============================================================"
echo "BUILD SUCCEEDED"
echo "============================================================"

echo
echo "Generated libraries:"
find "${CACHE_DIR}" \
    -type f \
    \( -name "*.so" -o -name "*.cubin" -o -name "*.o" \) \
    -printf "%TY-%Tm-%Td %TH:%TM:%TS %s %p\n" \
    | sort \
    | tail -50

echo
echo "Use this before launching your actual vLLM/Ray program:"
echo
echo "export PATH=\"$(dirname "${REAL_NINJA}"):\${PATH}\""
echo
echo "You can now run vLLM with enforce_eager=False."