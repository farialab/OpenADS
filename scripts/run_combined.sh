#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENADS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_DWI_PATH="${OPENADS_ROOT}/assets/examples/dwi/sub-02e8eb42"
DEFAULT_PWI_PATH="${OPENADS_ROOT}/assets/examples/pwi/sub-02e8eb42"
DEFAULT_GPU="1"
DEFAULT_DWI_CONFIG="${OPENADS_ROOT}/configs/dwi_pipeline.yaml"
DEFAULT_PWI_CONFIG="${OPENADS_ROOT}/configs/pwi_pipeline.yaml"

DWI_PATH="${DEFAULT_DWI_PATH}"
PWI_PATH="${DEFAULT_PWI_PATH}"
GPU="${DEFAULT_GPU}"
DWI_CONFIG="${DEFAULT_DWI_CONFIG}"
PWI_CONFIG="${DEFAULT_PWI_CONFIG}"
OUTPUT_ROOT=""
RUN_ALL=false
DWI_STAGES=""
PWI_STAGES=""
NO_MASK_COPY=false

usage() {
  cat << EOF
Usage:
  $(basename "$0") [options]

Options:
  --dwi <path>           DWI subject directory (default: ${DEFAULT_DWI_PATH})
  --pwi <path>           PWI subject directory (default: ${DEFAULT_PWI_PATH})
  --gpu <id>             GPU id (default: ${DEFAULT_GPU})
  --dwi-config <path>    DWI config (default: ${DEFAULT_DWI_CONFIG})
  --pwi-config <path>    PWI config (default: ${DEFAULT_PWI_CONFIG})
  --output-root <path>   Override output root for both DWI/PWI runs
  --all                  Run all stages for selected modality runs
  --dwi-stages <list>    DWI stages (comma-separated)
  --pwi-stages <list>    PWI stages (comma-separated)
  --no-mask-copy         Deprecated (kept for compatibility; no-op)
  -h, --help             Show this help

Behavior:
  - Runs DWI first, then PWI.
  - If --all is not set:
    - DWI runs only when --dwi-stages is provided
    - PWI runs only when --pwi-stages is provided
  - If both stage lists are empty and --all is not set, script exits with error.

Examples:
  $(basename "$0") --dwi /data/dwi/sub-001 --pwi /data/pwi/sub-001 --all --gpu 1
  $(basename "$0") --dwi /data/dwi/sub-001 --pwi /data/pwi/sub-001 --dwi-stages prepdata,registration,inference,report --pwi-stages prepdata,gen_ttp,registration,ttpadc_coreg,inference,report --gpu 1
  $(basename "$0") --dwi /data/dwi/sub-001 --pwi /data/pwi/sub-001 --pwi-stages prepdata,gen_ttp,registration,ttpadc_coreg,inference,report --gpu 1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dwi) DWI_PATH="$2"; shift 2 ;;
    --pwi) PWI_PATH="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --dwi-config) DWI_CONFIG="$2"; shift 2 ;;
    --pwi-config) PWI_CONFIG="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --all) RUN_ALL=true; shift 1 ;;
    --dwi-stages) DWI_STAGES="${2// /}"; shift 2 ;;
    --pwi-stages) PWI_STAGES="${2// /}"; shift 2 ;;
    --no-mask-copy) NO_MASK_COPY=true; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; echo ""; usage; exit 2 ;;
  esac
done

if [[ ! -d "${DWI_PATH}" ]]; then
  echo "DWI path does not exist or is not a directory: ${DWI_PATH}"
  exit 1
fi
if [[ ! -d "${PWI_PATH}" ]]; then
  echo "PWI path does not exist or is not a directory: ${PWI_PATH}"
  exit 1
fi
if [[ ! -f "${DWI_CONFIG}" ]]; then
  echo "DWI config not found: ${DWI_CONFIG}"
  exit 1
fi
if [[ ! -f "${PWI_CONFIG}" ]]; then
  echo "PWI config not found: ${PWI_CONFIG}"
  exit 1
fi

if [[ "${RUN_ALL}" == false && -z "${DWI_STAGES}" && -z "${PWI_STAGES}" ]]; then
  echo "No stages selected. Provide --all or at least one of --dwi-stages / --pwi-stages."
  exit 1
fi

if [[ "${NO_MASK_COPY}" == true ]]; then
  echo "⚠ --no-mask-copy is deprecated in run_combined.sh and has no effect."
fi

echo "==========================================================="
echo "OpenADS root: ${OPENADS_ROOT}"
echo "DWI subject:  ${DWI_PATH}"
echo "PWI subject:  ${PWI_PATH}"
echo "GPU:          ${GPU}"
echo "DWI config:   ${DWI_CONFIG}"
echo "PWI config:   ${PWI_CONFIG}"
[[ -n "${OUTPUT_ROOT}" ]] && echo "Output root:  ${OUTPUT_ROOT}"
if [[ "${RUN_ALL}" == true ]]; then
  echo "Run mode:     ALL"
else
  echo "Run mode:     Selected stages only"
  [[ -n "${DWI_STAGES}" ]] && echo "DWI stages:   ${DWI_STAGES}"
  [[ -n "${PWI_STAGES}" ]] && echo "PWI stages:   ${PWI_STAGES}"
fi
echo "==========================================================="

cd "${OPENADS_ROOT}"
export PYTHONPATH="${PYTHONPATH:-}:${OPENADS_ROOT}/src"

if [[ "${RUN_ALL}" == true || -n "${DWI_STAGES}" ]]; then
  DWI_CMD=(bash "${OPENADS_ROOT}/scripts/run_dwi.sh" "${DWI_PATH}" --config "${DWI_CONFIG}" --gpu "${GPU}")
  [[ -n "${OUTPUT_ROOT}" ]] && DWI_CMD+=(--output-root "${OUTPUT_ROOT}")
  if [[ "${RUN_ALL}" == true ]]; then
    DWI_CMD+=(--all)
  else
    DWI_CMD+=(--stages "${DWI_STAGES}")
  fi
  echo ""
  echo "[1/2] Running DWI pipeline..."
  "${DWI_CMD[@]}"
else
  echo ""
  echo "[1/2] Skipping DWI (no DWI stages selected)."
fi

if [[ "${RUN_ALL}" == true || -n "${PWI_STAGES}" ]]; then
  PWI_CMD=(bash "${OPENADS_ROOT}/scripts/run_pwi.sh" "${PWI_PATH}" --config "${PWI_CONFIG}" --gpu "${GPU}")
  [[ -n "${OUTPUT_ROOT}" ]] && PWI_CMD+=(--output-root "${OUTPUT_ROOT}")
  if [[ "${RUN_ALL}" == true ]]; then
    PWI_CMD+=(--all)
  else
    PWI_CMD+=(--stages "${PWI_STAGES}")
  fi
  echo ""
  echo "[2/2] Running PWI pipeline..."
  "${PWI_CMD[@]}"
else
  echo ""
  echo "[2/2] Skipping PWI (no PWI stages selected)."
fi

echo ""
echo "✅ Combined wrapper finished."
