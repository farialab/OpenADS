#!/bin/bash

set -euo pipefail

# ============================================================
# ADS Combined DWI + PWI Batch Pipeline Runner
# ============================================================

DEFAULT_DWI_ROOT="assets/examples/dwi"
DEFAULT_PWI_ROOT="assets/examples/pwi"
DEFAULT_GPU="1"
DEFAULT_DWI_CONFIG="configs/dwi_pipeline.yaml"
DEFAULT_PWI_CONFIG="configs/pwi_pipeline.yaml"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OPENADS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$OPENADS_ROOT"

export PYTHONPATH="${PYTHONPATH:-}:$OPENADS_ROOT/src"

# ============================================================
# Parse Arguments
# ============================================================
DWI_ROOT=""
PWI_ROOT=""
GPU="$DEFAULT_GPU"
DWI_CONFIG="$DEFAULT_DWI_CONFIG"
PWI_CONFIG="$DEFAULT_PWI_CONFIG"
RUN_ALL=false
DWI_STAGES=""
PWI_STAGES=""
NO_MASK_COPY=false
SUBJECT_PATTERN="sub-*"
PARALLEL=false
MAX_JOBS=1

show_help() {
    echo "==========================================================="
    echo "ADS Combined DWI + PWI Batch Pipeline Runner"
    echo "==========================================================="
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "This script:"
    echo "  1) Reads subject IDs from DWI root"
    echo "  2) Finds same IDs under PWI root"
    echo "  3) Runs combined pipeline (DWI then PWI) for matched IDs"
    echo ""
    echo "Input:"
    echo "  --dwi-root <path>       DWI dataset root (default: $DEFAULT_DWI_ROOT)"
    echo "  --pwi-root <path>       PWI dataset root (default: $DEFAULT_PWI_ROOT)"
    echo "  --subject-pattern <pat> Subject folder pattern (default: $SUBJECT_PATTERN)"
    echo ""
    echo "Pipeline Options:"
    echo "  --gpu <id>              GPU id (default: $DEFAULT_GPU)"
    echo "  --dwi-config <path>     DWI config (default: $DEFAULT_DWI_CONFIG)"
    echo "  --pwi-config <path>     PWI config (default: $DEFAULT_PWI_CONFIG)"
    echo "  --all                   Run all stages for both"
    echo "  --dwi-stages <list>     DWI stages (comma-separated)"
    echo "  --pwi-stages <list>     PWI stages (comma-separated)"
    echo "  --no-mask-copy          Disable DWI mask copy to PWI preprocess"
    echo ""
    echo "Execution Options:"
    echo "  --parallel              Run multiple subjects in parallel"
    echo "  --max-jobs <n>          Max parallel jobs (default: 1)"
    echo ""
    echo "  --help, -h              Show this help"
    echo ""
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dwi-root)
            DWI_ROOT="$2"; shift 2 ;;
        --pwi-root)
            PWI_ROOT="$2"; shift 2 ;;
        --subject-pattern)
            SUBJECT_PATTERN="$2"; shift 2 ;;
        --gpu)
            GPU="$2"; shift 2 ;;
        --dwi-config)
            DWI_CONFIG="$2"; shift 2 ;;
        --pwi-config)
            PWI_CONFIG="$2"; shift 2 ;;
        --all)
            RUN_ALL=true; shift 1 ;;
        --dwi-stages)
            DWI_STAGES="${2// /}"; shift 2 ;;
        --pwi-stages)
            PWI_STAGES="${2// /}"; shift 2 ;;
        --no-mask-copy)
            NO_MASK_COPY=true; shift 1 ;;
        --parallel)
            PARALLEL=true; shift 1 ;;
        --max-jobs)
            MAX_JOBS="$2"; shift 2 ;;
        --help|-h)
            show_help ;;
        *)
            echo "❌ Unknown argument: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if [[ -z "$DWI_ROOT" ]]; then
    DWI_ROOT="$DEFAULT_DWI_ROOT"
fi
if [[ -z "$PWI_ROOT" ]]; then
    PWI_ROOT="$DEFAULT_PWI_ROOT"
fi

# Resolve paths
if [[ ! "$DWI_ROOT" = /* ]]; then
    DWI_ROOT="$OPENADS_ROOT/$DWI_ROOT"
fi
if [[ ! "$PWI_ROOT" = /* ]]; then
    PWI_ROOT="$OPENADS_ROOT/$PWI_ROOT"
fi
if [[ ! "$DWI_CONFIG" = /* ]]; then
    DWI_CONFIG="$OPENADS_ROOT/$DWI_CONFIG"
fi
if [[ ! "$PWI_CONFIG" = /* ]]; then
    PWI_CONFIG="$OPENADS_ROOT/$PWI_CONFIG"
fi

if [[ ! -d "$DWI_ROOT" ]]; then
    echo "❌ DWI root not found: $DWI_ROOT"
    exit 1
fi
if [[ ! -d "$PWI_ROOT" ]]; then
    echo "❌ PWI root not found: $PWI_ROOT"
    exit 1
fi
if [[ ! -f "$DWI_CONFIG" ]]; then
    echo "❌ DWI config not found: $DWI_CONFIG"
    exit 1
fi
if [[ ! -f "$PWI_CONFIG" ]]; then
    echo "❌ PWI config not found: $PWI_CONFIG"
    exit 1
fi

# ============================================================
# Match Subject IDs: DWI root -> PWI root
# ============================================================
DWI_SUBJECTS=()
while IFS= read -r p; do
    DWI_SUBJECTS+=("$p")
done < <(find "$DWI_ROOT" -maxdepth 1 -mindepth 1 -type d -name "$SUBJECT_PATTERN" | sort)

if [[ ${#DWI_SUBJECTS[@]} -eq 0 ]]; then
    echo "❌ No DWI subjects found under: $DWI_ROOT (pattern: $SUBJECT_PATTERN)"
    exit 1
fi

MATCHED_DWI=()
MATCHED_PWI=()
for dwi_path in "${DWI_SUBJECTS[@]}"; do
    sid="$(basename "$dwi_path")"
    pwi_path="$PWI_ROOT/$sid"
    if [[ -d "$pwi_path" ]]; then
        MATCHED_DWI+=("$dwi_path")
        MATCHED_PWI+=("$pwi_path")
    fi
done

if [[ ${#MATCHED_DWI[@]} -eq 0 ]]; then
    echo "❌ No matched subject IDs found between:"
    echo "   DWI root: $DWI_ROOT"
    echo "   PWI root: $PWI_ROOT"
    exit 1
fi

# ============================================================
# Display Configuration
# ============================================================
echo "==========================================================="
echo "ADS Combined Batch Pipeline"
echo "==========================================================="
echo "DWI root       : $DWI_ROOT"
echo "PWI root       : $PWI_ROOT"
echo "Subject pattern: $SUBJECT_PATTERN"
echo "Matched subjects: ${#MATCHED_DWI[@]}"
echo "GPU            : $GPU"
echo "DWI config     : $DWI_CONFIG"
echo "PWI config     : $PWI_CONFIG"
if [[ "$RUN_ALL" == true ]]; then
    echo "Run mode       : ALL"
elif [[ -n "$DWI_STAGES" || -n "$PWI_STAGES" ]]; then
    echo "Run mode       : Specific stages"
    [[ -n "$DWI_STAGES" ]] && echo "  DWI stages   : $DWI_STAGES"
    [[ -n "$PWI_STAGES" ]] && echo "  PWI stages   : $PWI_STAGES"
else
    echo "Run mode       : Config-controlled"
fi
if [[ "$NO_MASK_COPY" == true ]]; then
    echo "Mask copy      : disabled"
fi
if [[ "$PARALLEL" == true ]]; then
    echo "Parallel mode  : YES (max jobs: $MAX_JOBS)"
else
    echo "Parallel mode  : NO"
fi
echo "==========================================================="
echo ""

# ============================================================
# Per-subject Processing
# ============================================================
process_subject_pair() {
    local dwi_path="$1"
    local pwi_path="$2"
    local sid
    sid="$(basename "$dwi_path")"

    echo "------------------------------------------"
    echo "Processing: $sid"
    echo "DWI: $dwi_path"
    echo "PWI: $pwi_path"
    echo "------------------------------------------"

    local cmd=(
        python3 "$OPENADS_ROOT/scripts/run_ads_combined.py"
        --dwi-subject-path "$dwi_path"
        --pwi-subject-path "$pwi_path"
        --dwi-config "$DWI_CONFIG"
        --pwi-config "$PWI_CONFIG"
        --gpu "$GPU"
    )

    if [[ "$RUN_ALL" == true ]]; then
        cmd+=(--all)
    else
        [[ -n "$DWI_STAGES" ]] && cmd+=(--dwi-stages "$DWI_STAGES")
        [[ -n "$PWI_STAGES" ]] && cmd+=(--pwi-stages "$PWI_STAGES")
    fi
    [[ "$NO_MASK_COPY" == true ]] && cmd+=(--no-mask-copy)

    "${cmd[@]}"
    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        echo "✅ Success: $sid"
        return 0
    else
        echo "❌ Failed: $sid (exit code: $exit_code)"
        return 1
    fi
}

export -f process_subject_pair
export OPENADS_ROOT DWI_CONFIG PWI_CONFIG GPU RUN_ALL DWI_STAGES PWI_STAGES NO_MASK_COPY

# ============================================================
# Run Batch
# ============================================================
TOTAL=${#MATCHED_DWI[@]}
SUCCESS=0
FAILED=0
FAILED_SUBJECTS=()
START_TIME=$(date +%s)

if [[ "$PARALLEL" == true ]]; then
    echo "🚀 Running in parallel mode..."
    echo ""

    TEMP_DIR=$(mktemp -d)
    for i in "${!MATCHED_DWI[@]}"; do
        dwi_path="${MATCHED_DWI[$i]}"
        pwi_path="${MATCHED_PWI[$i]}"
        sid="$(basename "$dwi_path")"
        (
            if process_subject_pair "$dwi_path" "$pwi_path" >/dev/null 2>&1; then
                echo "success" > "$TEMP_DIR/$sid.result"
            else
                echo "failed" > "$TEMP_DIR/$sid.result"
            fi
        ) &

        while [[ $(jobs -r | wc -l) -ge "$MAX_JOBS" ]]; do
            sleep 1
        done
    done

    wait

    for dwi_path in "${MATCHED_DWI[@]}"; do
        sid="$(basename "$dwi_path")"
        if [[ -f "$TEMP_DIR/$sid.result" ]] && [[ "$(cat "$TEMP_DIR/$sid.result")" == "success" ]]; then
            ((SUCCESS++))
        else
            ((FAILED++))
            FAILED_SUBJECTS+=("$sid")
        fi
    done
    rm -rf "$TEMP_DIR"
else
    echo "🔄 Running in sequential mode..."
    echo ""
    for i in "${!MATCHED_DWI[@]}"; do
        echo "[$((i+1))/$TOTAL]"
        if process_subject_pair "${MATCHED_DWI[$i]}" "${MATCHED_PWI[$i]}"; then
            ((SUCCESS++))
        else
            ((FAILED++))
            FAILED_SUBJECTS+=("$(basename "${MATCHED_DWI[$i]}")")
        fi
    done
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "==========================================================="
echo "Combined Batch Processing Summary"
echo "==========================================================="
echo "Total: $TOTAL | Success: $SUCCESS | Failed: $FAILED"
echo "Time : $((ELAPSED / 60))m $((ELAPSED % 60))s"

if [[ $FAILED -gt 0 ]]; then
    echo "Failed List:"
    for sid in "${FAILED_SUBJECTS[@]}"; do
        echo "  - $sid"
    done
    exit 1
fi

echo "✅ All Done!"
exit 0
