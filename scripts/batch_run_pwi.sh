#!/bin/bash

# ============================================================
# ADS PWI Batch Pipeline Runner
# ============================================================

# Default paths (relative to OpenADS folder)
DEFAULT_SUBJECTS_ROOT="assets/examples/pwi"
DEFAULT_GPU="1"
DEFAULT_CONFIG="configs/pwi_pipeline.yaml"

# ============================================================
# Script Start
# ============================================================

# Get the OpenADS root directory (parent of scripts folder)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OPENADS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to OpenADS root directory
cd "$OPENADS_ROOT"

# Set PYTHONPATH to include src folder
export PYTHONPATH=$PYTHONPATH:"$OPENADS_ROOT/src"

# ============================================================
# Parse Arguments
# ============================================================
SUBJECTS_ROOT=""
SUBJECTS_FILE=""
GPU="$DEFAULT_GPU"
CONFIG="$DEFAULT_CONFIG"
RUN_ALL=false
STAGES_LIST=""
SUBJECT_PATTERN="sub-*"
PARALLEL=false
MAX_JOBS=1

show_help() {
    echo "==========================================================="
    echo "ADS PWI Batch Pipeline Runner"
    echo "==========================================================="
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Batch process PWI subjects."
    echo ""
    echo "Input Options (choose one):"
    echo "  --subjects-file <file>  Text file with subject paths (one per line)"
    echo "  --subjects-root <path>  Folder path."
    echo "                          - If subfolders exist, processes all subfolders."
    echo "                          - If no subfolders, processes the path as a single subject."
    echo ""
    echo "Subject Selection:"
    echo "  --subject-pattern <pat> Pattern to match subject folders (default: sub-*)"
    echo "                          (Only used when scanning a root directory)"
    echo ""
    echo "Pipeline Options:"
    echo "  --gpu <device>          GPU device to use (default: $DEFAULT_GPU)"
    echo "  --config <file>         Config file path (default: $DEFAULT_CONFIG)"
    echo "  --all                   Run all pipeline stages"
    echo "  --stages <list>         Run specific stages (comma-separated)"
    echo ""
    echo "Execution Options:"
    echo "  --parallel              Run subjects in parallel"
    echo "  --max-jobs <n>          Maximum parallel jobs (default: 1)"
    echo "  --help, -h              Show this help message"
    echo ""
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --subjects-file)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "❌ Error: --subjects-file requires a file path"
                exit 1
            fi
            SUBJECTS_FILE="$2"
            shift 2
            ;;
        --subjects-root)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "❌ Error: --subjects-root requires a path"
                exit 1
            fi
            SUBJECTS_ROOT="$2"
            shift 2
            ;;
        --subject-pattern)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "❌ Error: --subject-pattern requires a pattern"
                exit 1
            fi
            SUBJECT_PATTERN="$2"
            shift 2
            ;;
        --gpu)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "❌ Error: --gpu requires a device number"
                exit 1
            fi
            GPU="$2"
            shift 2
            ;;
        --config)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "❌ Error: --config requires a file path"
                exit 1
            fi
            CONFIG="$2"
            shift 2
            ;;
        --all)
            RUN_ALL=true
            shift
            ;;
        --stages)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "❌ Error: --stages requires a comma-separated list"
                exit 1
            fi
            STAGES_LIST="$2"
            shift 2
            # Handle spaces after commas
            while [[ $# -gt 0 ]] && [[ ! "$1" == --* ]]; do
                if [[ ! "$1" =~ / ]] && [[ ! -d "$1" ]]; then
                    STAGES_LIST="${STAGES_LIST},$1"
                    shift
                else
                    break
                fi
            done
            STAGES_LIST="${STAGES_LIST// /}"
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --max-jobs)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "❌ Error: --max-jobs requires a number"
                exit 1
            fi
            MAX_JOBS="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            ;;
        *)
            echo "❌ Unknown argument: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================
# Validate Config
# ============================================================

# Convert config to absolute path if needed
if [[ ! "$CONFIG" = /* ]]; then
    CONFIG="$OPENADS_ROOT/$CONFIG"
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "❌ Error: Config file does not exist: $CONFIG"
    exit 1
fi

# ============================================================
# Get Subject Paths (LOGIC UPDATED)
# ============================================================
SUBJECT_PATHS=()

if [[ -n "$SUBJECTS_FILE" ]]; then
    # --------------------------------------------------------
    # Method 1: Read from text file
    # --------------------------------------------------------
    if [[ ! -f "$SUBJECTS_FILE" ]]; then
        # Check if it exists relative to OPENADS_ROOT
        if [[ -f "$OPENADS_ROOT/$SUBJECTS_FILE" ]]; then
            SUBJECTS_FILE="$OPENADS_ROOT/$SUBJECTS_FILE"
        else
            echo "❌ Error: Subjects file does not exist: $SUBJECTS_FILE"
            exit 1
        fi
    fi
    
    echo "Reading subjects from file: $SUBJECTS_FILE"
    
    line_number=0
    while IFS= read -r line || [[ -n "$line" ]]; do
        ((line_number++))
        
        # Skip empty lines and comments
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        
        # Trim whitespace
        subject_path=$(echo "$line" | xargs)
        
        # Resolve path: If relative, check existence from current location or OPENADS_ROOT
        final_path=""
        if [[ -d "$subject_path" ]]; then
            final_path="$subject_path"
        elif [[ -d "$OPENADS_ROOT/$subject_path" ]]; then
            final_path="$OPENADS_ROOT/$subject_path"
        fi

        if [[ -n "$final_path" ]]; then
            # Convert to absolute path for consistency in Python
            final_path=$(cd "$final_path" && pwd)
            SUBJECT_PATHS+=("$final_path")
        else
            echo "⚠️  Warning (line $line_number): Subject directory not found: $subject_path"
        fi
    done < "$SUBJECTS_FILE"
    
    INPUT_METHOD="subjects file"

else
    # --------------------------------------------------------
    # Method 2: Folder Path (Batch or Single)
    # --------------------------------------------------------
    
    # Default if not provided
    if [[ -z "$SUBJECTS_ROOT" ]]; then
        SUBJECTS_ROOT="$DEFAULT_SUBJECTS_ROOT"
    fi

    # Check if root exists (absolute or relative)
    if [[ ! -d "$SUBJECTS_ROOT" ]]; then
        if [[ -d "$OPENADS_ROOT/$SUBJECTS_ROOT" ]]; then
            SUBJECTS_ROOT="$OPENADS_ROOT/$SUBJECTS_ROOT"
        else
            echo "❌ Error: Subjects path does not exist: $SUBJECTS_ROOT"
            exit 1
        fi
    fi

    # Make absolute
    SUBJECTS_ROOT=$(cd "$SUBJECTS_ROOT" && pwd)

    echo "Checking folder: $SUBJECTS_ROOT"
    
    # Check for subfolders matching the pattern
    # -mindepth 1 ensures we don't return the root folder itself yet
    FOUND_SUBFOLDERS=($(find "$SUBJECTS_ROOT" -maxdepth 1 -mindepth 1 -type d -name "$SUBJECT_PATTERN" | sort))

    if [[ ${#FOUND_SUBFOLDERS[@]} -gt 0 ]]; then
        # CASE A: Subfolders found -> Batch Mode
        SUBJECT_PATHS=("${FOUND_SUBFOLDERS[@]}")
        INPUT_METHOD="directory (batch mode)"
        echo "Found ${#SUBJECT_PATHS[@]} subfolders matching '$SUBJECT_PATTERN'"
    else
        # CASE B: No subfolders found -> Single Subject Mode
        # The provided path is treated as the subject folder itself
        SUBJECT_PATHS=("$SUBJECTS_ROOT")
        INPUT_METHOD="directory (single subject mode)"
        echo "No subfolders matching '$SUBJECT_PATTERN' found."
        echo "Treating input path as a single subject folder."
    fi
fi

# ============================================================
# Final Validation
# ============================================================

if [[ ${#SUBJECT_PATHS[@]} -eq 0 ]]; then
    echo "❌ Error: No valid subject directories found."
    exit 1
fi

# ============================================================
# Display Configuration
# ============================================================
echo "==========================================================="
echo "ADS PWI Batch Pipeline"
echo "==========================================================="
echo "Input Method   : $INPUT_METHOD"
echo "Total Subjects : ${#SUBJECT_PATHS[@]}"
echo "Config File    : $CONFIG"
echo "GPU Device     : $GPU"

if [[ "$RUN_ALL" == true ]]; then
    echo "Run Mode       : All stages"
elif [[ -n "$STAGES_LIST" ]]; then
    echo "Run Mode       : Specific stages ($STAGES_LIST)"
else
    echo "Run Mode       : Stages controlled by YAML config"
fi

if [[ "$PARALLEL" == true ]]; then
    echo "Parallel Mode  : YES (Max jobs: $MAX_JOBS)"
else
    echo "Parallel Mode  : NO"
fi
echo "==========================================================="
echo ""

# List subjects (truncated if too long)
echo "Subjects to process:"
count=0
for subject_path in "${SUBJECT_PATHS[@]}"; do
    ((count++))
    if [[ $count -le 5 ]]; then
        echo "  - $(basename "$subject_path")"
    fi
done
if [[ ${#SUBJECT_PATHS[@]} -gt 5 ]]; then
    echo "  ... and $(( ${#SUBJECT_PATHS[@]} - 5 )) more"
fi
echo ""

# Confirmation for large batches
if [[ ${#SUBJECT_PATHS[@]} -gt 10 ]]; then
    read -p "⚠️  About to process ${#SUBJECT_PATHS[@]} subjects. Continue? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted by user"
        exit 0
    fi
    echo ""
fi

# ============================================================
# Build Python Command Arguments
# ============================================================
build_python_args() {
    local subject_path=$1
    local args=(
        "--config" "$CONFIG"
        "--subject-path" "$subject_path"
        "--gpu" "$GPU"
    )
    
    if [[ "$RUN_ALL" == true ]]; then
        args+=("--all")
    elif [[ -n "$STAGES_LIST" ]]; then
        args+=("--stages" "$STAGES_LIST")
    fi
    
    echo "${args[@]}"
}

# ============================================================
# Process Single Subject Function
# ============================================================
process_subject() {
    local subject_path=$1
    local subject_id=$(basename "$subject_path")
    
    echo "------------------------------------------"
    echo "Processing: $subject_id"
    echo "------------------------------------------"
    
    local python_args=($(build_python_args "$subject_path"))
    
    python "$OPENADS_ROOT/scripts/run_ads_pwi.py" "${python_args[@]}"
    
    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        echo "✅ Success: $subject_id"
        return 0
    else
        echo "❌ Failed: $subject_id (exit code: $exit_code)"
        return 1
    fi
}

# ============================================================
# Process Subjects
# ============================================================
TOTAL=${#SUBJECT_PATHS[@]}
SUCCESS=0
FAILED=0
FAILED_SUBJECTS=()
START_TIME=$(date +%s)

if [[ "$PARALLEL" == true ]]; then
    echo "🚀 Running in parallel mode..."
    echo ""
    
    # Export function and variables
    export -f process_subject
    export -f build_python_args
    export OPENADS_ROOT CONFIG GPU RUN_ALL STAGES_LIST
    
    TEMP_DIR=$(mktemp -d)
    
    for subject_path in "${SUBJECT_PATHS[@]}"; do
        subject_id=$(basename "$subject_path")
        (
            if process_subject "$subject_path"; then
                echo "success" > "$TEMP_DIR/$subject_id.result"
            else
                echo "failed" > "$TEMP_DIR/$subject_id.result"
            fi
        ) &
        
        while [[ $(jobs -r | wc -l) -ge $MAX_JOBS ]]; do
            sleep 1
        done
    done
    
    wait
    
    for subject_path in "${SUBJECT_PATHS[@]}"; do
        subject_id=$(basename "$subject_path")
        if [[ -f "$TEMP_DIR/$subject_id.result" ]] && [[ "$(cat "$TEMP_DIR/$subject_id.result")" == "success" ]]; then
            ((SUCCESS++))
        else
            ((FAILED++))
            FAILED_SUBJECTS+=("$subject_id")
        fi
    done
    rm -rf "$TEMP_DIR"

else
    echo "🔄 Running in sequential mode..."
    echo ""
    
    for i in "${!SUBJECT_PATHS[@]}"; do
        subject_path="${SUBJECT_PATHS[$i]}"
        echo "[$((i+1))/$TOTAL]"
        
        if process_subject "$subject_path"; then
            ((SUCCESS++))
        else
            ((FAILED++))
            FAILED_SUBJECTS+=("$(basename "$subject_path")")
        fi
    done
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# ============================================================
# Summary
# ============================================================
echo ""
echo "==========================================================="
echo "Batch Processing Summary"
echo "==========================================================="
echo "Total: $TOTAL | Success: $SUCCESS | Failed: $FAILED"
echo "Time : $((ELAPSED / 60))m $((ELAPSED % 60))s"

if [[ $FAILED -gt 0 ]]; then
    echo "Failed List:"
    for subject in "${FAILED_SUBJECTS[@]}"; do
        echo "  - $subject"
    done
    exit 1
fi

echo "✅ All Done!"
exit 0
