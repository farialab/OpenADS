#!/bin/bash

# ============================================================
# ADS DWI Pipeline Runner
# ============================================================

# Default subject path (relative to OpenADS folder)
DEFAULT_SUBJECT_PATH="assets/examples/dwi/sub-02e8eb42"

# Default GPU device
DEFAULT_GPU="1"

# Default config file (relative to OpenADS folder)
DEFAULT_CONFIG="configs/dwi_pipeline.yaml"

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
SUBJECT_PATH=""
GPU="$DEFAULT_GPU"
CONFIG="$DEFAULT_CONFIG"
OUTPUT_ROOT=""
RUN_ALL=false
STAGES_LIST=""

while [[ $# -gt 0 ]]; do
    case $1 in
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
        --output-root)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "❌ Error: --output-root requires a directory path"
                exit 1
            fi
            OUTPUT_ROOT="$2"
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
            # Collect all arguments until next flag or end
            STAGES_LIST="$2"
            shift 2
            # Handle case where user put spaces after commas (e.g., "prepdata, gen_mask")
            # Continue collecting if next arg doesn't start with -- and looks like a stage name
            while [[ $# -gt 0 ]] && [[ ! "$1" == --* ]] && [[ -z "$SUBJECT_PATH" ]]; do
                # Check if this looks like a stage name (not a path)
                if [[ ! "$1" =~ / ]] && [[ ! -d "$1" ]]; then
                    STAGES_LIST="${STAGES_LIST},$1"
                    shift
                else
                    break
                fi
            done
            # Remove spaces from stages list
            STAGES_LIST="${STAGES_LIST// /}"
            ;;
        --help|-h)
            echo "==========================================================="
            echo "ADS DWI Pipeline Runner"
            echo "==========================================================="
            echo ""
            echo "Usage: $0 [SUBJECT_PATH] [OPTIONS]"
            echo ""
            echo "Arguments:"
            echo "  SUBJECT_PATH          Path to subject directory (default: $DEFAULT_SUBJECT_PATH)"
            echo ""
            echo "Options:"
            echo "  --gpu <device>        GPU device to use (default: $DEFAULT_GPU)"
            echo "                        Example: --gpu 0"
            echo ""
            echo "  --config <file>       Config file path (default: $DEFAULT_CONFIG)"
            echo "                        Example: --config configs/custom_dwi.yaml"
            echo ""
            echo "  --output-root <dir>    Override output root directory"
            echo "                        Example: --output-root /data/openads_output"
            echo ""
            echo "  --all                 Run all pipeline stages (overrides YAML config)"
            echo "                        Without this flag, stages are controlled by the YAML config"
            echo ""
            echo "  --stages <list>       Run specific stages (comma-separated, overrides YAML)"
            echo "                        Example: --stages prepdata,gen_mask,inference"
            echo "                        Note: Use quotes if you include spaces after commas"
            echo "                              --stages \"prepdata, gen_mask, inference\""
            echo "                        Available stages: prepdata, gen_mask, skull_strip,"
            echo "                                         registration, inference, report"
            echo "                        Note: postprocessing is always run inside inference"
            echo ""
            echo "  --help, -h            Show this help message"
            echo ""
            echo "Stage Control:"
            echo "  To run specific stages via config, edit the 'stages' section in your YAML file:"
            echo "    stages:"
            echo "      prepdata: true"
            echo "      gen_mask: false"
            echo "      skull_strip: false"
            echo "      registration: false"
            echo "      inference: false"
            echo "      report: true"
            echo ""
            echo "Examples:"
            echo "  # Use defaults (stages controlled by YAML)"
            echo "  $0"
            echo ""
            echo "  # Process specific subject"
            echo "  $0 assets/examples/dwi/sub-12345"
            echo ""
            echo "  # Run all stages (override YAML)"
            echo "  $0 assets/examples/dwi/sub-02e8eb42 --all"
            echo ""
            echo "  # Run specific stages (no spaces)"
            echo "  $0 assets/examples/dwi/sub-02e8eb42 --stages prepdata,gen_mask"
            echo ""
            echo "  # Run specific stages (with spaces - use quotes)"
            echo "  $0 assets/examples/dwi/sub-02e8eb42 --stages \"prepdata, gen_mask, inference\""
            echo ""
            echo "  # Use specific GPU"
            echo "  $0 assets/examples/dwi/sub-02e8eb42 --gpu 0 --all"
            echo ""
            echo "  # Use custom configuration"
            echo "  $0 assets/examples/dwi/sub-02e8eb42 --config configs/custom_dwi.yaml"
            echo ""
            echo "  # Complete example"
            echo "  $0 assets/examples/dwi/sub-02e8eb42 \\"
            echo "     --config configs/dwi_pipeline.yaml \\"
            echo "     --gpu 1 \\"
            echo "     --stages prepdata,inference,report"
            echo ""
            exit 0
            ;;
        -*)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            if [[ -z "$SUBJECT_PATH" ]]; then
                SUBJECT_PATH="$1"
            else
                echo "❌ Error: Multiple subject paths provided"
                echo "First: $SUBJECT_PATH"
                echo "Second: $1"
                echo ""
                echo "💡 Tip: If you're using --stages with spaces after commas,"
                echo "   use quotes: --stages \"prepdata, gen_mask\""
                echo "   or remove spaces: --stages prepdata,gen_mask"
                exit 1
            fi
            shift
            ;;
    esac
done

# Use default subject path if not provided
if [[ -z "$SUBJECT_PATH" ]]; then
    SUBJECT_PATH="$DEFAULT_SUBJECT_PATH"
fi

# Convert to absolute path if it exists
if [[ -d "$SUBJECT_PATH" ]]; then
    SUBJECT_PATH=$(cd "$SUBJECT_PATH" && pwd)
elif [[ -d "$OPENADS_ROOT/$SUBJECT_PATH" ]]; then
    SUBJECT_PATH=$(cd "$OPENADS_ROOT/$SUBJECT_PATH" && pwd)
else
    echo "⚠️  Subject path does not exist: $SUBJECT_PATH (continue for non-prepdata stages)"
fi

# Extract subject ID from path
SUBJECT_ID=$(basename "$SUBJECT_PATH")

# Convert config to absolute path if relative
if [[ ! "$CONFIG" = /* ]]; then
    CONFIG="$OPENADS_ROOT/$CONFIG"
fi

# Check if config file exists
if [[ ! -f "$CONFIG" ]]; then
    echo "❌ Error: Config file does not exist: $CONFIG"
    exit 1
fi

# Display configuration
echo "==========================================================="
echo "ADS DWI Pipeline"
echo "==========================================================="
echo "OpenADS Root : $OPENADS_ROOT"
echo "Subject ID   : $SUBJECT_ID"
echo "Subject Path : $SUBJECT_PATH"
echo "Config File  : $CONFIG"
echo "GPU Device   : $GPU"
if [[ -n "$OUTPUT_ROOT" ]]; then
    echo "Output Root  : $OUTPUT_ROOT"
fi

if [[ "$RUN_ALL" == true ]]; then
    echo "Run Mode     : All stages (override YAML)"
elif [[ -n "$STAGES_LIST" ]]; then
    echo "Run Mode     : Specific stages (override YAML)"
    echo "Stages       : $STAGES_LIST"
else
    echo "Run Mode     : Stages controlled by YAML config"
fi

echo "==========================================================="
echo ""

# Build Python command arguments
PYTHON_ARGS=(
    "--config" "$CONFIG"
    "--subject-path" "$SUBJECT_PATH"
    "--gpu" "$GPU"
)

# Add --all flag if requested
if [[ "$RUN_ALL" == true ]]; then
    PYTHON_ARGS+=("--all")
elif [[ -n "$STAGES_LIST" ]]; then
    PYTHON_ARGS+=("--stages" "$STAGES_LIST")
fi
if [[ -n "$OUTPUT_ROOT" ]]; then
    PYTHON_ARGS+=("--output-root" "$OUTPUT_ROOT")
fi

# Run the Python script
python "$OPENADS_ROOT/scripts/run_ads_dwi.py" "${PYTHON_ARGS[@]}"

# Capture exit code
EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
    echo ""
    echo "==========================================================="
    echo "✅ Pipeline completed successfully"
    echo "==========================================================="
else
    echo ""
    echo "==========================================================="
    echo "❌ Pipeline failed with exit code: $EXIT_CODE"
    echo "==========================================================="
fi

exit $EXIT_CODE
