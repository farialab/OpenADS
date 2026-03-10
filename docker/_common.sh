#!/usr/bin/env bash
set -euo pipefail

DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${DOCKER_DIR}/.." && pwd)"

OPENADS_CPU_IMAGE="${OPENADS_CPU_IMAGE:-openads:cpu}"
OPENADS_GPU_IMAGE="${OPENADS_GPU_IMAGE:-openads:gpu}"
DEFAULT_OPENADS_OUTPUT_DIR="${PROJECT_ROOT}/docker_ads_output"
OPENADS_OUTPUT_DIR="${OPENADS_OUTPUT_DIR:-${DEFAULT_OPENADS_OUTPUT_DIR}}"

mkdir -p "${OPENADS_OUTPUT_DIR}"

repo_cd() {
  cd "${PROJECT_ROOT}"
}

docker_image_exists() {
  docker image inspect "$1" >/dev/null 2>&1
}

select_openads_image() {
  if docker_image_exists "${OPENADS_GPU_IMAGE}"; then
    echo "${OPENADS_GPU_IMAGE}"
    return 0
  fi
  if docker_image_exists "${OPENADS_CPU_IMAGE}"; then
    echo "${OPENADS_CPU_IMAGE}"
    return 0
  fi

  echo "No OpenADS Docker image found." >&2
  echo "Install one first:" >&2
  echo "  docker build -f docker/Dockerfile.gpu -t ${OPENADS_GPU_IMAGE} ." >&2
  echo "or:" >&2
  echo "  docker build -f docker/Dockerfile.cpu -t ${OPENADS_CPU_IMAGE} ." >&2
  exit 1
}

selected_image="$(select_openads_image)"

docker_runtime_args=()
if [[ "${selected_image}" == "${OPENADS_GPU_IMAGE}" ]]; then
  docker_runtime_args+=(--gpus all)
fi

docker_user_args=(
  --user "$(id -u):$(id -g)"
)

docker_output_args=(
  -v "${OPENADS_OUTPUT_DIR}:/app/output"
)

docker_x11_args=(
  -e "DISPLAY=${DISPLAY:-:0}"
  -e "QT_QPA_PLATFORM=xcb"
  -v /tmp/.X11-unix:/tmp/.X11-unix
)

parse_host_output_flag() {
  local -n _out_args=$1
  local parsed=()
  local host_output="${OPENADS_OUTPUT_DIR}"

  while (($# > 1)); do
    shift
    case "$1" in
      --host-output)
        shift
        if (($# == 0)); then
          echo "--host-output requires a value" >&2
          exit 1
        fi
        host_output="$1"
        ;;
      --host-output=*)
        host_output="${1#*=}"
        ;;
      *)
        parsed+=("$1")
        ;;
    esac
  done

  mkdir -p "${host_output}"
  docker_output_args=(
    -v "${host_output}:/app/output"
  )
  _out_args=("${parsed[@]}")
}
