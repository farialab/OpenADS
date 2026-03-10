#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

args=()
parse_host_output_flag args "$@"

echo "If this is a Linux X11 host, run: xhost +local:docker"

docker run --rm -it \
  "${docker_runtime_args[@]}" \
  "${docker_user_args[@]}" \
  "${docker_output_args[@]}" \
  "${docker_x11_args[@]}" \
  "${selected_image}" gui "${args[@]}"
