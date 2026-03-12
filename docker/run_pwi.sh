#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

args=()
host_subject_path=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host-subject-path)
      host_subject_path="$2"
      shift 2
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

parse_host_output_flag args "${args[@]}"

docker_input_args=()
if [[ -n "$host_subject_path" ]]; then
  subject_name="$(basename "$host_subject_path")"
  docker_input_args=(-v "$host_subject_path:/app/input/$subject_name")
  args+=(--subject-path "/app/input/$subject_name")
fi

docker run --rm -it \
  "${docker_runtime_args[@]}" \
  "${docker_user_args[@]}" \
  "${docker_output_args[@]}" \
  "${docker_input_args[@]}" \
  "${selected_image}" pwi "${args[@]}" --output-root /app/output