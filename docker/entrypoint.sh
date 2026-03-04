#!/usr/bin/env bash
set -e

usage() {
  echo "Usage: ads {api|dwi|pwi|combined|gui|cli} [ARGS...]"
  echo "Examples:"
  echo "  ads api                     # run FastAPI server (default)"
  echo "  ads dwi --all --subject-path /raw/sub-02e8eb42"
  echo "  ads pwi --all --subject-path /raw/sub-02e8eb42"
  echo "  ads combined --dwi-subject-path /raw/dwi/sub-02e8eb42 --pwi-subject-path /raw/pwi/sub-02e8eb42 --all"
  echo "  ads cli dwi --all --subject-path /raw/sub-02e8eb42"
  echo "  ads gui                     # run /app/GUI_launcher.py"
}

case "${1:-api}" in
  api) shift; exec python -m uvicorn server.app:app --host 0.0.0.0 --port 8000 "$@" ;;
  dwi) shift; exec python -m ads.cli dwi "$@" ;;
  pwi) shift; exec python -m ads.cli pwi "$@" ;;
  combined) shift; exec python scripts/run_ads_combined.py "$@" ;;
  cli) shift; exec python -m ads.cli "$@" ;;
  gui) shift; exec python /app/GUI_launcher.py "$@" ;;
  -h|--help) usage ;;
  *) exec "$@" ;;
esac
