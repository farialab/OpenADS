#!/usr/bin/env bash
set -e

usage() {
  echo "Usage: ads {api|dwi|pwi|combined|gui|cli} [ARGS...]"
  echo "Examples:"
  echo "  ads api                     # run FastAPI backend only (no HTML UI)"
  echo "  ads dwi --all --subject-path /raw/sub-02e8eb42"
  echo "  ads pwi --all --subject-path /raw/sub-02e8eb42"
  echo "  ads combined --dwi-subject-path /raw/dwi/sub-02e8eb42 --pwi-subject-path /raw/pwi/sub-02e8eb42 --all"
  echo "  ads cli dwi --all --subject-path /raw/sub-02e8eb42"
  echo "  ads gui                     # run /app/GUI_launcher.py (requires display forwarding in Docker)"
  echo
  echo "Docker usage:"
  echo "  docker run --rm -it openads:cpu --help"
  echo "  docker run --rm -it -p 8000:8000 openads:cpu api   # backend only"
  echo "  docker run --rm -it openads:cpu dwi --subject-path /app/assets/examples/dwi/sub-02e8eb42 --all --gpu 0"
}

case "${1:-api}" in
  api) shift; exec python -m uvicorn server.app:app --host 0.0.0.0 --port 8000 "$@" ;;
  dwi) shift; exec python -m ads.cli dwi "$@" ;;
  pwi) shift; exec python -m ads.cli pwi "$@" ;;
  combined) shift; exec python scripts/run_ads_combined.py "$@" ;;
  cli) shift; exec python -m ads.cli "$@" ;;
  gui) shift; exec python /app/GUI_launcher.py "$@" ;;
  -h|--help|help) usage; exit 0 ;;
  *) exec "$@" ;;
esac
