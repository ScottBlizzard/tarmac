#!/usr/bin/env bash
set -euo pipefail

status_file="${1:?usage: $0 STATUS_FILE PARENT_PID}"
parent_pid="${2:?usage: $0 STATUS_FILE PARENT_PID}"

while true; do
  state="$(cat "${status_file}" 2>/dev/null || true)"
  case "${state}" in
    complete*|failed*)
      kill -CONT "${parent_pid}" 2>/dev/null || true
      exit 0
      ;;
    *) sleep 20 ;;
  esac
done
