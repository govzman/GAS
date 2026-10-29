#!/usr/bin/env bash
# Source this helper from an experiment launcher; do not execute it directly.

_gas_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_gas_env_file="${_gas_repo_root}/.env"

if [[ ! -f "${_gas_env_file}" ]]; then
  echo "Missing ${_gas_env_file}. Copy .env.example to .env and set COMET_API_KEY." >&2
  return 1
fi

set -a
# shellcheck disable=SC1090
source "${_gas_env_file}"
set +a

if [[ -z "${COMET_API_KEY:-}" ]]; then
  echo "COMET_API_KEY is empty in ${_gas_env_file}." >&2
  return 1
fi
