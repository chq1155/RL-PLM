#!/usr/bin/env bash

set -euo pipefail

GPU_IDS=${GPU_IDS:-"0,1"}
MEMORY_THRESHOLD_MB=${MEMORY_THRESHOLD_MB:-1000}
REQUIRED_FREE_GPUS=${REQUIRED_FREE_GPUS:-1}
CHECK_INTERVAL_SEC=${CHECK_INTERVAL_SEC:-10}
COMMAND=${1:-}

if [[ -z "${COMMAND}" ]]; then
  echo "Usage: GPU_IDS=0,1 $0 \"python grpo.py --model-path ...\""
  exit 1
fi

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"

echo "Monitoring GPUs ${GPU_IDS} (threshold ${MEMORY_THRESHOLD_MB} MB, need ${REQUIRED_FREE_GPUS} free)."
echo "Polling every ${CHECK_INTERVAL_SEC}s. Press Ctrl+C to exit."

while true; do
  usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_IDS}")
  mapfile -t USAGE_ARRAY <<< "${usage}"

  free_count=0
  status=()
  for idx in "${!USAGE_ARRAY[@]}"; do
    mem=${USAGE_ARRAY[idx]}
    gpu=${GPU_LIST[idx]}
    status+=("GPU${gpu}:${mem}MB")
    if (( mem < MEMORY_THRESHOLD_MB )); then
      ((free_count++))
    fi
  done

  printf "%s - %s | free=%d\n" "$(date '+%H:%M:%S')" "${status[*]}" "${free_count}"

  if (( free_count >= REQUIRED_FREE_GPUS )); then
    echo "Launching: ${COMMAND}"
    bash -lc "${COMMAND}"
    echo "Command finished. Continuing monitoring..."
  fi

  sleep "${CHECK_INTERVAL_SEC}"
done
