#!/usr/bin/env bash
# Launch / resume a SAR-AIRcraft D-FINE run under the frozen protocol, in the background.
#
#   bash tools/train_sar_aircraft.sh <config.yml> [gpus=0,1]     # new run
#   bash tools/train_sar_aircraft.sh --resume <run_dir> [gpus]    # resume from <run_dir>/last.pth
#
# Global batch is fixed in the config (16); with 2 GPUs that is 8 per GPU (~21 GB peak at
# 1280 multi-scale). The server is shared: the script aborts if a selected GPU is in use.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
GIT_REV=$(git -C "$REPO" rev-parse HEAD) || { echo "cannot read git revision of $REPO"; exit 1; }
EXP_ROOT=${EXP_ROOT:-/root/zxy/experiments}
INIT=${INIT:-/root/zxy/pretrain/dfine/dfine_m_obj2coco_airplane7.pth}
SEED=${SEED:-0}

if [[ "${1:-}" == "--resume" ]]; then
    OUT=$(cd "$2" && pwd); GPUS=${3:-0,1}
    CFG="$OUT/source_config.txt"; CFG=$(cat "$CFG")
    [[ -f "$OUT/last.pth" ]] || { echo "no $OUT/last.pth"; exit 1; }
    MODE=resume
else
    CFG=${1:?config path}; GPUS=${2:-0,1}
    NAME=$(basename "$CFG" .yml)
    OUT="$EXP_ROOT/${NAME}_seed${SEED}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
    [[ -e "$OUT" ]] && { echo "$OUT exists"; exit 1; }
    MODE=new
fi
NPROC=$(awk -F, '{print NF}' <<< "$GPUS")

for g in ${GPUS//,/ }; do
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
    if (( used > 1000 )); then echo "GPU $g busy (${used} MiB used), aborting"; exit 1; fi
done

cd "$REPO"
set +u  # conda's activate scripts reference unset variables
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate dfine_zxy
set -u
ulimit -Sn 65535

TS=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)
if [[ $MODE == new ]]; then
    mkdir -p "$OUT"
    echo "$CFG" > "$OUT/source_config.txt"
    echo "$GIT_REV" > "$OUT/git_revision.txt"
    git status --short > "$OUT/git_status.txt"
    python - "$CFG" "$OUT/resolved_config.yml" <<'PY'
import sys, yaml
from src.core import YAMLConfig
cfg = YAMLConfig(sys.argv[1]).yaml_cfg
cfg.pop("__include__", None)
open(sys.argv[2], "w").write(yaml.safe_dump(cfg, sort_keys=False))
PY
    ARGS=(-c "$CFG" -t "$INIT" --use-amp --seed "$SEED" --output-dir "$OUT")
    LOG="$OUT/console.log"
else
    ARGS=(-c "$CFG" -r "$OUT/last.pth" --use-amp --seed "$SEED" --output-dir "$OUT")
    LOG="$OUT/resume_$TS.console.log"
fi

CMD="CUDA_VISIBLE_DEVICES=$GPUS torchrun --standalone --nproc_per_node=$NPROC train.py ${ARGS[*]}"
echo "$TS $MODE gpus=$GPUS init=$INIT seed=$SEED" >> "$OUT/launch_history.txt"
echo "$CMD" >> "$OUT/launch_history.txt"
nvidia-smi > "$OUT/nvidia_smi_$TS.txt"

nohup env CUDA_VISIBLE_DEVICES="$GPUS" PYTHONUNBUFFERED=1 \
    torchrun --standalone --nproc_per_node="$NPROC" train.py "${ARGS[@]}" \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT/launcher.pid"
disown "$PID"
echo "started pid=$PID"
echo "OUT=$OUT"
echo "tail -f $LOG"
