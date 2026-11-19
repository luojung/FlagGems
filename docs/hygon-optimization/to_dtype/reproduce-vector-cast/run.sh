#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 3 ]]; then
    echo "Usage: DEV=... DTK_ROOT=... bash run.sh BASE_CHECKOUT NEW_OUTPUT_DIR CARD" >&2
    exit 2
fi
: "${DEV:?Set DEV to the FlagOS development environment}"
: "${DTK_ROOT:?Set DTK_ROOT to the DTK 26.04 installation}"
scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
baseline=$(realpath -e -- "$1")
card=$3
tree=7d6ebe9c65b06ab3917fdc76142920f042d668f1
[[ $card =~ ^[0-9]+$ ]]
[[ $(git -C "$baseline" rev-parse HEAD:src) == "$tree" ]] || {
    echo "The baseline checkout must have src tree $tree" >&2
    exit 2
}
[[ -z $(git -C "$baseline" status --porcelain -- src) ]] || {
    echo "The baseline src directory must be clean" >&2
    exit 2
}

umask 077
mkdir -- "$2"
cd -- "$2"
export FLAGOS_DEV_CACHE="${FLAGOS_DEV_CACHE:-$PWD/cache}"
mkdir -p profiles "$FLAGOS_DEV_CACHE"
export FLAGOS_DEV_CACHE=$(realpath -e -- "$FLAGOS_DEV_CACHE")
export HIP_VISIBLE_DEVICES=$card ROCR_VISIBLE_DEVICES=$card CUDA_VISIBLE_DEVICES=$card
cp -- "$scripts"/{cases.json,worker.py,model_worker.py,analyze.py,model_analyze.py,summarize.py,dtype_hip_vectors.hip} .
"$DTK_ROOT/bin/hipcc" -O3 -fPIC -shared --offload-arch=gfx936 dtype_hip_vectors.hip -o dtype_hip_vectors.so

profile() {
    local kind=$1 worker=$2 analysis=$3 rows=$4
    bash "$DEV/bin/run" --profile boxing --source "$baseline" -- \
        env DTK_ROOT="$DTK_ROOT" "$DTK_ROOT/bin/hipprof" --hip-trace --hiptx-trace --output-type 0 \
        -o "$PWD/profiles/$kind" python -u "$worker" \
        --name "$kind" --cases "$rows" --physical-card "$card" 2>&1 | tee "$kind.log"
    grep -qx DTYPE_COMPARE_COMPLETE "$kind.log"
    python3 "$analysis" "$PWD" "$kind" "$kind-records.jsonl"
}

profile model model_worker.py model_analyze.py "${MODEL_ROWS:-$(seq -s, 771 780)}"
profile scale worker.py analyze.py "${SCALE_ROWS:-$(seq -s, 1056 1080)}"
python3 summarize.py "$PWD"
