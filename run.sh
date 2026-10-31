#!/usr/bin/env bash
# ./run.sh [-c] [-d] [--quick] [--jobs N]        whole pipeline (scripts 01 to 11)
# ./run.sh [-c] [-d] scripts/NN_name.py [args]    one script
# ./run.sh [-c] test | build | status | stop
#   -c  run inside container/segbench.sif    -d  run detached, output in logs/
set -euo pipefail
cd "$(dirname "$0")"
ENV=(PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1)
export "${ENV[@]}"
SIF=container/segbench.sif

container=0
detach=0
while [ $# -gt 0 ]; do
    case "$1" in
        -c) container=1 ;;
        -d) detach=1 ;;
        *) break ;;
    esac
    shift
done

case "${1:-}" in
    build) exec apptainer build --fakeroot --force "$SIF" container/segbench.def ;;
    status) ps -o pid,etime,%cpu,%mem,cmd -s "$(cat logs/latest.pid)"; tail -n 3 logs/latest.log; exit 0 ;;
    stop) pg=$(cat logs/latest.pid); kill -TERM -- "-$pg"; while ps -s "$pg" > /dev/null; do sleep 1; done; echo "=== stopped: $(date -Is)" >> logs/latest.log; exit 0 ;;
esac

if [ "$detach" = 1 ]; then
    mkdir -p logs
    name="logs/run_$(date +%Y%m%d_%H%M%S)"
    args=("$@")
    if [ "$container" = 1 ]; then args=(-c ${args[@]+"${args[@]}"}); fi
    echo "=== started: $(date -Is) on $(hostname): ./run.sh ${args[*]-}" > "$name.log"
    setsid nohup bash -c 'echo $$ > "$0.pid"; ./run.sh "$@"; echo "=== finished: $(date -Is), exit status $?"' \
        "$name" ${args[@]+"${args[@]}"} >> "$name.log" 2>&1 < /dev/null &
    ln -sfn "${name#logs/}.log" logs/latest.log
    ln -sfn "${name#logs/}.pid" logs/latest.pid
    echo "running in the background: tail -f logs/latest.log | ./run.sh status | ./run.sh stop"
    exit 0
fi

py=(python3)
if [ "$container" = 1 ]; then
    opts=()
    for e in "${ENV[@]}"; do opts+=(--env "$e"); done
    py=(apptainer exec --cleanenv --bind "$PWD" "${opts[@]}" "$SIF" python3)
fi

if [ "${1:-}" = test ]; then exec "${py[@]}" -m pytest tests; fi
if [[ "${1:-}" == *.py ]]; then exec "${py[@]}" "$@"; fi
for s in 01_generate_benchmark 02_evaluate 03_noise 04_sensitivity 05_runtime 06_agreement \
         07_controllers 08_figures 09_tables 10_videos 11_validate; do
    echo "=== $s"
    "${py[@]}" "scripts/$s.py" "$@"
done
echo "=== done"
