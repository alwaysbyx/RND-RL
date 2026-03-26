#!/bin/bash
# Wait for tmux session 'trpo_exp' to finish, then run ManiSkill TRPO experiments.

SESSION="trpo_exp"
POLL=60
SCRIPT="/home/yubian/research/RND-RL/scripts/train_trpo/run_trpo_maniskill.py"
LOG="/home/yubian/research/RND-RL/scripts/train_trpo/trpo_maniskill.log"
CONDA_ENV="rl"

echo "=============================================="
echo " Waiting for tmux session '$SESSION' to finish"
echo " Poll interval: ${POLL}s"
echo " Will run: $SCRIPT"
echo "=============================================="

while true; do
    # Check if session still exists
    if ! tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Session '$SESSION' no longer exists. Starting."
        break
    fi

    # Session exists — check if any python processes are still running in it
    PANE_PID=$(tmux list-panes -t "$SESSION" -F '#{pane_pid}' 2>/dev/null | head -1)
    if [ -n "$PANE_PID" ]; then
        CHILD_PYTHONS=$(pgrep -P "$PANE_PID" -a 2>/dev/null | grep -c python)
        if [ "$CHILD_PYTHONS" -eq 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] No python processes in '$SESSION'. Starting."
            break
        fi
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] '$SESSION' still running. Waiting ${POLL}s..."
    sleep "$POLL"
done

echo ""
echo "=============================================="
echo " Launching ManiSkill TRPO experiments"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

conda run --no-capture-output -n "$CONDA_ENV" python "$SCRIPT" --phase maniskill 2>&1 | tee "$LOG"
