#!/bin/bash
# Run all phases in order.
# Edit N_SEEDS, N_EPISODES, and GPU settings as needed.

set -e

N_EPISODES=10000
N_PRETRAIN_SEEDS=10
N_META_SEEDS=3
TOTAL_PRETRAIN_STEPS=256000
TOTAL_META_STEPS=64000
TOTAL_RL_STEPS=100000
DATA_DIR=data
CKPT_DIR=checkpoints

# Detect available GPUs; fall back to 1 if nvidia-smi unavailable
N_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
if [ "$N_GPUS" -lt 1 ]; then N_GPUS=1; fi
echo "Using $N_GPUS GPU(s) for parallelism"

# Minimum free GPU memory required to launch a job (in MiB)
# batch_size=128 + grad_accum=8 needs ~1.5-2 GB; add headroom for fragmentation
MIN_FREE_MiB=3000

# Maximum concurrent jobs per GPU
MAX_JOBS_PER_GPU=1

# Track running jobs per GPU using a temp directory for lock files
GPU_SLOTS_DIR=$(mktemp -d)
trap "rm -rf $GPU_SLOTS_DIR" EXIT

# Pick a GPU with enough free memory AND fewer than MAX_JOBS_PER_GPU running jobs.
# Waits until a slot is available. Prints the GPU index to stdout.
pick_gpu() {
    while true; do
        # Sort GPUs by free memory descending — prefer the one with most headroom
        local best_gpu=""
        local best_free=0
        while IFS=', ' read -r idx free_mem; do
            # Count current running jobs on this GPU (robust: count files via glob)
            local running=0
            for _f in "$GPU_SLOTS_DIR"/gpu_${idx}_*; do
                [ -e "$_f" ] && running=$((running + 1))
            done
            if [ "$running" -lt "$MAX_JOBS_PER_GPU" ] && [ "$free_mem" -ge "$MIN_FREE_MiB" ]; then
                if [ "$free_mem" -gt "$best_free" ]; then
                    best_gpu="$idx"
                    best_free="$free_mem"
                fi
            fi
        done < <(nvidia-smi --query-gpu=index,memory.free \
                    --format=csv,noheader,nounits 2>/dev/null)
        if [ -n "$best_gpu" ]; then
            echo "$best_gpu"
            return
        fi
        sleep 5
    done
}

# Register a job on a GPU (call after pick_gpu, before launching)
register_gpu_job() {
    local gpu=$1
    local pid=$2
    touch "$GPU_SLOTS_DIR/gpu_${gpu}_${pid}"
}

# Clean up finished GPU job slots (call periodically)
cleanup_gpu_slots() {
    for f in "$GPU_SLOTS_DIR"/gpu_*; do
        [ -e "$f" ] || continue
        local pid=$(basename "$f" | sed 's/gpu_[0-9]*_//')
        if ! kill -0 "$pid" 2>/dev/null; then
            rm -f "$f"
        fi
    done
}

# Launch a job on the best available GPU. Usage: launch_on_gpu <command...>
# Sets CUDA_VISIBLE_DEVICES and tracks the slot.
# Returns GPU index via the global variable LAUNCHED_GPU.
launch_on_gpu() {
    cleanup_gpu_slots
    local gpu=$(pick_gpu)
    CUDA_VISIBLE_DEVICES=$gpu "$@" &
    local pid=$!
    register_gpu_job "$gpu" "$pid"
    LAUNCHED_GPU=$gpu
    # Small delay so nvidia-smi can see the allocation before next pick
    sleep 2
}

echo "=== Phase 1+2: Generate Expert Data ==="
python data_gen.py --split pretrain --n_episodes $N_EPISODES --out_dir $DATA_DIR
python data_gen.py --split posttrain --n_episodes 1000 --out_dir $DATA_DIR

echo "=== Phase 3: Pretrain Transformer (10 seeds) ==="
for seed in $(seq 0 $((N_PRETRAIN_SEEDS - 1))); do
    launch_on_gpu python train_pretrain.py \
        --data_path $DATA_DIR/pretrain.pkl \
        --save_dir $CKPT_DIR/pretrain \
        --seed $seed \
        --lam 0.01 \
        --total_steps $TOTAL_PRETRAIN_STEPS \
        --batch_size 128 \
        --grad_accum 8
    echo "  seed=$seed -> GPU $LAUNCHED_GPU"
done
wait
echo "Pretraining done."

echo "=== Phase 3 Validation: Linear Probe ==="
python probe.py \
    --mode probe \
    --model_path $CKPT_DIR/pretrain/seed0_step${TOTAL_PRETRAIN_STEPS}.pt \
    --data_path $DATA_DIR/pretrain.pkl

echo "=== Phase 4: Supervised Controller ==="
python probe.py \
    --mode controller \
    --model_path $CKPT_DIR/pretrain/seed0_step${TOTAL_PRETRAIN_STEPS}.pt \
    --data_path $DATA_DIR/pretrain.pkl \
    --save_path $CKPT_DIR/controller.pt

echo "=== Phase 5: Metacontroller (3 seeds × 10 pretrained) ==="
for base_seed in $(seq 0 $((N_PRETRAIN_SEEDS - 1))); do
    for meta_seed in $(seq 0 $((N_META_SEEDS - 1))); do
        launch_on_gpu python metacontroller.py \
            --model_path $CKPT_DIR/pretrain/seed${base_seed}_step${TOTAL_PRETRAIN_STEPS}.pt \
            --data_path $DATA_DIR/pretrain.pkl \
            --save_dir $CKPT_DIR/metacontroller/base${base_seed} \
            --seed $meta_seed \
            --alpha 0.1 \
            --total_steps $TOTAL_META_STEPS
        echo "  base=$base_seed meta=$meta_seed -> GPU $LAUNCHED_GPU"
    done
done
wait
echo "Metacontroller training done."

echo "=== Phase 6: Internal RL (30 runs) ==="
for base_seed in $(seq 0 $((N_PRETRAIN_SEEDS - 1))); do
    for meta_seed in $(seq 0 $((N_META_SEEDS - 1))); do
        launch_on_gpu python internal_rl.py \
            --base_model_path $CKPT_DIR/pretrain/seed${base_seed}_step${TOTAL_PRETRAIN_STEPS}.pt \
            --meta_model_path $CKPT_DIR/metacontroller/base${base_seed}/seed${meta_seed}_step${TOTAL_META_STEPS}.pt \
            --save_dir $CKPT_DIR/internal_rl/base${base_seed}_meta${meta_seed} \
            --seed ${meta_seed} \
            --total_steps $TOTAL_RL_STEPS
        echo "  base=$base_seed meta=$meta_seed -> GPU $LAUNCHED_GPU"
    done
done
wait
echo "Internal RL done."

echo "=== Phase 7: Baselines ==="
for seed in 0 1 2; do
    launch_on_gpu python baselines.py \
        --baseline raw_action_rl \
        --base_model_path $CKPT_DIR/pretrain/seed${seed}_step${TOTAL_PRETRAIN_STEPS}.pt \
        --save_dir $CKPT_DIR/baselines \
        --seed $seed
    echo "  raw_action_rl seed=$seed -> GPU $LAUNCHED_GPU"

    launch_on_gpu python baselines.py \
        --baseline no_temporal \
        --base_model_path $CKPT_DIR/pretrain/seed${seed}_step${TOTAL_PRETRAIN_STEPS}.pt \
        --meta_model_path $CKPT_DIR/metacontroller/base${seed}/seed0_step${TOTAL_META_STEPS}.pt \
        --save_dir $CKPT_DIR/baselines \
        --seed $seed
    echo "  no_temporal seed=$seed -> GPU $LAUNCHED_GPU"
done
wait

echo "=== All phases complete ==="
