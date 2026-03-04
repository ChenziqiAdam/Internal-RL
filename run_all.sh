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
MIN_FREE_MiB=4000

# Pick a GPU with >= MIN_FREE_MiB free memory; waits until one is available.
# Prints the GPU index to stdout.
pick_gpu() {
    while true; do
        gpu=$(nvidia-smi --query-gpu=index,memory.free \
                --format=csv,noheader,nounits 2>/dev/null \
              | awk -F', ' -v min="$MIN_FREE_MiB" '$2 >= min {print $1; exit}')
        if [ -n "$gpu" ]; then
            echo "$gpu"
            return
        fi
        sleep 10
    done
}

echo "=== Phase 1+2: Generate Expert Data ==="
python data_gen.py --split pretrain --n_episodes $N_EPISODES --out_dir $DATA_DIR
python data_gen.py --split posttrain --n_episodes 1000 --out_dir $DATA_DIR

echo "=== Phase 3: Pretrain Transformer (10 seeds) ==="
for seed in $(seq 0 $((N_PRETRAIN_SEEDS - 1))); do
    gpu=$(pick_gpu)
    echo "  seed=$seed -> GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python train_pretrain.py \
        --data_path $DATA_DIR/pretrain.pkl \
        --save_dir $CKPT_DIR/pretrain \
        --seed $seed \
        --lam 0.01 \
        --total_steps $TOTAL_PRETRAIN_STEPS \
        --batch_size 1024 \
        --grad_accum 1 &
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
        gpu=$(pick_gpu)
        echo "  base=$base_seed meta=$meta_seed -> GPU $gpu"
        CUDA_VISIBLE_DEVICES=$gpu python metacontroller.py \
            --model_path $CKPT_DIR/pretrain/seed${base_seed}_step${TOTAL_PRETRAIN_STEPS}.pt \
            --data_path $DATA_DIR/pretrain.pkl \
            --save_dir $CKPT_DIR/metacontroller/base${base_seed} \
            --seed $meta_seed \
            --alpha 0.1 \
            --total_steps $TOTAL_META_STEPS &
    done
done
wait
echo "Metacontroller training done."

echo "=== Phase 6: Internal RL (30 runs) ==="
for base_seed in $(seq 0 $((N_PRETRAIN_SEEDS - 1))); do
    for meta_seed in $(seq 0 $((N_META_SEEDS - 1))); do
        gpu=$(pick_gpu)
        echo "  base=$base_seed meta=$meta_seed -> GPU $gpu"
        CUDA_VISIBLE_DEVICES=$gpu python internal_rl.py \
            --base_model_path $CKPT_DIR/pretrain/seed${base_seed}_step${TOTAL_PRETRAIN_STEPS}.pt \
            --meta_model_path $CKPT_DIR/metacontroller/base${base_seed}/seed${meta_seed}_step${TOTAL_META_STEPS}.pt \
            --save_dir $CKPT_DIR/internal_rl/base${base_seed}_meta${meta_seed} \
            --seed ${meta_seed} \
            --total_steps $TOTAL_RL_STEPS &
    done
done
wait
echo "Internal RL done."

echo "=== Phase 7: Baselines ==="
for seed in 0 1 2; do
    gpu=$(pick_gpu)
    echo "  raw_action_rl seed=$seed -> GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python baselines.py \
        --baseline raw_action_rl \
        --base_model_path $CKPT_DIR/pretrain/seed${seed}_step${TOTAL_PRETRAIN_STEPS}.pt \
        --save_dir $CKPT_DIR/baselines \
        --seed $seed &

    gpu=$(pick_gpu)
    echo "  no_temporal seed=$seed -> GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python baselines.py \
        --baseline no_temporal \
        --base_model_path $CKPT_DIR/pretrain/seed${seed}_step${TOTAL_PRETRAIN_STEPS}.pt \
        --meta_model_path $CKPT_DIR/metacontroller/base${seed}/seed0_step${TOTAL_META_STEPS}.pt \
        --save_dir $CKPT_DIR/baselines \
        --seed $seed &
done
wait

echo "=== All phases complete ==="
