#!/usr/bin/env bash
source /path/to/anaconda3/etc/profile.d/conda.sh
conda activate emdnerf
cd /path/to/EMD-NeRF
export PYTHONPATH=Train:${PYTHONPATH}
current_time=$(date +%s)
scene=710
seed=0
python /path/to/EMD-NeRF/run_emdnerf.py \
--now $current_time \
--name replicate \
--data_dir /path/to/datasets/scannet \
--num_workers 32 \
--task train \
--ckpt_dir emdnerf_replicate_results${seed} \
--expname scene${scene} \
--scene_id scene0${scene}_00 \
--num_iterations 500000 \
--N_rand 1024 \
--depth_weight 0.007 \
--hypo_model ddp \
--seed ${seed} \
--uncertainty_dir ddp_uncertainty \
--gamma 1