# SPVLA: A Self-Play Vision-Language-Action Model
Author: Ningze Zhong, Bo Wu<br>
Institution: MIT-IBM Watsons AI Lab, Sun Yat-sen University

This repo is adopted from [Openvla: An Open-Source Vision-Language-Action Model](https://github.com/openvla/openvla)

**Run the Self-play code(example)** <br>
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/selfplay.py  --data_root_dir "/root/autodl-tmp/modified_libero_rlds" --dataset_name libero_object_no_noops --run_root_dir "/root/znz/openvla/object-self-fintune" --adapter_tmp_dir "/root/znz/openvla/object-self-fintune-weight" --lora_rank 32 --batch_size 20 --grad_accumulation_steps 1 --learning_rate 5e-4 --image_aug False --save_steps 1000

object-it1-ckpt:
