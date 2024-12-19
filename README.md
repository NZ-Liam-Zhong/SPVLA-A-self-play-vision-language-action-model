# SPVLA: A Self-Play Vision-Language-Action Model
Author: Ningze Zhong, Bo Wu<br>
Institution: MIT-IBM Watsons AI Lab, Sun Yat-sen University

This repo is adopted from [Openvla: An Open-Source Vision-Language-Action Model](https://github.com/openvla/openvla)

**Run the Self-play code(example)** <br>
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/selfplay.py  --data_root_dir "/root/autodl-tmp/modified_libero_rlds" --dataset_name libero_object_no_noops --run_root_dir "/root/autodl-tmp/object-self-fintune" --adapter_tmp_dir "/root/autodl-tmp/object-self-fintune-weight" --lora_rank 32 --batch_size 20 --grad_accumulation_steps 1 --learning_rate 5e-4 --image_aug False --save_steps 1000
<br><br>
object-it1-ckpt:<br>
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint /root/autodl-tmp/object-self-fintune/object-selfplay-post-training-it-0-batch20-steps1000 \
  --task_suite_name libero_object \
  --center_crop True
<br>
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/selfplay.py  --data_root_dir "/root/autodl-tmp/modified_libero_rlds" --dataset_name libero_object_no_noops --run_root_dir "/root/autodl-tmp/object-self-fintune" --adapter_tmp_dir "/root/autodl-tmp/object-self-fintune-weight" --lora_rank 32 --batch_size 20 --grad_accumulation_steps 1 --learning_rate 5e-4 --image_aug False --save_steps 500
<br><br>

**发现问题**<br>
最大最小值在selfplay时候变化很大
从 <br>
action logit max tensor(81.5000, device='cuda:0', grad_fn=<MaxBackward1>)<br>
action logit min tensor(-30.3750, device='cuda:0', grad_fn=<MinBackward1>)<br>
到<br>
action logit max tensor(32.5000, device='cuda:0', grad_fn=<MaxBackward1>)<br>
action logit min tensor(-13.5625, device='cuda:0', grad_fn=<MinBackward1>)<br>
