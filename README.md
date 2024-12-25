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
为了利用好激活函数softmax，应该最少限制在-10到10之间<br><br>

**logits/100且打印normalized**<br>
打印看看<br> 刚开始：
normalized_loss tensor(14.6523, device='cuda:0', grad_fn=<DivBackward0>)
action logit max tensor(81.5000, device='cuda:0', grad_fn=<MaxBackward1>)
action logit min tensor(-30.7500, device='cuda:0', grad_fn=<MinBackward1>)
policy_generated_logps tensor([-7.0163, -6.6541, -6.9303, -6.9013, -7.1450, -6.6807, -6.7530, -6.6553,
        -6.7315, -6.8726, -6.7445, -6.8804, -6.7395, -6.7113, -6.6961, -6.7386,
        -6.8094, -6.4930, -7.0677, -6.8346], device='cuda:0',
       grad_fn=<DivBackward0>)
policy_real_logps tensor([-6.7637, -7.2877, -7.7926, -7.6746, -7.2455, -7.5221, -7.2654, -6.8276,
        -7.2140, -7.5758, -7.1417, -7.6933, -6.9663, -7.2616, -7.6227, -7.8281,
        -7.8090, -7.2046, -8.2798, -7.2241], device='cuda:0',
       grad_fn=<DivBackward0>)
opponent_generated_logps tensor([-7.0163, -6.6541, -6.9303, -6.9013, -7.1450, -6.6807, -6.7530, -6.6553,
        -6.7315, -6.8726, -6.7445, -6.8804, -6.7395, -6.7113, -6.6961, -6.7386,
        -6.8094, -6.4930, -7.0677, -6.8346], device='cuda:0')
opponent_real_logps tensor([-6.7637, -7.2877, -7.7926, -7.6746, -7.2455, -7.5221, -7.2654, -6.8276,
        -7.2140, -7.5758, -7.1417, -7.6933, -6.9663, -7.2616, -7.6227, -7.8281,
        -7.8090, -7.2046, -8.2798, -7.2241], device='cuda:0')
losses tensor(0.6931, device='cuda:0', grad_fn=<MeanBackward0>) <br><br>
后面<br>
normalized_loss tensor(11.5913, device='cuda:0', grad_fn=<DivBackward0>)
action logit max tensor(123., device='cuda:0', grad_fn=<MaxBackward1>)
action logit min tensor(-109., device='cuda:0', grad_fn=<MinBackward1>)
policy_generated_logps tensor([-8.5292, -8.5315, -8.5918, -8.5074, -8.5899, -8.4077, -8.6029, -8.4920,
        -8.5417, -8.5515, -8.5615, -8.4028, -8.5031, -8.5403, -8.5087, -8.5621,
        -8.4173, -8.5794, -8.5269, -8.4724], device='cuda:0',
       grad_fn=<DivBackward0>)
policy_real_logps tensor([-3.4695, -2.7139, -3.5787, -2.7079, -2.9757, -2.4497, -3.3944, -3.2402,
        -3.8402, -3.3026, -3.5874, -2.2434, -2.1681, -3.2883, -3.5046, -3.2906,
        -2.7181, -3.2841, -2.4803, -2.4318], device='cuda:0',
       grad_fn=<DivBackward0>)
opponent_generated_logps tensor([-6.7702, -6.7520, -7.0657, -6.6429, -6.8517, -7.1741, -7.1349, -6.9976,
        -6.7092, -6.8655, -7.0010, -7.0303, -6.8468, -6.7418, -6.8808, -6.9920,
        -7.0775, -6.6567, -6.6924, -6.8059], device='cuda:0')
opponent_real_logps tensor([-7.7851, -7.4338, -7.6182, -6.8065, -7.3121, -7.5605, -8.1175, -7.3400,
        -7.4286, -7.4237, -7.4137, -7.8213, -7.4238, -7.4983, -7.6964, -7.7987,
        -7.8672, -7.5173, -7.1686, -7.3622], device='cuda:0')


Mbr><br>需要解决训练的不稳定性
<br>**证明self-play的闭式解**<br>
![图片](https://github.com/user-attachments/assets/0b6d7a30-1276-4863-8240-a72c61cb2740)
<br><br><br>
beta设置为0.7<br><br>
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint /root/autodl-tmp/object-self-fintune/object-selfplay-post-training-it-0-batch20-steps1000 \
  --task_suite_name libero_object \
  --center_crop True

  torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/selfplay.py  --data_root_dir "/root/autodl-tmp/modified_libero_rlds" --dataset_name libero_object_no_noops --run_root_dir "/root/autodl-tmp/object-self-fintune-new2" --adapter_tmp_dir "/root/autodl-tmp/object-self-fintune-weight-new2" --lora_rank 32 --batch_size 20 --grad_accumulation_steps 1 --learning_rate 2e-4 --image_aug False --save_steps 500
<br><br>

