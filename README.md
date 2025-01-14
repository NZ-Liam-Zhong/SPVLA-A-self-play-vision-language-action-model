# SPVLA: A Self-Play Vision-Language-Action Model
Author: Ningze Zhong, Bo Wu<br> 
Institution: MIT-IBM Watsons AI Lab, Sun Yat-sen University<br>

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
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/selfplay.py  --data_root_dir "/root/autodl-tmp/modified_libero_rlds" --dataset_name libero_spatial_no_noops --run_root_dir "/root/autodl-tmp/spatial-self-fintune-001" --adapter_tmp_dir "/root/autodl-tmp/spatial-self-fintune-weight-001" --lora_rank 32 --batch_size 20 --grad_accumulation_steps 1 --learning_rate 5e-4 --image_aug False --save_steps 1000
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
  --pretrained_checkpoint /root/autodl-tmp/spatial-self-fintune-new2/try_ok_new\
  --task_suite_name libero_spatial \
  --center_crop True

  torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/selfplay.py  --data_root_dir "/root/autodl-tmp/modified_libero_rlds" --dataset_name libero_spatial_no_noops --run_root_dir "/root/autodl-tmp/spatial-self-fintune-new2" --adapter_tmp_dir "/root/autodl-tmp/spatial-self-fintune-weight-new2" --lora_rank 32 --batch_size 20 --grad_accumulation_steps 1 --learning_rate 2e-4 --image_aug False --save_steps 500
<br><br>
没问题标准的<br><br>
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint /root/autodl-tmp/spatial-self-fintune-new2/try_ok_new\
  --task_suite_name libero_spatial \
  --center_crop True<br><br>
  /root/autodl-tmp/openvla-7b-finetuned-libero-spatial<br><br>
  python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint /root/autodl-tmp/openvla-7b-finetuned-libero-spatial
  --task_suite_name libero_spatial \
  --center_crop True

  <br><br>
  (openvla) root@autodl-container-852f45bfea-b1111753:~/znz/openvla#  python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint /root/autodl-tmp/openvla-7b-finetuned-libero-spatial
  --task_suite_name libero_spatial \
  --center_crop True
2024-12-25 21:32:38.947795: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2024-12-25 21:32:38.979465: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-12-25 21:32:38.979500: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-12-25 21:32:38.980982: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2024-12-25 21:32:38.987062: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI AVX512_BF16 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-12-25 21:32:39.681459: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
[robosuite WARNING] No private macro file found! (macros.py:53)
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[robosuite WARNING] To setup, run: python /root/miniconda3/envs/openvla/lib/python3.10/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
2024-12-25 21:32:43.137142: W tensorflow/core/common_runtime/gpu/gpu_device.cc:2348] TensorFlow was not built with CUDA kernel binaries compatible with compute capability 9.0. CUDA kernels will be jit-compiled from PTX, which could take 30 minutes or longer.
 GenerateConfig(model_family='openvla', pretrained_checkpoint='/root/autodl-tmp/openvla-7b-finetuned-libero-spatial', load_in_8bit=False, load_in_4bit=False, center_crop=True, task_suite_name='libero_spatial', num_steps_wait=10, num_trials_per_task=50, run_id_note=None, local_log_dir='./experiments/logs', use_wandb=False, wandb_project='YOUR_WANDB_PROJECT', wandb_entity='YOUR_WANDB_ENTITY', seed=7)
[*] Instantiating Pretrained VLA model
[*] Loading in BF16 with Flash-Attention Enabled
<frozen importlib._bootstrap>:283: DeprecationWarning: the load_module() method is deprecated and slated for removal in Python 3.12; use exec_module() instead
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:01<00:00,  3.33it/s]
Loaded model: <class 'transformers_modules.openvla.openvla-7b.31f090d05236101ebfc381b61c674dd4746d4ce0.modeling_prismatic.OpenVLAForActionPrediction'>
Logging to local log file: ./experiments/logs/EVAL-libero_spatial-openvla-2024_12_25-21_32_51.txt
[info] using task orders [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Task suite: libero_spatial
  0%|                                                                                                                                                                                                                             | 0/10 [00:00<?, ?it/s][Warning]: datasets path /root/znz/openvla/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /root/znz/openvla/LIBERO/libero/libero/../datasets does not exist!

Task: pick up the black bowl between the plate and the ramekin and place it on the plate                                                                                                                                          | 0/50 [00:00<?, ?it/s]
Starting episode 1...
Floating point exception (core dumped)
bash: --task_suite_name: command not found



<br><br>


torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/selfplay.py  --data_root_dir "/root/autodl-tmp/modified_libero_rlds" --dataset_name libero_object_no_noops --run_root_dir "/root/autodl-tmp/object-self-fintune-001" --adapter_tmp_dir "/root/autodl-tmp/object-self-fintune-weight-001" --lora_rank 32 --batch_size 20 --grad_accumulation_steps 1 --learning_rate 5e-4 --image_aug False --save_steps 1000

<br><br>

normalized_loss tensor(0.3668, device='cuda:0', grad_fn=<DivBackward0>)
action logit max tensor(80., device='cuda:0', grad_fn=<MaxBackward1>)
action logit min tensor(-30.3750, device='cuda:0', grad_fn=<MinBackward1>)
policy_generated_logps tensor([-0.5363, -0.4312, -0.4809, -0.4551, -0.4214, -0.4969, -0.5220, -0.4139,
        -0.4825, -0.5190, -0.4813, -0.3952, -0.4699, -0.5303, -0.5558, -0.4357,
        -0.5106, -0.5337, -0.5381, -0.4857], device='cuda:0',
       grad_fn=<DivBackward0>)
policy_real_logps tensor([-1.5006e+00, -1.0885e-02, -8.5359e-03, -2.0855e-02, -1.2409e-04,
        -5.3313e-01, -5.3277e-03, -1.1909e+00, -7.7429e-02, -1.0546e+00,
        -1.1074e+00, -3.9870e-03, -4.1570e-03, -1.7525e-01, -1.6335e-02,
        -6.3176e-01, -5.1611e-01, -4.3150e-01, -2.5046e-02, -2.2378e-02],
       device='cuda:0', grad_fn=<DivBackward0>)
opponent_generated_logps tensor([-0.5363, -0.4312, -0.4809, -0.4551, -0.4214, -0.4969, -0.5220, -0.4139,
        -0.4825, -0.5190, -0.4813, -0.3952, -0.4699, -0.5303, -0.5558, -0.4357,
        -0.5106, -0.5337, -0.5381, -0.4857], device='cuda:0')
opponent_real_logps tensor([-1.5006e+00, -1.0885e-02, -8.5359e-03, -2.0855e-02, -1.2409e-04,
        -5.3313e-01, -5.3277e-03, -1.1909e+00, -7.7429e-02, -1.0546e+00,
        -1.1074e+00, -3.9870e-03, -4.1570e-03, -1.7525e-01, -1.6335e-02,
        -6.3176e-01, -5.1611e-01, -4.3150e-01, -2.5046e-02, -2.2378e-02],
       device='cuda:0')

<br><br><br>




torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/selfplay.py  --data_root_dir "/root/autodl-fs/modified_libero_rlds" --dataset_name libero_spatial_no_noops --run_root_dir "/root/autodl-tmp/spatial-self-fintune-001" --adapter_tmp_dir "/root/autodl-tmp/spatial-self-fintune-weight-001" --lora_rank 32 --batch_size 20 --grad_accumulation_steps 1 --learning_rate 5e-4 --image_aug False --save_steps 1000
<br><br>


python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint /root/autodl-fs/spatial-self-fintune-002/002 \
  --task_suite_name libero_spatial \
  --center_crop True

重新进行训练
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/selfplay.py --data_root_dir "/root/autodl-fs/modified_libero_rlds" --dataset_name libero_spatial_no_noops --run_root_dir "/root/autodl-tmp/spatial-self-fintune-002" --adapter_tmp_dir "/root/autodl-tmp/spatial-self-fintune-weight-002" --lora_rank 32 --batch_size 20 --grad_accumulation_steps 1 --learning_rate 5e-4 --image_aug False --save_steps 1000 



<br><br>
1.wandb看loss和梯度曲线<br>
2.loss平滑项这样可以没那么抖<br>
3.loss的gradient norm曲线<br>

1000组在
/root/autodl-tmp/spatial-self-fintune-002/results
损失函数
![图片](https://github.com/user-attachments/assets/5d9c3091-2ef2-4382-9b74-212d57aef5e6)
evaluate:<br>
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint /root/autodl-tmp/spatial-self-fintune-002/results \
  --task_suite_name libero_spatial \
  --center_crop True


