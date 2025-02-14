# SPVLA: A Self-Play Vision-Language-Action Model

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

对标的数据<br>
![图片](https://github.com/user-attachments/assets/ebfd5ecf-95a1-408d-beea-55b42717ba80)




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
python experiments/robot/libero/run_libero_eval.py --model_family openvla --pretrained_checkpoint /root/autodl-tmp/spatial-self-fintune-002/results --task_suite_name libero_spatial --center_crop True

<br>
# episodes completed so far: 500
# successes: 416 (83.2%)
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [30:41<00:00, 36.83s/it]
Current task success rate: 0.68██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [30:41<00:00, 41.17s/it]
Current total success rate: 0.832
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [4:16:42<00:00, 1540.20s/it]
Exception ignored in: <function MjRenderContext.__del__ at 0x7f268f25c3a0>
Traceback (most recent call last):

![图片](https://github.com/user-attachments/assets/4b628e17-ea04-477d-8f02-04263cfebbb9)\
![图片](https://github.com/user-attachments/assets/901d4667-2d0d-40fb-9c25-bcec2a86acab)\
![图片](https://github.com/user-attachments/assets/d5c6222a-602a-4bd2-9dbe-fe14c1d60a14)\
![图片](https://github.com/user-attachments/assets/6dbb4196-bdfa-455e-857c-94c138acb772)\
![图片](https://github.com/user-attachments/assets/b5bbe44d-9c7b-43bc-97fc-9c96b4a58913)\

<br><br><br>
参考这个改，改成跳过太难的任务\
\
# 定义 loss 阈值
loss_threshold = 1.0  # 根据需求调整\
device = losses.device\

# 筛选出有效的 losses（小于阈值）\
valid_mask = losses < loss_threshold\
filtered_losses = torch.where(valid_mask, losses, torch.tensor(0.0, device=device))\

# 计算有效 loss 的数量\
valid_loss_count = valid_mask.sum().item()\

# 动态调整 grad_accumulation_steps\
effective_grad_accumulation_steps = max(valid_loss_count, 1)  # 防止为 0\

# 打印调试信息\
print("Valid Loss Count:", valid_loss_count)\
print("Effective Grad Accumulation Steps:", effective_grad_accumulation_steps)\

# 归一化有效的 losses
normalized_filtered_losses = filtered_losses / effective_grad_accumulation_steps\

# 反向传播\
normalized_filtered_losses.backward()\

<br><br>

现在尝试skip hard <br>
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/selfplay_skip_hard.py  --data_root_dir "/root/autodl-fs/modified_libero_rlds" --dataset_name libero_spatial_no_noops --run_root_dir "/root/autodl-tmp/spatial-self-fintune-skiphard" --adapter_tmp_dir "/root/autodl-tmp/spatial-self-fintune-weight-skiphard" --lora_rank 32 --batch_size 20 --grad_accumulation_steps 1 --learning_rate 5e-4 --image_aug False --save_steps 1000

![image](https://github.com/user-attachments/assets/8b3b7ea1-4fae-4b1c-b73e-8f58fb6937a8)

loss_tensor大概在0.45~0.57之间

![image](https://github.com/user-attachments/assets/44ff17af-b8c4-4133-9e27-d57052d56350)


也可以看出图片处理成了pixel values为小数的格式

![image](https://github.com/user-attachments/assets/12d10f17-3c85-45a2-9c1d-e6ce5e9c481c)
更多结果

![image](https://github.com/user-attachments/assets/46d95c42-ae5b-4441-8800-ed5c41bcb8f8)

找到原因了！<br>
自博弈的loss<br>
![image](https://github.com/user-attachments/assets/1193fbdb-3a90-4342-a417-8fb2d1c9ebe9)<br>
来源于<br>
![image](https://github.com/user-attachments/assets/70e795da-8fbb-4e18-8746-a93075b4f667)
事实上可以看成减少两个分布之间的重要性采样，如果差别大就会方差大难以学习效果不好<br>
重要性采样定义<br>
![image](https://github.com/user-attachments/assets/8f0be1de-3617-4ff2-bdc5-f7dfff9c41bd)<br>
台湾大学hungyi lee课程<br>
![image](https://github.com/user-attachments/assets/8efe4c18-2774-4ce0-b70f-5cbe849be5b1)<br>
当两个差别大的时候会导致方差大，难以收敛<br>
尝试解决重要性采样的问题<br>
如何解决？正在阅读 https://math.arizona.edu/~tgk/mc/book_chap6.pdf


1.openvla (bc) 2.pi0(FM) 3.RDT(DP) 4.RT1(trans) 5.RT2(LLM + bc)

(1)RTX 8*H100 
Open X Embodiment: Generalist > Specialists?
Openvla RTX pretrained finetune LIEBRO


分析<br>
![image](https://github.com/user-attachments/assets/8f13b693-59c2-4fa2-aaa4-86cbf19c8bcf)<br><br>

![6cabf12042766aee40db3fb382ce8b7](https://github.com/user-attachments/assets/b211c065-47a6-453b-a9d8-f6e0e1250055)<br>
![a0cd59f933bfbd20424b2f6d806433b](https://github.com/user-attachments/assets/4f91e46a-3e01-40a8-9f73-2c7d92a334b0)<br>
![b3e928f9f420d2a50d1a2b957a4b874](https://github.com/user-attachments/assets/2a1b3f68-240c-4971-af86-cd9aa0bf6d93)<br>


<br>加了entropy之后<br>
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/selfplay.py --data_root_dir "/root/autodl-fs/modified_libero_rlds" --dataset_name libero_spatial_no_noops --run_root_dir "/root/autodl-fs/spatial-self-fintune-entropy" --adapter_tmp_dir "/root/autodl-fs/spatial-self-fintune-weight-entropy" --lora_rank 32 --batch_size 20 --grad_accumulation_steps 1 --learning_rate 5e-4 --image_aug False --save_steps 100

beta=0.6 eta=0.1<br>
![image](https://github.com/user-attachments/assets/1be6e66b-343c-4caf-aade-9723622da771)
![image](https://github.com/user-attachments/assets/c6e3a2da-79db-43dc-badf-b1e396cd5941)
<br>
![e94faff6e55e38970c62529e25aa9e0](https://github.com/user-attachments/assets/088e2ab7-ff4e-49b6-accd-3d34b93e9f98)

文件位置<br>
/root/autodl-fs/entropy-100-001/01<br>
test的时候<br>
python experiments/robot/libero/run_libero_eval.py --model_family openvla --pretrained_checkpoint /root/autodl-fs/entropy-100-001/01 --task_suite_name libero_spatial --center_crop True

<br>修改参数<br>   beta=0.798
        eta=0.002<br>
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/selfplay.py --data_root_dir "/root/autodl-fs/modified_libero_rlds" --dataset_name libero_spatial_no_noops --run_root_dir "/root/autodl-fs/spatial-self-fintune-entropy3" --adapter_tmp_dir "/root/autodl-fs/spatial-self-fintune-weight-entropy3" --lora_rank 32 --batch_size 20 --grad_accumulation_steps 1 --learning_rate 5e-4 --image_aug False --save_steps 500


<br>位置在/root/autodl-fs/spatial-self-fintune-entropy2/try2<br>

python experiments/robot/libero/run_libero_eval.py --model_family openvla --pretrained_checkpoint /root/autodl-fs/spatial-self-fintune-entropy2/try2 --task_suite_name libero_spatial --center_crop True

![image](https://github.com/user-attachments/assets/9780ef29-5d15-4d75-a4f8-45dd6786b548)


再尝试一下0.5，0.1<br>
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/selfplay.py --data_root_dir "/root/autodl-fs/modified_libero_rlds" --dataset_name libero_spatial_no_noops --run_root_dir "/root/autodl-fs/spatial-self-fintune-entropy3" --adapter_tmp_dir "/root/autodl-fs/spatial-self-fintune-weight-entropy3" --lora_rank 32 --batch_size 20 --grad_accumulation_steps 1 --learning_rate 5e-4 --image_aug False --save_steps 500

<br>max 65 min-28<br>

![image](https://github.com/user-attachments/assets/fdc9e181-ad26-4313-aa99-5c86a1438231)

在这里/root/autodl-fs/entropy-0.5-0.1/it-1 是spatial

python experiments/robot/libero/run_libero_eval.py --model_family openvla --pretrained_checkpoint /root/autodl-fs/entropy-0.5-0.1/it-1 --task_suite_name libero_spatial --center_crop True<br>

87.6%

<br>spatiial it -2<br>
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/selfplay.py --data_root_dir "/root/autodl-fs/modified_libero_rlds" --dataset_name libero_spatial_no_noops --run_root_dir "/root/autodl-fs/spatial-self-fintune-it2" --adapter_tmp_dir "/root/autodl-fs/spatial-self-fintune-weight-it2" --lora_rank 32 --batch_size 20 --grad_accumulation_steps 1 --learning_rate 5e-4 --image_aug False --save_steps 500
