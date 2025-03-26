# SPVLA: A Self-Play Vision-Language-Action Model

This repo is adopted from [Openvla: An Open-Source Vision-Language-Action Model](https://github.com/openvla/openvla)<br>

**How to install**<br>
下载这个repo的全部代码<br>
Use the setup commands below to get started:

```bash
# Create and activate conda environment
conda create -n openvla python=3.10 -y
conda activate openvla

# Install PyTorch. Below is a sample command to do this, but you should check the following link
# to find installation instructions that are specific to your compute platform:
# https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y  # UPDATE ME!

# Clone and install the openvla repo
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```

### LIBERO Simulation Benchmark Evaluations

In the [updated OpenVLA paper (v2)](https://arxiv.org/abs/2406.09246), we discuss fine-tuning OpenVLA
on a simulated benchmark, [LIBERO](https://libero-project.github.io/main.html), in Appendix E.
Please see the paper for details, such as how we modify the provided demonstration datasets to
improve the overall performance of all methods.

We copy the results to the section below and then discuss how to reproduce the results for OpenVLA.

#### OpenVLA Fine-Tuning Results

| Method | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-Long | Average |
|--------|----------------|---------------|-------------|-------------|---------|
| Diffusion Policy from scratch | 78.3 ± 1.1% | **92.5 ± 0.7%** | 68.3 ± 1.2% | 50.5 ± 1.3% | 72.4 ± 0.7% |
| Octo fine-tuned | 78.9 ± 1.0% | 85.7 ± 0.9% | **84.6 ± 0.9%** | 51.1 ± 1.3% | 75.1 ± 0.6% |
| OpenVLA| **84.7 ± 0.9%** | 88.4 ± 0.8% | 79.2 ± 1.0% | **53.7 ± 1.3%** | **76.5 ± 0.6%** |
|(ours) | **87.4%** | 87.6% | 78.2% | **54.7** | **暂无** |

Each success rate is the average over 3 random seeds x 500 rollouts each (10 tasks x 50 rollouts per task).

#### LIBERO Setup

Clone and install the [LIBERO repo](https://github.com/Lifelong-Robot-Learning/LIBERO):

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

Additionally, install other required packages:
```bash
cd openvla
pip install -r experiments/robot/libero/libero_requirements.txt
```

 To download the modified versions of the LIBERO datasets that we used in our fine-tuning
experiments, run the command below. This will download the LIBERO-Spatial, LIBERO-Object, LIBERO-Goal,
and LIBERO-10 datasets in RLDS data format (~10 GB total). You can use these to fine-tune OpenVLA or
train other methods. This step is optional since we provide pretrained OpenVLA checkpoints below.
(Also, you can find the script we used to generate the modified datasets in raw HDF5 format
[here](experiments/robot/libero/regenerate_libero_dataset.py) and the code we used to convert these
datasets to the RLDS format [here](https://github.com/moojink/rlds_dataset_builder).)
```bash
git clone git@hf.co:datasets/openvla/modified_libero_rlds
```
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/selfplay.py --data_root_dir "/root/autodl-fs/modified_libero_rlds" --dataset_name libero_object_no_noops（这里选择任务） --run_root_dir "/root/autodl-fs/0.4-0.1-object-self-fintune-it-1"（选择想要保存的ckpt位置） --adapter_tmp_dir "/root/autodl-fs/0.4-0.1-object-self-fintune-weight-it-1"（lora的ckpt的位置） --lora_rank 32 --batch_size 20 --grad_accumulation_steps 1 --learning_rate 5e-4 --image_aug False --save_steps 300


**结束**

