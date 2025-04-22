"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

# import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


#下面导入几个selfplay需要用到的库
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch.nn as nn
import torch.nn.functional as F

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on


@dataclass
class FinetuneConfig:
    # fmt: off

    #object的finetune
    vla_path: str = "/root/autodl-fs/openvla-7b-finetuned-libero-object" 
    vla_path_base: str = "/root/autodl-fs/openvla-7b-prismatic"                           # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
    dataset_name: str = "droid_wipe"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16                                            # Fine-tuning batch size
    #max_steps: int = 200_000
    max_steps: int = 50_000                                        # Max number of fine-tuning steps
    save_steps: int = 1000  
    learning_rate: float = 2e-5                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    # wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    # wandb_entity: str = "stanford-voltron"                          # Name of entity to log under
    # run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    # fmt: on


##############################################################################################
## spin_loss(adapted from)代码来源于https://github.com/uclaml/SPIN/blob/main/spin/alignment/trainer.py  ######
##############################################################################################
def spin_loss(
        policy_real_logps: torch.FloatTensor,
        policy_generated_logps: torch.FloatTensor,
        opponent_real_logps: torch.FloatTensor,
        opponent_generated_logps: torch.FloatTensor,
        reference_free: bool = False,
        loss_type: str = "sigmoid"
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the SPIN loss for a batch of policy and reference model log probabilities.

        Args:
            policy_real_logps: Log probabilities of the policy model for the real responses. Shape: (batch_size,)
            policy_generated_logps: Log probabilities of the policy model for the generated responses. Shape: (batch_size,)
            opponent_real_logps: Log probabilities of the reference model for the real responses. Shape: (batch_size,)
            opponent_generated_logps: Log probabilities of the reference model for the generated responses. Shape: (batch_size,)
            beta: Temperature parameter for the SPIN loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, real_rewards, generated_rewards).
            The losses tensor contains the SPIN loss for each example in the batch.
            The real_rewards and generated_rewards tensors contain the rewards for the real and generated responses, respectively.
        """
        pi_logratios = policy_real_logps - policy_generated_logps
        ref_logratios = opponent_real_logps - opponent_generated_logps
        entropy_regular=opponent_real_logps - policy_generated_logps

        beta=0.5
        eta=0.01

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios

        if loss_type == "sigmoid":
            #print("we are using sigmoid")
            losses = -F.logsigmoid(beta * logits-eta*entropy_regular)
        elif loss_type == "hinge":
            #print("we are using hinge")
            losses = torch.relu(1 - beta * logits-eta*entropy_regular)
        else:
            raise ValueError(f"Unknown loss type: . Should be one of ['sigmoid', 'hinge']")

        real_rewards = beta * (policy_real_logps - opponent_real_logps).detach()
        generated_rewards = beta * (policy_generated_logps - opponent_generated_logps).detach()

        # print(f"losses: {losses}")
        # print(f"policy_real_logps: {policy_real_logps}")
        # print(f"policy_generated_logps: {policy_generated_logps}")
        # print(f"opponent_real_logps: {opponent_real_logps}")
        # print(f"opponent_generated_logps: {opponent_generated_logps}")
        # print(f"logits: {logits}")
        # print(f"real_rewards: {real_rewards}")
        # print(f"generated_rewards: {generated_rewards}")

        losses = losses.mean()
        
        return losses, real_rewards, generated_rewards

def _get_batch_logps(
        mask:torch.Tensor,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        # if not self.is_encoder_decoder:
        #     labels = labels[:, 1:].clone()
        #     logits = logits[:, :-1, :]
        #loss_mask = labels != self.label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        #labels[labels == self.label_pad_token_id] = 0

        # print("logits shape",logits.shape)
        # print("labels.unsqueeze(2) shape",labels.unsqueeze(2).shape)
        # print("Max label index:", labels.max())
        # print("Min label index:", labels.min())
        # if average_log_prob:
        #     return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        # else:
        #     return (per_token_logps * loss_mask).sum(-1)
        #device = torch.device('cuda:0')  # 选择 cuda:0 设备

        #device = logits.device
        #loss_mask=torch.ones(1,37).to(device)
        
        loss_mask = mask.to(labels.device)
        #print("loss_mask",loss_mask)

        # print("begin",begin)
        # print("mask",loss_mask)
        labels = labels.clone()  # Use .clone() to avoid inplace modification
        #print("labels 1",labels)
        labels = labels*loss_mask
        #print("labels 2",labels)
        # print("logits shape",logits.shape)
        # print("labels.unsqueeze(2) shape",labels.unsqueeze(2).shape)
        # print("Max label index:", labels.max())
        # print("Min label index:", labels.min())


        #下面是原来的代码
        #per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        # 对 logits 的每个值除以 10
        #scaled_logits = logits / 10
        # 计算 log softmax
        #print("logits",logits)

        #这里改了，原来是用这样合适一点
        log_softmax_logits = logits.log_softmax(-1)
        #log_softmax_logits = logits.softmax(dim=-1)


        #print("log_softmax_logits",log_softmax_logits)
        # 按照 labels 的索引提取每个 token 的 log probability
        per_token_logps = torch.gather(log_softmax_logits, dim=2, index=labels.unsqueeze(2)).squeeze(2)
        #print("per_token_logps",per_token_logps)



        # print("这一托per_token_logps",per_token_logps)
        # print("loss mask",loss_mask)

        # print("这一托per_token_logps",per_token_logps)
        # print("loss mask",loss_mask)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)
        
##############################################################################################
## spin_loss(adapted from)代码来源于https://github.com/uclaml/SPIN/blob/main/spin/alignment/trainer.py  ######
##############################################################################################



@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Running the first self-play training on OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")
    print("Self-play for Test Time Adaptation!")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    epoch_losses = [] 

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"+ "self-play exp 1"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    # if cfg.run_id_note is not None:
    #     exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path_base, trust_remote_code=False)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path_base,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)


    ####################################################################################
    ##################下面复制一个新的vla，用处就是作为reference model#######################
    ####################################################################################

    # 复制 vla 创建 vla2，并将参数设置为不可训练
    vla2 = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )


    # 将 vla2 的所有参数冻结，确保它不参与训练
    for param in vla2.parameters():
        param.requires_grad = False

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla2 = prepare_model_for_kbit_training(vla2)
    else:
        vla2 = vla2.to(device_id)

    #####################################################################################
    ###################################复制完成###########################################
    #####################################################################################


    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # vla_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # ---
    # print("action_tokenizer",action_tokenizer)
    batch_transform= RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    # print("batch_transform",batch_transform)
    # print("img",img)
    # print("depth",depth)
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )

    #查看数据集,temp
    print("vla_dataset",vla_dataset)

    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    

    # # Initialize Logging =>> W&B
    # if distributed_state.is_main_process:
    #     wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    #查看数据,temp
    #print("dataloader",dataloader)

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            #temp
            #print("batch",batch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                #print("input_ids",batch["input_ids"].shape)
                #print("attention mask",batch["attention_mask"].shape)
                #print("pixel_values",batch["pixel_values"].shape)
                #print("labels",batch["labels"].shape)
                #print("input_ids",batch["input_ids"])
                #print("attention mask",batch["attention_mask"])
                #print("pixel_values",batch["pixel_values"])
                #print("labels",batch["labels"])
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                )
                loss = output.loss #7-13

                ##########################################
                ############ref_model输出##################
                ##########################################
                ref_output: CausalLMOutputWithPast = vla2(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                )
                ref_loss = ref_output.loss

                ##########################################
                #################这里输出完毕###############
                ##########################################

            #print("output shape",output.shape)

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps
            print("normalized_loss",normalized_loss)

            # Backward pass
            #原来的loss
            #normalized_loss.backward()

            #print("output_logits",output.logits.shape)

            # Compute Accuracy and L1 Loss for Logging
            action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]

            

            #ref_model输出
            ref_action_logits = ref_output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]

            ref_action_preds = ref_action_logits.argmax(dim=2)

            #输出完毕

            

            #这个输出非常重要
            #print("action_logits",action_logits)
            print("action logit max",torch.max(action_logits))
            print("action logit min",torch.min(action_logits))

            action_preds = action_logits.argmax(dim=2)

            # print(" ref_action_preds",ref_action_preds)
            # print("action_preds",action_preds)

            #这个输出非常重要
            #print("action_preds",action_preds)


            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            # print("action_gt",action_gt)
            # print("action_preds",action_preds)
            # print("action_logits",action_logits)


            


            #这里开始计算spin，首先action_logits是预测的logits，action_preds是最大的输出，action_gt是真实的标签，
            #关键的求概率部分
            #print("action_logits",action_logits.shape)
            #print("action_preds",action_preds.shape)
            # action_logits torch.Size([16, 37, 32064])
            # action_preds torch.Size([16, 37])
            # mask = action_gt > action_tokenizer.action_token_begin_idx
            # print("action_gt",action_gt)
            # print("tokenizer",action_tokenizer.action_token_begin_idx)
            # print("mask",mask)

            mask = action_gt > action_tokenizer.action_token_begin_idx

            policy_generated_logps = _get_batch_logps(
            mask,
            action_logits,#(-15,60)
            action_preds,
            average_log_prob=True,
        )
            print("policy_generated_logps",policy_generated_logps)

            policy_real_logps = _get_batch_logps(
            mask,
            action_logits,
            action_gt,
            average_log_prob=True,
        )
            print("policy_real_logps",policy_real_logps)
            opponent_generated_logps = _get_batch_logps(
            mask,
            ref_action_logits,#(-30,80)
            ref_action_preds,
            average_log_prob=True,
        )
            print("opponent_generated_logps",opponent_generated_logps)
            opponent_real_logps = _get_batch_logps(
            mask,
            ref_action_logits,
            action_gt,
            average_log_prob=True,
        )
            print("opponent_real_logps",opponent_real_logps)
            losses, real_rewards, generated_rewards = spin_loss(
            policy_real_logps,
            policy_generated_logps,
            opponent_real_logps,
            opponent_generated_logps,
        )
            losses2=losses/ cfg.grad_accumulation_steps
            normalized_loss2=normalized_loss
            losses2=losses2+normalized_loss2
            #losses2=normalized_loss
            losses2.backward()#0.6

            #看看我的losses是什么
            print("losses",losses)
        


            # #这个输出非常重要
            # print("action_gt",action_gt)

            mask = action_gt > action_tokenizer.action_token_begin_idx

            #重要，首先预测mask
            # print("mask",mask)

            #print("action_logits",action_logits)
            #print("action_gt",action_gt)
            #print("action_tokenizer.action_token_begin_idx",action_tokenizer.action_token_begin_idx)

            #print("乱七八糟的东西",vla.module.vision_backbone.featurizer.patch_embed.num_patches)

            #print("action_logits",action_logits.shape)
            #print("action_preds",action_preds.shape)
            #print("action_gt",action_gt.shape)

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # Compute L1 Loss on Predicted (Continuous) Actions
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

            # Store recent train metrics
            recent_losses.append(losses2.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            #   =>> Equal to current step metrics when not using gradient accumulation
            #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

            epoch_losses.append(smoothened_loss)

            # Push Metrics to W&B (every 10 gradient steps)
            # if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
            #     wandb.log(
            #         {
            #             "train_loss": smoothened_loss,
            #             "action_accuracy": smoothened_action_accuracy,
            #             "l1_loss": smoothened_l1_loss,
            #         },
            #         step=gradient_step_idx,
            #     )

            # Optimizer Step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                progress.update()

            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

                    # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                    save_dir = adapter_dir if cfg.use_lora else run_dir

                    # Save Processor & Weights
                    processor.save_pretrained(run_dir)
                    vla.module.save_pretrained(save_dir)

                # Wait for processor and adapter weights to be saved by main process
                dist.barrier()

                # Merge LoRA weights into model backbone for faster inference
                #   =>> Note that merging is slow and can be done post-hoc to speed up training
                if cfg.use_lora:
                    base_vla = AutoModelForVision2Seq.from_pretrained(
                        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=False
                    )
                    merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                    merged_vla = merged_vla.merge_and_unload()
                    if distributed_state.is_main_process:
                        if cfg.save_latest_checkpoint_only:
                            # Overwrite latest checkpoint
                            merged_vla.save_pretrained(run_dir)

                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")
                        else:
                            # Prepare to save checkpoint in new directory
                            checkpoint_dir = Path(str(run_dir) + f"--{gradient_step_idx}_chkpt")
                            os.makedirs(checkpoint_dir, exist_ok=True)

                            # Save dataset statistics to new directory
                            save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)

                            # Save processor and model weights to new directory
                            processor.save_pretrained(checkpoint_dir)
                            merged_vla.save_pretrained(checkpoint_dir)

                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")
                    print("Successfully saved!")
                
                #save loss figure
                # 绘制loss变化曲线
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', label='Training Loss')
                plt.title('Training Loss vs. Epochs')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()

                # 保存为png图片
                plt.savefig('loss_vs_epochs.png')
                plt.show()

                # Block on Main Process Checkpointing
                dist.barrier()


if __name__ == "__main__":
    finetune()
