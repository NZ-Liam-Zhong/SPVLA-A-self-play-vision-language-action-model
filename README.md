# SPVLA: A Self-Play Vision-Language-Action Model
Author: Ningze Zhong, Bo Wu<br>
Institution: MIT-IBM Watsons AI Lab, Sun Yat-sen University

This repo is adopted from [Openvla: An Open-Source Vision-Language-Action Model](https://github.com/openvla/openvla)

**Run the Self-play code(example)** <br>
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/selfplay.py  --data_root_dir "/root/autodl-tmp/modified_libero_rlds" --dataset_name libero_object_no_noops --run_root_dir "/root/znz/openvla/object-self-fintune" --adapter_tmp_dir "/root/znz/openvla/object-self-fintune-weight" --lora_rank 32 --batch_size 20 --grad_accumulation_steps 1 --learning_rate 5e-4 --image_aug False --save_steps 1000
<br><br>
object-it1-ckpt:<br>
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint /root/znz/openvla/object-self-fintune/object-selfplay-post-training-it-0-batch20-steps1000 \
  --task_suite_name libero_object \
  --center_crop True

  ## 着重分析
  代码在/root/znz/openvla/experiments/robot/openvla_utils.py
  <br>
  def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading in BF16 with Flash-Attention Enabled")

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Move model to device.
    # Note: `.to()` is not supported for 8-bit or 4-bit bitsandbytes models, but the model will
    #       already be set to the right devices and casted to the correct dtype upon loading.
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    # Load dataset stats used during finetuning (for action un-normalization).
    dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    return vla

