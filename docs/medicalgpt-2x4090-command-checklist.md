# MedicalGPT 双卡 RTX 4090 可执行命令清单（2 天资源版）

> 资源：2 × RTX 4090（共 48 GB 显存），连续可用 2 天  
> 目标：完整跑通数据准备 → 双卡训练 → 推理验证 → 评估归档的完整闭环

---

## 目录

1. [环境与依赖检查](#1-环境与依赖检查)
2. [数据准备（迷你集 + 正式集）](#2-数据准备迷你集--正式集)
3. [双卡训练（torchrun / DeepSpeed）](#3-双卡训练torchrun--deepspeed)
4. [训练中监控](#4-训练中监控)
5. [推理验证（训练前后对比）](#5-推理验证训练前后对比)
6. [评估与结果归档](#6-评估与结果归档)
7. [两天时间规划](#7-两天时间规划)
8. [参数建议区间](#8-参数建议区间)
9. [故障排查命令](#9-故障排查命令)

---

## 1. 环境与依赖检查

```bash
# 检查 GPU 状态
nvidia-smi

# 检查双卡是否均可见（应输出 2）
python -c "import torch; print('GPU count:', torch.cuda.device_count())"

# 检查 PyTorch + CUDA 版本匹配
python -c "import torch; print(torch.__version__, torch.version.cuda)"

# 检查 transformers / peft / deepspeed 版本
pip show transformers peft deepspeed accelerate bitsandbytes | grep -E "^Name|^Version"

# 安装或更新依赖（首次运行）
pip install -r requirements.txt

# 可选：安装 flash-attention 提升吞吐
pip install flash-attn --no-build-isolation

# 验证全部核心库可导入
python -c "import torch, transformers, peft, datasets, accelerate; print('All OK')"
```

---

## 2. 数据准备（迷你集 + 正式集）

### 2.1 下载或准备原始数据

```bash
# 查看项目自带数据目录结构
ls -lh data/

# 若使用 Hugging Face 数据集，示例下载
python -c "
from datasets import load_dataset
ds = load_dataset('shibing624/medical', split='train')
ds.to_json('data/medical_train_full.json', force_ascii=False)
print('总样本数:', len(ds))
"
```

### 2.2 制作迷你数据集（快速验证用）

```bash
# 截取前 500 条做迷你集
head -n 500 data/medical_train_full.json > data/mini_train.json

# 截取前 50 条做迷你验证集
head -n 50 data/medical_train_full.json > data/mini_eval.json

# 检查样本格式（查看前 2 条）
python -c "
import json
with open('data/mini_train.json') as f:
    for i, line in enumerate(f):
        if i >= 2: break
        print(json.loads(line))
"
```

### 2.3 准备正式训练集

```bash
# 统计正式集样本数量
wc -l data/medical_train_full.json

# 按 9:1 划分训练集 / 验证集（示例脚本）
python - <<'EOF'
import json, random
random.seed(42)
with open('data/medical_train_full.json') as f:
    lines = f.readlines()
random.shuffle(lines)
split = int(len(lines) * 0.9)
with open('data/train.json', 'w') as f:
    f.writelines(lines[:split])
with open('data/eval.json', 'w') as f:
    f.writelines(lines[split:])
print(f'Train: {split}, Eval: {len(lines)-split}')
EOF
```

---

## 3. 双卡训练（torchrun / DeepSpeed）

### 3.1 迷你集快速验证（先跑通再扩大）

```bash
# 迷你集验证：LoRA 微调，50 步跑通即可
torchrun --nproc_per_node=2 train.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --train_file_path data/mini_train.json \
    --validation_file_path data/mini_eval.json \
    --output_dir output/mini_run \
    --num_train_epochs 1 \
    --max_steps 50 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --max_source_length 512 \
    --max_target_length 512 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.05 \
    --fp16 True \
    --save_steps 50 \
    --eval_steps 50 \
    --use_lora True \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_target_modules "q_proj,v_proj" \
    --logging_steps 10 \
    --report_to none \
    --ddp_find_unused_parameters False
```

### 3.2 正式训练：torchrun 推荐配置

```bash
# 正式训练：双卡 LoRA SFT
torchrun --nproc_per_node=2 train.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --train_file_path data/train.json \
    --validation_file_path data/eval.json \
    --output_dir output/sft_lora_run1 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_source_length 1024 \
    --max_target_length 512 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --fp16 True \
    --save_strategy steps \
    --save_steps 200 \
    --eval_strategy steps \
    --eval_steps 200 \
    --save_total_limit 3 \
    --use_lora True \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj" \
    --logging_steps 10 \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True
```

### 3.3 可选：DeepSpeed ZeRO-2 配置模板

```bash
# 使用 DeepSpeed ZeRO-2（适合显存紧张时进一步降显存）
torchrun --nproc_per_node=2 train.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --train_file_path data/train.json \
    --validation_file_path data/eval.json \
    --output_dir output/sft_ds_run1 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_source_length 1024 \
    --max_target_length 512 \
    --learning_rate 2e-4 \
    --fp16 True \
    --use_lora True \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj" \
    --deepspeed ds_config_zero2.json \
    --logging_steps 10 \
    --save_steps 200 \
    --gradient_checkpointing True
```

**`ds_config_zero2.json` 模板**

```json
{
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "reduce_scatter": true,
    "overlap_comm": true,
    "contiguous_gradients": true
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "gradient_accumulation_steps": 4,
  "gradient_clipping": 1.0,
  "train_micro_batch_size_per_gpu": 4,
  "wall_clock_breakdown": false
}
```

---

## 4. 训练中监控

### 4.1 GPU 实时监控

```bash
# 每 3 秒刷新一次 GPU 状态（温度、显存、利用率）
watch -n 3 nvidia-smi

# 只输出关键指标（显存占用 + 利用率），后台持续记录
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \
    --format=csv -l 5 > logs/gpu_monitor.csv &

# 查看 GPU 进程占用
nvidia-smi pmon -s u -d 5
```

### 4.2 训练日志跟踪

```bash
# 实时跟踪训练输出（nohup 后台运行时）
tail -f logs/train.log

# 过滤关键指标行（loss / eval）
tail -f logs/train.log | grep -E "loss|eval|step"

# 查看最近 100 行日志
tail -n 100 logs/train.log
```

### 4.3 TensorBoard 可视化（可选）

```bash
# 启动 TensorBoard
tensorboard --logdir output/sft_lora_run1 --port 6006 --bind_all

# 检查 TensorBoard 是否在运行
ps aux | grep tensorboard
```

### 4.4 估算剩余时间

```bash
# 查看 checkpoint 生成进度（按时间排序）
ls -lt output/sft_lora_run1/checkpoint-* | head -10

# 查看当前训练步数（从日志提取）
grep -oP "step \K[0-9]+" logs/train.log | tail -1
```

---

## 5. 推理验证（训练前后对比）

### 5.1 准备固定测试问题集

```bash
cat > data/test_prompts.json << 'EOF'
[
  {"instruction": "请问高血压患者的日常饮食需要注意什么？", "input": ""},
  {"instruction": "糖尿病的典型症状有哪些？", "input": ""},
  {"instruction": "阿司匹林的常见副作用是什么？", "input": ""},
  {"instruction": "如何判断是否需要就医？", "input": "患者出现持续发热38.5度超过3天"},
  {"instruction": "青霉素过敏的处理方式是什么？", "input": ""}
]
EOF
```

### 5.2 训练前基线推理

```bash
# 使用原始基座模型推理（保存为基线）
python inference.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --test_file data/test_prompts.json \
    --output_file output/inference_baseline.json \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --top_p 0.9
```

### 5.3 训练后推理对比

```bash
# 使用 LoRA checkpoint 推理
python inference.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --adapter_name_or_path output/sft_lora_run1/checkpoint-best \
    --test_file data/test_prompts.json \
    --output_file output/inference_finetuned.json \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --top_p 0.9

# 对比两份输出
python - <<'EOF'
import json
with open('output/inference_baseline.json') as f:
    base = json.load(f)
with open('output/inference_finetuned.json') as f:
    fted = json.load(f)
for i, (b, ft) in enumerate(zip(base, fted)):
    print(f"\n===== 问题 {i+1} =====")
    print(f"[基线]    {b['output'][:200]}")
    print(f"[微调后]  {ft['output'][:200]}")
EOF
```

### 5.4 合并 LoRA 权重（可选，用于独立部署）

```bash
# 将 LoRA adapter 合并进基座模型，输出完整模型
python merge_peft_adapter.py \
    --base_model_name_or_path meta-llama/Llama-2-7b-hf \
    --peft_model_path output/sft_lora_run1/checkpoint-best \
    --output_dir output/merged_model

# 验证合并后模型可正常加载
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('output/merged_model')
tok = AutoTokenizer.from_pretrained('output/merged_model')
print('合并模型加载成功，参数量:', sum(p.numel() for p in model.parameters()))
"
```

---

## 6. 评估与结果归档

### 6.1 运行项目自带评估（若有）

```bash
# 运行评估脚本（根据项目实际脚本名调整）
python evaluate.py \
    --model_name_or_path output/sft_lora_run1/checkpoint-best \
    --eval_file data/eval.json \
    --output_dir output/eval_results \
    --batch_size 8
```

### 6.2 归档实验结果

```bash
# 创建实验归档目录（固定时间戳变量，避免多次调用 date 不一致）
EXP_TS=$(date +%Y%m%d_%H%M)
EXP_DIR="archive/exp_${EXP_TS}"
mkdir -p "${EXP_DIR}"

# 归档：配置、日志、推理对比、评估结果
cp -r output/sft_lora_run1/checkpoint-best "${EXP_DIR}/"
cp logs/train.log "${EXP_DIR}/"
cp output/inference_baseline.json output/inference_finetuned.json "${EXP_DIR}/"
cp output/eval_results/*.json "${EXP_DIR}/" 2>/dev/null || true

# 记录关键参数（追加到归档目录）
cat > "${EXP_DIR}/config_summary.txt" << EOF
实验时间：$(date)
基座模型：meta-llama/Llama-2-7b-hf
训练方式：LoRA（rank=16, alpha=32）
数据规模：（填写）
关键参数：bs=4, grad_acc=4, lr=2e-4, epoch=3, max_len=1024
最终 loss：（填写）
评估指标：（填写）
EOF

echo "归档完成，路径：${EXP_DIR}"
```

---

## 7. 两天时间规划

### Day 1（重点：跑通 + 迷你验证）

| 时段        | 任务                                             | 预计时长 |
|-------------|--------------------------------------------------|----------|
| 上午 1h     | 环境检查、依赖安装、依赖验证                     | 1 h      |
| 上午 1h     | 数据下载、格式确认、制作迷你集                   | 1 h      |
| 上午 1h     | 迷你集跑通训练（50 步）、确认 loss 下降          | 1 h      |
| 下午 1.5h   | 用迷你 checkpoint 跑推理、保存基线对比输出       | 1.5 h    |
| 下午 1.5h   | 启动正式训练（全量数据 × 3 epoch），后台运行     | 1.5 h    |
| 傍晚 1h     | 监控 GPU、跟踪日志、排查早期问题                 | 1 h      |
| 晚上        | 正式训练持续运行，记录日志                       | 过夜     |

### Day 2（重点：推理对比 + 评估归档）

| 时段        | 任务                                             | 预计时长 |
|-------------|--------------------------------------------------|----------|
| 上午 1h     | 检查训练状态、查看 loss 曲线、确认 checkpoint    | 1 h      |
| 上午 1.5h   | 训练后推理对比（与基线 A/B 对比）                | 1.5 h    |
| 上午 1.5h   | 运行评估脚本（若有）、记录指标                   | 1.5 h    |
| 下午 2h     | 归档实验结果、整理笔记（train_notes.md 等）      | 2 h      |
| 下午 1h     | 做一次可控改动（调参或换数据集），启动对照训练   | 1 h      |
| 傍晚 1h     | 监控对照训练、记录结果                           | 1 h      |
| 晚上 1h     | 撰写 final_summary.md，规划后续实验              | 1 h      |

---

## 8. 参数建议区间

> 基于 2 × RTX 4090（各 24 GB 显存）+ LoRA 微调，以下参数可直接使用。

| 参数                          | 7B 模型建议值         | 13B 模型建议值        | 说明                               |
|-------------------------------|-----------------------|-----------------------|------------------------------------|
| `per_device_train_batch_size` | 4 ～ 8                | 2 ～ 4                | 显存不足时优先降此值               |
| `gradient_accumulation_steps` | 4 ～ 8                | 8 ～ 16               | 等效 bs = bs × acc × nproc         |
| `max_source_length`           | 512 ～ 1024           | 512 ～ 1024           | 超过 1024 显存压力显著上升         |
| `max_target_length`           | 256 ～ 512            | 256 ～ 512            | 医学回答通常 200 ～ 400 字         |
| `learning_rate`               | 1e-4 ～ 3e-4          | 1e-4 ～ 2e-4          | LoRA 微调常用，全参需降 10×        |
| `num_train_epochs`            | 2 ～ 5                | 2 ～ 3                | 数据量小时 epoch 可适当增大        |
| `save_steps` / `eval_steps`   | 100 ～ 500            | 100 ～ 500            | 按步数约为总步数 10% ～ 20% 设置   |
| `lora_rank`                   | 8 ～ 32               | 8 ～ 16               | rank 越大效果越好，显存也越多      |
| `lora_alpha`                  | = rank 或 2 × rank    | = rank 或 2 × rank    | 常用 alpha = 2 × rank              |
| `warmup_ratio`                | 0.03 ～ 0.1           | 0.03 ～ 0.1           | 建议不低于 0.03                    |

---

## 9. 故障排查命令

### 9.1 OOM（显存溢出）

```bash
# 查看当前显存占用详情
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# 清理残留 GPU 进程（慎用，会杀掉所有 GPU 进程）
fuser -k /dev/nvidia*

# 降低参数重新运行（优先顺序）
# 1. 降 per_device_train_batch_size（如从 4 降到 2）
# 2. 降 max_source_length（如从 1024 降到 512）
# 3. 增大 gradient_accumulation_steps（保持等效 bs）
# 4. 开启 gradient_checkpointing（用时间换空间）
# 5. 启用 DeepSpeed ZeRO-2
```

### 9.2 loss 不收敛

```bash
# 检查数据格式（随机抽取 5 条检查 label 是否正确）
python - <<'EOF'
import json, random
with open('data/train.json') as f:
    lines = f.readlines()
samples = random.sample(lines, 5)
for s in samples:
    d = json.loads(s)
    print(d)
    print('---')
EOF

# 检查 label mask：确认 input/instruction 部分被屏蔽，只有 output 参与 loss
# 在数据处理脚本中搜索 IGNORE_INDEX 或 -100

grep -n "IGNORE_INDEX\|-100\|ignore_index" \
    $(find . -name "*.py" | head -20) 2>/dev/null | head -20

# 尝试降低学习率（如从 2e-4 降到 5e-5）
# 检查 loss 初始值是否合理（7B LLaMA 初始 loss 约在 2~3）
grep "loss" logs/train.log | head -5
```

### 9.3 训练吞吐低（tokens/s 偏低）

```bash
# 查看 GPU 利用率（应 > 80%）
nvidia-smi dmon -s u -d 5

# 检查是否为数据加载瓶颈
# 增大 dataloader_num_workers（建议 4）
--dataloader_num_workers 4

# 检查 NVLink 是否正常（双卡间通信）
nvidia-smi nvlink --status

# 查看每步训练时间（从日志提取）
grep "it/s\|s/it" logs/train.log | tail -10

# 确认 fp16/bf16 开启
grep "fp16\|bf16" logs/train.log | head -3
```

### 9.4 多卡分布式问题

```bash
# 检查 NCCL 通信是否正常
NCCL_DEBUG=INFO torchrun --nproc_per_node=2 train.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --max_steps 5 \
    --per_device_train_batch_size 1 \
    --output_dir output/nccl_test 2>&1 | grep -E "NCCL|error|Error"

# 检查端口冲突（默认 29500）
lsof -i :29500

# 如果端口冲突，指定其他端口
torchrun --nproc_per_node=2 --master_port=29600 train.py ...

# 确认 ddp_find_unused_parameters 设置正确
# LoRA 微调中通常需要设置为 False
--ddp_find_unused_parameters False
```

### 9.5 checkpoint 加载失败

```bash
# 列出所有 checkpoint 并按时间排序
ls -lt output/sft_lora_run1/ | grep checkpoint

# 验证 checkpoint 完整性（检查关键文件）
ls output/sft_lora_run1/checkpoint-200/

# 从特定 checkpoint 恢复训练
torchrun --nproc_per_node=2 train.py \
    ... \
    --resume_from_checkpoint output/sft_lora_run1/checkpoint-200
```

---

> 建议：首次遇到报错时，先用 `tail -n 50 logs/train.log` 查看完整错误，再对照本清单排查。  
> 遇到无法解决的问题时，记录完整命令和报错到 `run_log.md`，方便后续追溯。
