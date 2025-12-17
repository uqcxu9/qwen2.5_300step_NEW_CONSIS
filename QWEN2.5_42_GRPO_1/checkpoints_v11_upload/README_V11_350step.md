# V11 训练 - 350 步 Checkpoint

## 训练配置 (V11)

| 参数 | 值 |
|------|------|
| 基础模型 | Qwen2.5-7B-Instruct |
| 微调方法 | LoRA (rank=8, alpha=16) |
| 当前步数 | 350 |
| 总步数 | 700 |
| Checkpoint 保存 | 350, 700 步 |
| 验证 | 700 步 |

## V11 改进

1. **冻结机制**: `_FREEZE_STATS=True` 在 5000 样本后冻结自适应统计量
2. **自适应目标**: 消费和工作目标基于数据驱动的分位数
3. **Layer 2 惩罚**: 只在 `work_target_mix < 0.85` 时启用
4. **一致性修复**: simulate.py 和 prepare_verl_data.py 的宏观指标计算对齐

## 文件说明

- `lora_adapter/` - LoRA 权重 (~78MB)
- `huggingface/` - HuggingFace 配置
- `optim_world_size_1_rank_0.pt` - 优化器状态 (~155MB, Git LFS)
- `extra_state_world_size_1_rank_0.pt` - 额外状态
- `data.pt` - 数据状态
- `fsdp_config.json` - FSDP 配置

## ⚠️ 注意：缺少 model_world_size_1_rank_0.pt

由于文件太大 (15GB)，`model_world_size_1_rank_0.pt` 未上传。

### 恢复方法

在新实例上，需要从基础模型重新生成完整 checkpoint：

```python
# 加载基础模型 + LoRA adapter
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained(
    "/workspace/models/Qwen2.5-7B-Instruct",
    torch_dtype="auto"
)
model = PeftModel.from_pretrained(base_model, "lora_adapter/")

# 然后保存完整的 state_dict
import torch
torch.save(model.state_dict(), "model_world_size_1_rank_0.pt")
```

## 训练分数 (350 步附近)

- **critic/score/mean**: ~0.45-0.55
- **actor/entropy**: ~0.04-0.06

