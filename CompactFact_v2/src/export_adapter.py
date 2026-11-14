"""Сохранить только обучаемую часть дообученного GLiNER (LoRA + головы) в safetensors."""
import sys, json, os
import torch
from safetensors.torch import load_file, save_file
src, dst = sys.argv[1], sys.argv[2]
p = os.path.join(src, "model.safetensors")
sd = load_file(p) if os.path.exists(p) else torch.load(os.path.join(src, "pytorch_model.bin"), map_location="cpu", weights_only=True)
keep = {k: v.contiguous() for k, v in sd.items() if "lora_" in k or "token_rep_layer.bert_layer" not in k}
save_file(keep, dst)
print(len(keep), "tensors,", round(sum(v.numel() for v in keep.values()) / 1e6, 2), "M params,", round(os.path.getsize(dst) / 1e6, 1), "MB")
