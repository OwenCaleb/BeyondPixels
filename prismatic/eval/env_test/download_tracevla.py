#!/usr/bin/env python
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

def main():
    model_path = "furonghuang-lab/tracevla_7b"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TraceVLA] Using device: {device}")

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        num_crops=1,
    )

    # 新版 transformers 推荐用 dtype / attn_implementation
    if device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32

    vla = AutoModelForVision2Seq.from_pretrained(
        model_path,
        dtype=dtype,                      # 以前叫 torch_dtype
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # 以前叫 _attn_implementation
        use_cache=True,
    ).to(device)

    vla.eval()
    print("[TraceVLA] Model & processor loaded OK.")

if __name__ == "__main__":
    main()


'''

pip install --upgrade "transformers>=4.46.0"

~/.cache/huggingface/hub
'''