# #!/usr/bin/env python
# import torch
# from transformers import AutoProcessor, AutoModelForVision2Seq

# def main():
#     # model_path = "furonghuang-lab/tracevla_7b"
#     
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"[TraceVLA] Using device: {device}")

#     processor = AutoProcessor.from_pretrained(
#         model_path,
#         trust_remote_code=True,
#         num_crops=1,
#     )

#     # 新版 transformers 推荐用 dtype / attn_implementation
#     if device == "cuda" and torch.cuda.is_bf16_supported():
#         dtype = torch.bfloat16
#     elif device == "cuda":
#         dtype = torch.float16
#     else:
#         dtype = torch.float32

#     vla = AutoModelForVision2Seq.from_pretrained(
#         model_path,
#         torch_dtype=dtype,                      # 以前叫 torch_dtype
#         trust_remote_code=True,
#         attn_implementation="flash_attention_2",  # 以前叫 _attn_implementation
#     ).to(device)

#     vla.eval()
#     print(f"[TraceVLA] Model & processor loaded OK. dtype={dtype}")

# if __name__ == "__main__":
#     main()


# '''
# 台式机器 7B GPU Out of Memory

# pip install --upgrade "transformers>=4.46.0"

# ~/.cache/huggingface/hub
# '''


# Load Processor & VLA
from transformers import AutoModelForCausalLM , AutoProcessor
from PIL import Image
import json
import torch
model_path = "furonghuang-lab/tracevla_phi3v"
processor = AutoProcessor.from_pretrained(
    model_path, trust_remote_code=True, num_crops=1
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    _attn_implementation='flash_attention_2',
    use_cache=False
).cuda()
