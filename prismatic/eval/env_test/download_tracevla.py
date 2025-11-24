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

#!/usr/bin/env python
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

def main():
    model_path = "furonghuang-lab/tracevla_phi3v"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TraceVLA] Using device: {device}")

    # 1. Processor / tokenizer（已经没问题了）
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        num_crops=1,
    )

    # 2. dtype 选择
    if device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # 3. 用 AutoModelForCausalLM + trust_remote_code 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,                      # 新版用 dtype，而不是 torch_dtype
        trust_remote_code=True,           # 关键：用仓库里的 modeling_phi3_v.py
        attn_implementation="flash_attention_2",
    ).to(device)

    model.eval()
    print(f"[TraceVLA] Model & processor loaded OK. dtype={dtype}, device={device}")

if __name__ == "__main__":
    main()
