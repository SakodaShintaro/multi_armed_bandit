"""日本語LLMのテスト."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

size = "7b"

if size == "3b":
    tokenizer = AutoTokenizer.from_pretrained(
        "stabilityai/japanese-stablelm-3b-4e1t-instruct"
    )
    model = AutoModelForCausalLM.from_pretrained(
        "stabilityai/japanese-stablelm-3b-4e1t-instruct",
        trust_remote_code=True,
        torch_dtype="auto",
    )
elif size == "7b":
    tokenizer = AutoTokenizer.from_pretrained(
        "stabilityai/japanese-stablelm-instruct-gamma-7b"
    )
    model = AutoModelForCausalLM.from_pretrained(
        "stabilityai/japanese-stablelm-instruct-gamma-7b",
        torch_dtype="auto",
    )
model.eval()

if torch.cuda.is_available():
    model = model.to("cuda")

while True:
    print("\n\n準備が出来たらEnter: ", end="")
    _ = input()

    f = open("input.txt")
    lines = f.readlines()
    prompt = "\n".join(lines)
    print(prompt)

    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape, device=input_ids.device)

    tokens = model.generate(
        input_ids.to(device=model.device),
        attention_mask=attention_mask.to(device=model.device),
        max_new_tokens=512,
        temperature=1,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    out = tokenizer.decode(
        tokens[0][input_ids.shape[1] :], skip_special_tokens=True
    ).strip()
    print("#### Output ####")
    print(out)
