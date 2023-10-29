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


def build_prompt(user_query: str, inputs: str = "", sep: str = "\n\n### ") -> str:
    """Promptを作成する."""
    sys_msg = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。"
    p = sys_msg
    roles = ["指示", "応答"]
    msgs = [": \n" + user_query, ": \n"]
    if inputs:
        roles.insert(1, "入力")
        msgs.insert(1, ": \n" + inputs)
    for role, msg in zip(roles, msgs):
        p += sep + role + msg
    return p


# Infer with prompt without any additional input
user_inputs = {"user_query": "与えられたことわざの意味を小学生でも分かるように教えてください。", "inputs": "情けは人のためならず"}
prompt = build_prompt(**user_inputs)

input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

tokens = model.generate(
    input_ids.to(device=model.device),
    max_new_tokens=512,
    temperature=1,
    top_p=0.95,
    do_sample=True,
)

out = tokenizer.decode(
    tokens[0][input_ids.shape[1] :], skip_special_tokens=True
).strip()
print(out)
