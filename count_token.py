# count_tokens.py  (Ollama, chạy local)
import os
import ollama
from dotenv import load_dotenv
load_dotenv()

client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
MODEL = os.getenv("MODEL", "qwen2.5:3b")

def count(text):
    # raw=True: không bọc chat template, nên prompt_eval_count chỉ đếm đúng `text`
    r = client.generate(model=MODEL, prompt=text, raw=True, options={"num_predict": 1})
    return r.prompt_eval_count

samples = {
    "en": "How do I request a refund for my order?",
    "vi": "Làm thế nào để yêu cầu hoàn tiền cho đơn hàng?",
    "vi_khong_dau": "Lam the nao de yeu cau hoan tien cho don hang?",
    "code": "def add(a, b): return a + b",
}
for k, v in samples.items():
    print(f"{k:14} {len(v):4} ký tự  {count(v):4} token")
