# ask.py  (Ollama, chạy local)
import os
import ollama
from dotenv import load_dotenv
load_dotenv()

client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
MODEL = os.getenv("MODEL", "qwen2.5:3b")

SYSTEM = """Bạn trả lời câu hỏi CHỈ dựa trên nội dung trong thẻ <context>.
Nếu context không chứa đáp án, trả lời đúng một câu: "Tài liệu không đề cập."
Trích nguyên văn câu trong context làm căn cứ."""


def ask(context, question, temperature=1):
    r = client.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"<context>\n{context}\n</context>\n\nCâu hỏi: {question}"},
        ],
        options={"temperature": temperature, "num_predict": 1000},
    )
    return r.message.content, r

ctx = """Khách được hoàn tiền trong 7 ngày kể từ khi nhận hàng.
Sản phẩm giảm giá trên 50% không được hoàn tiền.
Tiền hoàn về tài khoản sau 3 đến 5 ngày làm việc.
Hoàn tiền trong 14 ngày"""

for q in [
    "Bao lâu thì nhận được tiền hoàn?",
    "Hàng sale 70% có hoàn được không?",
    "Phí ship có được hoàn không?",
    "Làm sao để nhận tiền hoàn?",
    "Trả hàng hoàn tiền như nào?"   # không có trong context
]:
    ans, usage = ask(ctx, q)
    print(q, "\n→", ans, f"\n({usage.prompt_eval_count} in / {usage.eval_count} out)\n")