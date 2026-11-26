# cosine.py
import numpy as np
from sentence_transformers import SentenceTransformer

# ---- Bước 1: hàm cosine, đúng công thức ở trên ----
def cosine(a, b):
    dot = np.dot(a, b)                               # nhân từng cặp số rồi cộng lại
    do_dai = np.linalg.norm(a) * np.linalg.norm(b)   # tích hai độ dài vector
    return dot / do_dai

# Thử với số nhỏ trước, phải ra 0.96, 0.0, -1.0 (khớp ví dụ tính tay)
print(cosine([3, 4], [4, 3]))
print(cosine([3, 4], [-4, 3]))
print(cosine([3, 4], [-3, -4]))

# ---- Bước 2: biến câu thành vector ----
model = SentenceTransformer("BAAI/bge-m3")

sents = [
    "Tôi muốn hoàn tiền đơn hàng",        # 0: hoàn tiền
    "Làm sao để trả lại tiền?",           # 1: hoàn tiền, khác từ
    "How do I get a refund?",             # 2: hoàn tiền, tiếng Anh
    "Thời tiết Hà Nội hôm nay thế nào",   # 3: chủ đề khác
    "Lãi suất ngân hàng đang tăng",       # 4: chủ đề khác
    "Con mèo nằm trên ghế",               # 5: chủ đề khác
]
emb = model.encode(sents)
print("shape:", emb.shape)    # (6, 1024): 6 câu, mỗi câu 1024 số
print("Câu 0:", sents[0])    # in câu gốc trước, để biết 5 số bên dưới là của câu nào
print(emb[0][:5])             # xem thử 5 số đầu của câu 0

# ---- Bước 3: so từng cặp câu, điểm cao nhất in trước ----
pairs = []
for i in range(len(sents)):
    for j in range(i + 1, len(sents)):
        pairs.append((cosine(emb[i], emb[j]), sents[i], sents[j]))
pairs.sort(reverse=True)

for score, a, b in pairs:
    print(f"{score:.2f}  {a}  <->  {b}")