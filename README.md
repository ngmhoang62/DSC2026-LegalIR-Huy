# burst_userft_maxrecall — gói tái tạo đầy đủ

Tái tạo cấu hình **`burst_userft_maxrecall`** cho bộ DSC2026 LegalIR: cả phép
đo **CV (Recall@5 = 0.9561)** lẫn **file submission** (md5 `2fb9a8a3b7`).

Gói có **ba đường chạy**. Hai đường đầu không cần GPU, model hay mạng — chạy
trong vài phút. Đường thứ ba tải model rồi tính lại toàn bộ điểm từ đầu.

---

## Cài đặt

```bash
pip install -r requirements.txt
```

Python 3.10+.

---

## Đường 0 — đo lại CV (0.9561)

```bash
python evaluate_cv.py
```

In ra Recall@5 trên 600 query CV, pooled và theo từng block, kèm phán quyết gate:

```
THAM CHIEU 7 kenh (alpha=0)             0.9536  a:0.9750  b:0.9700  c:0.9700  d:0.9356
burst_userft_maxrecall (bản đã ship)    0.9561 (+0.0025)  a:0.9750  b:0.9700  c:0.9850  d:0.9356   PASS
```

- Thời gian: **3–5 phút**, không cần GPU/model/mạng
- `--all` chạy thêm ablation từng kênh (chỉ +aiteamvn_ft, chỉ +jina_ft, bỏ từng kênh…)

Gate dùng xuyên suốt dự án: **vượt tham chiếu pooled VÀ không block nào tụt**.
Hai lần trước, "pooled tăng nhưng một block tụt" đều kéo theo leaderboard giảm.

---

## Đường 1 — dựng lại submission từ cache (khuyến nghị)

```bash
python reproduce.py
```

Dòng cuối phải là:

```
submission.json md5[:10] = 2fb9a8a3b7  (expected 2fb9a8a3b7)
MATCH -- reproduced exactly
```

- Thời gian: **2–3 phút**
- **Không cần GPU, không cần model, không cần mạng**
- File để nộp: `results/burst_userft_maxrecall/submission.zip`

Kiểm tra gói đủ file mà không chạy: `python reproduce.py --check`

Được vậy vì mọi tầng tính điểm đã chạy sẵn và lưu trong `results/`. Lần chạy
này chỉ **hợp nhất điểm** (LTR) rồi **chọn 5 văn bản**.

---

## Đường 2 — tải model và tính lại từ đầu

```bash
python download_models.py        # ~5.5 GB
python run_full_pipeline.py      # ~2 giờ trên GPU 8 GB
```

`run_full_pipeline.py` xoá 3 cache kênh phụ, chấm lại bằng chính model, rồi gọi
`reproduce.py`. Dùng khi bạn đổi model, hoặc muốn chứng minh cache không bị sửa tay.

### Đường 2 KHÔNG cho md5 giống hệt — và đó là bình thường

Đã kiểm chứng ngày 2026-09-09: xoá cả 6 cache rồi chấm lại từ model cho kết quả

| kênh | trùng bit với cache cũ | đổi top-5 |
|---|---|---|
| `jina_ft` (CV + public) | **100%** | 0 |
| `aiteamvn_ft` (CV + public) | **100%** | 0 |
| `title_embed` (CV) | 82.1% (lệch ≤ 6.5e-4) | 6/600 |
| `title_embed` (public) | 76.1% (lệch ≤ 6.7e-4) | 9/1000 |

Kết quả cuối: **CV vẫn đúng 0.9561**, mọi block y hệt, submission khác đúng
**1/1000 query** (`q124570`, chỉ ở slot thứ 5) — md5 thành `f9b56b21…`.

Nguyên nhân: `title_embed` encode 7198 tiêu đề trong một lượt batch lớn, thành
phần batch và padding đổi giữa các lần chạy nên fp16 trên GPU ra khác ở chữ số
thứ tư. Hai model fine-tune chấm theo từng query nên tất định tuyệt đối.

**Nên: md5 `2fb9a8a3b7` chỉ đảm bảo cho đường 1 (đọc cache).** Đường 2 đúng về
mặt khoa học nhưng có thể lệch 1-2 query. Nếu cần bản y hệt để nộp, dùng đường 1.

### Model được tải từ đâu

**Google Drive** — các checkpoint fine-tune, không có trên Hub:

| thư mục | dùng cho |
|---|---|
| `AITeamVN_Vietnamese_Embedding/` | kênh `aiteamvn_ft` |
| `jina_finetuned/` | kênh `jina_ft` |
| `vietlegal_finetuned_results_HNSW/` | LoRA harrier (không dùng trong cấu hình này) |

Link: `https://drive.google.com/drive/folders/1ahUyUuHcegozzSoWOBj3E8OuGvgtcLUV`

**Hugging Face** — model nền công khai:

| repo | thư mục | dùng cho |
|---|---|---|
| `AITeamVN/Vietnamese_Embedding` | `models/AITeamVN_Vietnamese_Embedding` | kênh `dense`, `title_embed` |
| `jinaai/jina-reranker-v2-base-multilingual` | `models/jina-reranker-v2-base-multilingual` | kênh `jina`; **và là code nền để nạp weight `jina_ft`** |
| `AITeamVN/Vietnamese_Reranker` | `models/AITeamVN_Vietnamese_Reranker` | kênh `crossenc` |
| `darklethelong/vnlegal-lal` | `models/vnlegal-lal` | kênh `vnlegal_lal` |

Các lệnh khác:

```bash
python download_models.py --check    # xem đã có gì
python download_models.py --drive    # chỉ tải checkpoint fine-tune
python download_models.py --hf       # chỉ tải model nền
```

Nếu gdown báo lỗi quyền, mở link Drive bằng trình duyệt và tải tay vào
`models/from_drive/`.

---

## Cấu hình được tái tạo

LTR `LogisticRegression(C=0.15, class_weight="balanced")` huấn luyện trên 600
query CV, áp lên 1000 query public, **alpha=0** nên mọi query trả đủ 5 văn bản
(5000/5000 slot).

| nhóm | kênh |
|---|---|
| rank view | base, expanded, jina, dense, corpus |
| score | jina, dense, expansion, e5, corpus |
| kênh phụ | vnlegal_lal, crossenc, **aiteamvn_ft**, **jina_ft**, **title_embed** |
| feature khác | doctype, citation |

Ba kênh in đậm là phần thêm so với bản production:

- **`aiteamvn_ft`** — AITeamVN Vietnamese Embedding fine-tune (bi-encoder,
  XLMRoberta 1024×24). Recall@5 đứng một mình: **0.8906**
- **`jina_ft`** — Jina reranker v2 fine-tune (cross-encoder, XLMRoberta 768×12).
  Recall@5 đứng một mình: **0.8978** — kênh đơn mạnh nhất đo được
- **`title_embed`** — tương đồng embedding giữa câu hỏi và **tiêu đề** văn bản

---

## Nội dung gói

```
evaluate_cv.py               đường 0: đo Recall@5 trên 600 query CV
reproduce.py                 đường 1: kiểm tra → dựng → verify md5
download_models.py           tải model (Drive + Hugging Face)
run_full_pipeline.py         đường 2: tính lại điểm từ model rồi dựng
requirements.txt

run_vnlegal_extra_channel_submission.py    runner chính
score_cv_custom_encoder.py   chấm CV bằng bi-encoder bất kỳ
score_cv_jina_ft.py          chấm CV bằng Jina fine-tune
score_public_models.py       chấm public (bi-encoder hoặc Jina)
tune_title_embedding.py      chấm CV cho title_embed
score_public_title_embed.py  chấm public cho title_embed
build_corpus_dense_index.py  dựng lại corpus chunk index (không bắt buộc)
*.py                         30 module phụ thuộc

DSC2026-LegalIR-main/v4_run/public_test_dataset/
    selected-contexts/       8532 văn bản corpus
    public-official.json     1000 query public
    train.json               7000 query train (huấn luyện LTR)

results/                     31 file cache điểm số theo kênh
```

Danh sách file bắt buộc nằm trong `REQUIRED` của `reproduce.py`. Nó **không
phải liệt kê thủ công** — thu được bằng cách chạy thật với `sys.addaudithook`
để bắt mọi lần mở file, kể cả bên trong numpy và pickle.

---

## Hai chi tiết dễ vấp

**1. `reproduce.py` vô hiệu hoá `ensure_vnlegal_model()`.** Hàm gốc tải 1.2 GB
từ HuggingFace *rồi mới* phát hiện điểm đã có trong cache. Không chặn thì máy
không có mạng sẽ fail dù chẳng cần model đó.

**2. Không nạp trực tiếp được thư mục `jina_finetuned`.** Nó kèm bản sao code
Jina có `import create_position_ids_from_input_ids` — hàm này đã bị bỏ ở
transformers v5. Cách xử lý trong `score_cv_jina_ft.py` và
`score_public_models.py`: khởi tạo model từ `jina-reranker-v2-base-multilingual`
rồi **ghi đè 153 tensor** từ file weight. Kiến trúc trùng khít
(XLMRobertaForSequenceClassification 768×12, vocab 250002) nên an toàn; script
có `assert` chặn nếu xuất hiện key lạ.

---

## Lưu ý về con số 0.9561

Đó là **Recall@5 trên 600 query CV**, không phải điểm leaderboard. Trên bộ dữ
liệu này, chênh lệch CV cỡ ±0.005 nhiều lần **không** chuyển thành thay đổi trên
leaderboard — độ nhiễu của CV 600 query (std ≈ 0.008) lớn hơn khoảng cách cần
đo. Coi CV là bộ lọc thô, leaderboard mới là trọng tài.
