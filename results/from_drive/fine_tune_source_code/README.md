# Fine-tune model rerank/embedding — DSC2026 Task 1 (Legal IR)

Hướng dẫn **chạy lại từ đầu** trên máy khác. Bạn không cần biết lịch sử dự án,
không cần dựng lại index, không cần chạy lại retrieval — toàn bộ tầng sinh ứng
viên đã nằm trong cache kèm theo gói này.

**Mục tiêu của các script ở đây**: thay **một kênh** trong hệ thống fusion 6 kênh
bằng một model vừa fine-tune, rồi đo xem recall/F2 chính thức có tăng hay không.

Baseline chung mà mọi lần chạy so vào (đã kiểm chứng lại từ cache ngày 2026-09-21,
chạy trên CPU, khớp đúng từng chữ số):

```
recall = 0.9511      f2 = 0.6144
per-block   a=0.9800   b=0.9600   c=0.9800   d=0.9289
```

---

## 0. Đọc 2 phút này trước

Có **hai pipeline khác nhau** trong thư mục này, đừng nhầm:

| | **BURST** (chính) | **HNSW** (phụ, độc lập) |
|---|---|---|
| File | `*_finetune_*.ipynb`, `finetune_*.py` | `vietlegal-tune-kaggle-hnsw.ipynb`, `qwen3-embedding-tune-kaggle-hnsw.ipynb` |
| Ý tưởng | Đọc cache 6 kênh, thay 1 kênh bằng model mới, đo lại fusion | Fine-tune bi-encoder rồi retrieve toàn corpus bằng HNSW |
| Dữ liệu | `results/` + `selected-contexts/` (**kèm trong gói**) | `chunk-context/` + `warmup.json`/`train.json` (**bạn phải tự chuẩn bị**) |
| Chạy ở đâu | Colab **hoặc** Kaggle | Chỉ Kaggle (đường dẫn `/kaggle/input/...` hard-code) |
| Thời gian/epoch | 20 phút – 2 giờ | 2–4 giờ |
| Đo được gì | Recall/F2 **chính thức** của cả hệ thống | Chỉ Recall@5 của riêng bi-encoder đó |

**Nếu bạn chỉ có thời gian cho một thứ: chạy BURST trên Colab** (mục 5). Nó cho
con số so sánh được trực tiếp với bản đã nộp, và không cần dữ liệu gì thêm.

---

## 1. Cần gì để bắt đầu

### Đã có sẵn trong gói — không cần làm gì

| Thứ | Ở đâu | Dung lượng |
|---|---|---|
| Cache retrieval tầng 1 (11 file thực sự được đọc) | `results/` | 81 MB |
| Corpus 8.532 văn bản (`DocumentStore` đọc trực tiếp) | `DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts/` | 519 MB |
| Nhãn train (7.000 truy vấn) | `DSC2026-LegalIR-main/v4_run/public_test_dataset/train.json` | 1,5 MB |
| Công thức chấm điểm của BTC (để đối chiếu) | `DSC2026-LegalIR-main/scoring.py` | — |
| Toàn bộ code | `fine_tune/` | 692 KB |

### Bạn phải tự chuẩn bị

| Thứ | Dùng cho | Cách có |
|---|---|---|
| **GPU** | tất cả | Colab (A100/L4/T4) hoặc Kaggle (2×T4) |
| **Internet trong notebook** | tất cả | `models/` trong gói **chỉ có placeholder** — mọi trọng số tải từ HuggingFace |
| `chunk-context/` (8.532 file) | **chỉ** pipeline HNSW | sinh từ `selected-contexts/` — xem mục 8.1 |
| `warmup.json` (500 câu) | **chỉ** pipeline HNSW | file gốc của BTC |

> **Pipeline BURST không cần `chunk-context/` cũng không cần `warmup.json`.**
> Nếu bạn chỉ chạy BURST, mục 1 coi như xong.

---

## 2. Cây thư mục bắt buộc

Gói này là thư mục `ref/`. Nội dung phải giữ **nguyên cấu trúc con** — code dò
đường bằng đường dẫn tương đối tính từ thư mục cha của `fine_tune/`:

```
ref/
├── fine_tune/                     <- bạn đang đọc README trong này
│   ├── burst_common.py            harness CPU: eval bundle, đặc trưng LTR, LOBO CV, ghi/resume
│   ├── torch_common.py            phần torch: chấm điểm, loss, sampler, quản VRAM
│   ├── finetune_aiteamvn.py       AITeamVN/Vietnamese_Embedding  -> kênh `dense`
│   ├── finetune_jina.py           jina-reranker-v2-base-multilingual -> kênh `jina`
│   ├── finetune_vnlegal_lal.py    darklethelong/vnlegal-lal -> kênh `vnlegal_lal`
│   ├── finetune_qwen3_reranker.py Qwen/Qwen3-Reranker-4B (LoRA) -> thay kênh `jina`
│   ├── finetune_prism_reranker.py infgrad/Prism-Qwen3.5-Reranker-2B (LoRA) -> thay kênh `jina`
│   ├── make_notebooks.py          sinh lại notebook từ template (chỉ dùng ở local)
│   ├── colab_finetune_*.ipynb     4 notebook Colab
│   ├── 02_/03_finetune_*.ipynb    2 notebook Kaggle
│   └── *-hnsw.ipynb               2 notebook pipeline HNSW (Kaggle)
├── results/                       cache retrieval — KHÔNG sinh lại được, đừng xoá
├── DSC2026-LegalIR-main/
│   ├── scoring.py
│   └── v4_run/public_test_dataset/
│       ├── train.json
│       └── selected-contexts/     8.532 file context_*.json
└── models/                        chỉ có placeholder, trọng số tải từ hub
```

### Nếu bạn nhận được file zip

Zip có thể lẫn **~8.600 file `*Zone.Identifier`** (rác Windows, ~34 MB).
Xoá đi cho gọn, không ảnh hưởng gì:

```bash
find ref -name '*Zone.Identifier' -delete
```

Hai thư mục sau **không cần cho bất cứ script nào** ở đây — nếu có thì xoá để
tiết kiệm dung lượng: `ref/old_fine_tune/` (bản cũ, đã bị thay thế),
`ref/DSC/` (1,3 GB gồm merged model của một lần chạy HNSW cũ).

---

## 3. Kiểm tra gói trước khi tốn một giây GPU nào

Bước này **chỉ cần CPU**, không nạp model, không cần torch. Nó xác nhận mọi
file cache đọc được và tái tạo đúng baseline. Nếu bước này sai thì có chạy GPU
cũng vô nghĩa, nên hãy làm trước.

```bash
pip install "numpy>=2" "scikit-learn>=1.6" tqdm
```

```bash
cd ref/fine_tune
python - <<'EOF'
import os, sys
from pathlib import Path
REF = Path("..").resolve()          # đang đứng ở ref/fine_tune, nên ".." là ref/
sys.path.insert(0, str(REF / "fine_tune"))
os.environ["BURST_ROOT"] = str(REF)

import burst_common as bc
documents = bc.DocumentStore(REF / bc.DATA_SUBDIR, preload=True)
bundle    = bc.build_eval_bundle(REF, documents)
baseline, _, _ = bc.lobo_evaluate(bundle)

print(f"\nrecall = {baseline['recall']:.4f}   (phải ra 0.9511)")
print(f"f2     = {baseline['f2']:.4f}   (phải ra 0.6144)")
for n, b in baseline["blocks"].items():
    print(f"  block {n}: recall={b['recall']:.4f}  f2={b['f2']:.4f}")

examples = bc.build_train_pool(REF, negatives=48, limit=None, seed=2026,
                               documents=documents)
print(f"\ntrain pool: {len(examples)} truy vấn  (phải ra 1650)")
EOF
```

Kết quả đúng phải trông như thế này (~2 phút):

```
[load_queries] train.json <- .../public_test_dataset/train.json
Eval bundle: 600 queries, pool min=25 mean=39.2 max=52, candidate ceiling=0.9847
Channel Recall@5 on its own (cached scores):
  corpus       R@5=0.7975  pool coverage=1.000
  dense        R@5=0.8603  pool coverage=0.999
  e5           R@5=0.8467  pool coverage=0.518
  expansion    R@5=0.8419  pool coverage=0.803
  jina         R@5=0.8617  pool coverage=0.999
  vnlegal_lal  R@5=0.1011  pool coverage=0.999

recall = 0.9511   (phải ra 0.9511)
f2     = 0.6144   (phải ra 0.6144)
  block a: recall=0.9800  f2=0.6098
  block b: recall=0.9600  f2=0.6201
  block c: recall=0.9800  f2=0.6055
  block d: recall=0.9289  f2=0.6158
Train pool: 1650 queries (0 skipped), 48 hard negatives each,
            gold-in-top20 of the lexical pool = 0.937

train pool: 1650 truy vấn  (phải ra 1650)
```

`pool coverage` thấp ở `e5` (0.518) và `expansion` (0.803) là **bình thường**:
hai kênh đó chỉ có điểm cho một phần pool ứng viên mở rộng, phần thiếu được LTR
xử lý bằng giá trị khuyết. Không phải dấu hiệu cache hỏng.

Nếu `recall` ra khác 0.9511 → cache bị thiếu hoặc lẫn phiên bản, **dừng lại**,
đừng chạy fine-tune.

---

## 4. Chọn đường chạy

| Đường | Nền tảng | File | Khi nào dùng |
|---|---|---|---|
| **A** | Colab | `colab_finetune_*.ipynb` (4 file) | **Khuyến nghị.** Kết quả tự lưu vào Drive, resume dễ |
| **B** | Kaggle | `02_`/`03_finetune_*.ipynb` | Có 2×T4 miễn phí, hoặc muốn chạy headless qua Save & Run All |
| **C** | bất kỳ | `finetune_*.py` qua CLI | Chạy tự động/hàng loạt, hoặc trên server riêng |
| **D** | Kaggle | `*-hnsw.ipynb` (2 file) | Pipeline khác, cần dữ liệu thêm — xem mục 8 |

Đường A, B, C **chạy cùng một logic** (`burst_common` + `torch_common`), chỉ khác
plumbing đường dẫn và batch size. Số đo giữa chúng so sánh được với nhau.

---

## 5. Đường A — Colab (khuyến nghị)

### 5.1 Chuẩn bị Drive

Upload **nguyên thư mục `ref/`**, giữ cấu trúc con, lên đúng đường dẫn:

```
Google Drive / MyDrive / ref /
                          ├── fine_tune/
                          ├── results/
                          ├── DSC2026-LegalIR-main/
                          └── models/
```

Sau khi mount, notebook đọc `/content/drive/MyDrive/ref`. Muốn đặt chỗ khác thì
sửa biến `REF` ở Cell 2 của notebook.

Output ghi ra **thư mục Drive khác**: `/content/drive/MyDrive/fine_tune_work/<tag>/`
— tách khỏi `ref/` để không lẫn output vào input, và **tự persist** nên runtime
bị ngắt cũng không mất gì.

> Upload 519 MB `selected-contexts/` (8.532 file nhỏ) lên Drive bằng web rất chậm.
> Nhanh hơn: nén `selected-contexts` thành **một** file zip, upload, rồi giải nén
> bằng một cell Colab (`!unzip -q -o ... -d ...`). Phần còn lại của `ref/` upload
> trực tiếp cũng được vì ít file.

### 5.2 Chạy notebook nào

Mỗi notebook là một model độc lập. Chạy tuần tự, xong một model mới sang model
sau (đừng chạy song song — cùng đụng vào Drive).

| Notebook | Model (tải từ hub) | Thay kênh | Cách train | Batch × Accum | MAX_LENGTH |
|---|---|---|---|---|---|
| `colab_finetune_aiteamvn.ipynb` | AITeamVN/Vietnamese_Embedding | `dense` | full FT, 8 block trên | 8 × 2 | 512 |
| `colab_finetune_jina.ipynb` | jinaai/jina-reranker-v2-base-multilingual | `jina` | full FT toàn bộ | 8 × 2 | 512 |
| `colab_finetune_prism_reranker.ipynb` | infgrad/Prism-Qwen3.5-Reranker-2B | `jina` | LoRA r=16 | 4 × 4 | 1024 |
| `colab_finetune_qwen3_reranker.ipynb` | Qwen/Qwen3-Reranker-4B | `jina` | LoRA r=16 | 2 × 8 | 1024 |

Cả 4 đều dùng `EPOCHS=3`, `NEGATIVE_DEPTH=48`, `SEED=2026`, effective batch = 16.
LR khác nhau có chủ ý: **1e-5** cho full fine-tune, **2e-4** cho LoRA (update qua
ma trận rank thấp có biên độ nhỏ hơn nhiều nên cần LR cao hơn).

**Ba notebook `jina`-slot (`jina`, `prism`, `qwen3`) là ba ứng viên cạnh tranh
cùng một chỗ.** Output của chúng vào 3 thư mục riêng (`jina/`, `prism_reranker/`,
`qwen3_reranker/`); ai cho `recall` fusion cao hơn trong `history.json` thì
người đó thắng.

### 5.3 Runtime Colab

- Runtime type: **GPU**. A100 tốt nhất; L4/T4 vẫn chạy được nhưng phải hạ batch (mục 11).
- 2 notebook LoRA (prism/qwen3) có sẵn cell `!pip install -U "transformers>=4.51.0" "peft>=0.10.0" accelerate` — **chạy cell đó trước khi import bất cứ thứ gì**, nâng cấp sau khi torch/transformers đã vào runtime thường không có tác dụng cho tới khi restart.
- 2 notebook full-FT (aiteamvn/jina) **không có cell pip** — chúng dựa vào `transformers` Colab cài sẵn. Nếu `colab_finetune_jina.ipynb` lỗi import ở bước nạp model, thêm một cell đầu tiên rồi Restart runtime:
  ```
  !pip install -q "transformers<5"
  ```
  (Jina là model duy nhất nạp qua `trust_remote_code=True`; `torch_common.patch_hub_code_compat()`
  đã vá phần lệch với transformers 5.x nhưng đây vẫn là mắt yếu duy nhất của gói.)

### 5.4 Cách chạy từng cell

Notebook cắt thành ~10 cell, **mỗi epoch một cell**, cố ý như vậy:

| Cell | Làm gì | Tốn bao lâu |
|---|---|---|
| 1 | pip install (chỉ prism/qwen3) | 1–2 phút |
| 2 | mount Drive, cấu hình, kiểm tra đường dẫn | vài giây |
| 3 | **dựng eval bundle từ cache — CPU** | 1–3 phút |
| 4 | dựng train pool + `RunRecorder` | ~1 phút |
| 5 | tải trọng số, gắn LoRA (nếu có), sanity check 1 batch | 3–15 phút |
| 6 | optimizer/scheduler + định nghĩa 2 hàm chạy | vài giây |
| 7/8/9 | **epoch 1 / 2 / 3** — train rồi chấm điểm, ghi ngay xuống Drive | 20 phút – 2 giờ mỗi cell |
| 10 | bảng tổng kết lịch sử | vài giây |

**Cell 3 in ra baseline `0.9511 / 0.6144`.** Đó là chốt kiểm tra: nếu số đó sai,
dừng lại trước khi tốn GPU ở cell 5.

**Bị ngắt giữa chừng?** Chạy lại từ cell 1. Notebook đọc `history.json` cũ, bỏ
qua epoch đã ghi, nạp lại `best_state.pt` — không tốn lại GPU cho phần đã xong.
Muốn làm sạch thì đặt `RESUME = False` ở cell 2.

> Resume khôi phục **trọng số**, không khôi phục trạng thái optimizer, nên epoch
> tiếp theo khởi động lại momentum của Adam. Với 3 epoch thì đây là đánh đổi
> chấp nhận được, thay vì phải lưu thêm 2× kích thước model mỗi lần.

---

## 6. Đường B — Kaggle

### 6.1 Tạo dataset

Tạo **một** dataset Kaggle, đặt tên sao cho nó mount tại `/kaggle/input/fine_tune`
(tên dataset `fine_tune`). Upload **nội dung bên trong `ref/`** — nghĩa là
`results/`, `DSC2026-LegalIR-main/`, `fine_tune/`, `models/` phải nằm **ngay ở
gốc dataset**, chứ không phải lồng thêm một cấp `ref/`.

Kiểm tra nhanh sau khi attach: phải thấy `/kaggle/input/fine_tune/fine_tune/burst_common.py`.
Nếu bạn thấy `/kaggle/input/fine_tune/ref/fine_tune/burst_common.py` thì bọc sai
một cấp — cell 2 sẽ assert fail ngay.

Notebook settings cần bật: **GPU** và **Internet**.

### 6.2 Notebook

Upload từng file `.ipynb` lên Kaggle dưới dạng **notebook** (không nhét vào dataset):

```
02_finetune_aiteamvn.ipynb   ->  03_finetune_jina.ipynb
```

Batch trên Kaggle thấp hơn Colab (T4 15 GB vs A100 40 GB) nhưng **giữ nguyên
effective batch = 16**, nên số đo so sánh được:

| | Kaggle (T4) | Colab (A100) | effective batch |
|---|---|---|---|
| aiteamvn | batch 4 × accum 4 | batch 8 × accum 2 | 16 |
| jina | batch 4 × accum 4 | batch 8 × accum 2 | 16 |

**Chưa có notebook Kaggle cho prism/qwen3** — hai model LoRA đó chỉ có bản Colab.
Trên Kaggle hãy dùng đường CLI (mục 7).

### 6.3 Hai chế độ

1. **Save Version → Save & Run All (Commit)** — chạy headless, đóng trình duyệt được, Kaggle tự lưu `/kaggle/working` thành output. An toàn nhất.
2. **Chạy tương tác** — tải `<tag>_results.zip` về sau **mỗi** epoch qua panel Output. Ở chế độ này `/kaggle/working` mất khi session chết, đừng dồn tới cuối mới tải.

### 6.4 Hai GPU T4

Nếu Kaggle cấp 2×T4, model tự được bọc `nn.DataParallel` (đặt `MAX_GPUS = 0` để
dùng hết GPU; `= 1` để tắt). `scale_for_gpus` nhân `batch_size` lên 2 và chia
`accum` xuống 2 → effective batch vẫn 16, nên hai GPU chỉ **rút ngắn thời gian**,
không làm lệch phép toán huấn luyện.

Checkpoint lưu ra đã được **strip tiền tố `module.`** mà DataParallel thêm vào,
nên `best_state.pt` từ lần chạy 2 GPU nạp được y hệt vào runner gốc.

---

## 7. Đường C — CLI

Không cần copy file ra ngoài: script chỉ **đọc** từ `--root` và chỉ **ghi** vào
`--work`, không thao tác ghi nào chạm vào thư mục input read-only.

```bash
cd ref/fine_tune

# 3 model full fine-tune
python finetune_aiteamvn.py     --epochs 3 --work ./runs
python finetune_jina.py         --epochs 3 --work ./runs
python finetune_vnlegal_lal.py  --epochs 3 --work ./runs

# 2 model LoRA (cần: pip install "peft>=0.10.0" accelerate)
python finetune_qwen3_reranker.py  --epochs 3 --lr 2e-4 --batch-size 2 --accum 8 \
                                   --max-length 1024 --negatives 3 --work ./runs
python finetune_prism_reranker.py  --epochs 3 --lr 2e-4 --batch-size 4 --accum 4 \
                                   --max-length 1024 --negatives 4 --work ./runs
```

Trên Kaggle, `--root` tự nhận `/kaggle/input/fine_tune` nếu thư mục đó tồn tại,
`--work` mặc định `/kaggle/working`:

```python
REF = "/kaggle/input/fine_tune/fine_tune"
!python {REF}/finetune_aiteamvn.py --epochs 3
```

### Tham số hay dùng

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--root` | tự dò | thư mục `ref/`. Ghi đè bằng biến môi trường `BURST_ROOT` |
| `--work` | `/kaggle/working` | nơi ghi checkpoint/log. Biến `BURST_WORK` |
| `--epochs` | 3 | |
| `--lr` | 1e-5 | LoRA cần 2e-4 |
| `--batch-size` / `--accum` | 4 / 4 | effective batch = tích hai số này |
| `--negatives` | 7 | hard negative mỗi truy vấn mỗi bước |
| `--negative-depth` | 48 | lấy negative sâu bao nhiêu trong pool lexical đã cache |
| `--max-length` | 512 | |
| `--eval-batch-size` | 32 | |
| `--max-gpus` | 0 (= hết) | `1` để tắt DataParallel |
| `--train-top-layers` | 0 (= toàn bộ) | chỉ train N block trên cùng + head, đỡ VRAM |
| `--precision` | auto | `auto`/`fp16`/`bf16`/`fp32` |
| `--eval-before-training` | tắt | chấm model **chưa** fine-tune qua cùng đường đánh giá — mốc so sánh sạch |
| `--no-resume` | — | bỏ qua `history.json`/`best_state.pt` cũ |
| `--keep-every-epoch` | tắt | giữ checkpoint mọi epoch thay vì chỉ epoch tốt nhất |
| `--init-checkpoint` | auto | **chỉ jina**: tiếp tục từ checkpoint đã triển khai nếu có |
| `--lora-r` / `--lora-alpha` / `--lora-dropout` | 16 / 32 / 0.05 | chỉ qwen3/prism |

`finetune_prism_reranker.py` dùng chung toàn bộ vòng lặp train với
`finetune_qwen3_reranker.py` (chỉ khác HF repo và system prompt), nên nhận **cùng
một bộ cờ**.

---

## 8. Đường D — Pipeline HNSW (Kaggle, cần dữ liệu thêm)

Hai notebook `vietlegal-tune-kaggle-hnsw.ipynb` và
`qwen3-embedding-tune-kaggle-hnsw.ipynb` là **pipeline hoàn toàn khác**: fine-tune
bi-encoder bằng InfoNCE + LoRA rồi retrieve toàn corpus qua HNSW. Chúng **không
đọc `results/`**, không dính gì tới BURST, và đo một chỉ số khác (Recall@5 của
riêng model đó trên `warmup.json`, không phải recall/F2 chính thức của hệ thống).

Đường dẫn trong 2 notebook này **hard-code cho Kaggle**, chưa có bản Colab.

### 8.1 Dữ liệu cần — và cách tự sinh

Notebook đọc từ dataset mount tại `/kaggle/input/datasets/nhtclone/legal-data`:

```
legal-data/
└── data/
    ├── chunk-context/     8.532 file, mỗi file {"id": <int>, "chunk": {"0": "...", "1": "..."}}
    ├── warmup.json        {qid: {"question": ..., "answer": [...]}}  — 500 câu
    └── train.json         cùng format — 7.000 câu (có thể copy từ gói này)
```

- `train.json` — lấy luôn từ `DSC2026-LegalIR-main/v4_run/public_test_dataset/train.json` trong gói.
- `warmup.json` — file gốc của BTC, **không kèm trong gói**.
- `chunk-context/` — **không kèm trong gói**, phải sinh từ `selected-contexts/`
  (thư mục này thì có sẵn trong gói, ở
  `DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts/`).

  Notebook sinh chunk là `chunking_legal_context.ipynb`, **không nằm trong gói này** —
  xin người gửi. Quy tắc nó áp dụng: cắt theo ngưỡng **512 token** của tokenizer
  `mainguyen9/vietlegal-harrier-0.6b`, bỏ preamble ngắn hơn 30 ký tự, nối lại các
  dòng bị word-wrap `\r\n` do convert PDF/Word.

  Sinh xong hãy kiểm tra **signature** khớp — notebook tính
  `(số file .json, tổng bytes)` và dùng nó để quyết định cache còn dùng được không:

  ```python
  import os
  d = "data/chunk-context"
  fs = sorted(f for f in os.listdir(d) if f.endswith(".json"))
  print((len(fs), sum(os.path.getsize(os.path.join(d, f)) for f in fs)))
  # phải ra (8532, 380681028)
  ```

### 8.2 Dataset state (tuỳ chọn, chỉ để tiết kiệm thời gian)

Notebook còn dò một dataset thứ hai (`nhtclone/vietlegal-state` /
`nhtclone/qwen3-state`) chứa `corpus_emb.npy` + `corpus_emb.meta.json` + kết quả
retrieval baseline, để **không phải encode lại corpus** (tốn 2–3 giờ GPU).

Nếu bạn không có dataset đó:

- `vietlegal-tune-kaggle-hnsw.ipynb` mặc định `SKIP_BASELINE = True` → nó bỏ qua
  hẳn bước baseline, dùng `BASE_RECALL_REF = 0.8292` làm mốc epoch 0. **Chạy được
  ngay, không cần state.**
- `qwen3-embedding-tune-kaggle-hnsw.ipynb` mặc định `SKIP_BASELINE = False` →
  lần chạy đầu sẽ encode corpus từ đầu, **tốn ~2–3 giờ GPU** trước khi vào epoch 1.
  Chấp nhận được (Qwen3-Embedding-4B chưa từng được đo trên bộ này nên cần baseline
  thật), hoặc đặt `SKIP_BASELINE = True` để bỏ qua và chỉ xem xu hướng giữa các epoch.

`_legal_state_meta_ok()` kiểm tra cache theo cả 3 điều kiện `model` + `max_seq_len`
+ `signature`; lệch một cái là nó tự bỏ cache và encode lại. Nói cách khác: cache
sai không làm ra số sai, chỉ làm chậm.

### 8.3 Cấu hình

| | vietlegal | qwen3-embedding |
|---|---|---|
| Model | `mainguyen9/vietlegal-harrier-0.6b` | `Qwen/Qwen3-Embedding-4B` |
| Epochs | 5 | 3 |
| Train batch | 32 | 4 |
| Encode batch | 128 | 32 |
| MAX_SEQ_LEN | 512 | 512 |
| LoRA | r=16, alpha=32, all-linear | giống |
| LR | 2e-4 | 2e-4 |
| HNSW | M=16, ef_construction=200, ef_search=300 | giống |
| Ngưỡng lưu checkpoint | `0.8292 + 0.005` | `0.8292 + 0.005` |

⚠️ `BASE_RECALL_REF = 0.8292` trong notebook **qwen3** là mốc tham chiếu **chéo
model** (lấy từ vnlegal-lal), không phải baseline zero-shot thật của
Qwen3-Embedding-4B. Đừng đọc `saved_checkpoint: false` ở đó là "model tệ".

---

## 9. Đọc kết quả

Mọi đường chạy đều ghi vào `<work>/<tag>/`, với `tag` ∈ `{dense_aiteamvn, jina,
vnlegal_lal, qwen3_reranker, prism_reranker}`:

| File | Nội dung |
|---|---|
| `history.json` | **đọc file này trước.** Mọi epoch: recall/precision/F2 tổng + theo từng block, recall riêng của kênh, train loss, và dòng `baseline` từ cache |
| `best_state.pt` | `{"state_dict": ...}` — nạp bằng `torch.load(...)["state_dict"]` + `load_state_dict(..., strict=False)`. Với qwen3/prism đây là **chỉ adapter LoRA**, vài MB, không phải base model |
| `best_predictions.json` | `{qid: {"answer": [...]}}` sau ngưỡng động — đúng định dạng submission |
| `best_ranking.json` | top-20 xếp hạng hợp nhất mỗi truy vấn |
| `best_channel_scores.pkl` | `{qid: {doc: score}}` — thả thẳng vào pipeline làm score cache của kênh đó |
| `<tag>_results.zip` | gói nhỏ (không kèm `.pt`) để tải về nhanh |

### "Tốt nhất" nghĩa là gì

`best_*` = **epoch tốt nhất của chính lần chạy này** trên holdout 600 truy vấn,
**không** phải "đã vượt baseline cache". Epoch đầu luôn được lưu; epoch sau chỉ
ghi đè khi validate tăng. Tiêu chí: `recall` chính thức (sau ngưỡng động), hoà
thì xét `f2` — đúng công thức trong `DSC2026-LegalIR-main/scoring.py`.

Việc có vượt baseline hay không ghi riêng ở cột **`beats_baseline`** của
`history.json` và không ảnh hưởng tới chuyện lưu. Lý do: baseline kênh `jina`
đến từ một checkpoint **không nằm trong gói này**, nên buộc phải vượt nó đồng
nghĩa với vứt đi trọng số duy nhất mà lần chạy tạo ra.

---

## 10. Ba điều cần biết trước khi tin một con số

**1. Ngưỡng nhiễu của bài này là std 0.008 trên Recall.** Dự án đã có **4 lần**
"thắng trên CV nhưng thua trên leaderboard thật". Chênh lệch nhỏ hơn 0.008 không
phải bằng chứng của gì cả.

**2. Tập train và tập đánh giá GIAO NHAU — có chủ ý.** Toàn bộ ~1.650 truy vấn
có cache retrieval (`train.json` index 0–1149 và 1250–1749) được đưa vào train,
**kể cả 600 truy vấn của block LOBO** dùng để đánh giá. Nghĩa là Recall/F2 in ra
sau mỗi epoch **không còn là ước lượng generalization** — model đã thấy chính các
truy vấn đó lúc train, nên số sẽ lạc quan hơn thực tế.

Đây là đánh đổi có chủ đích cho lần fine-tune cuối để tận dụng hết dữ liệu nhãn,
không phải rò rỉ ngoài ý muốn. Dùng số đó để **so các epoch/model với nhau**, đừng
dùng nó để dự đoán điểm leaderboard. (Việc pipeline generalize tốt đã được kiểm
chứng riêng trước đó: CV 0.9511, held-out 300 truy vấn 0.9483.)

Muốn số đo sạch hơn: đặt `TRAIN_QUERIES` hoặc `--train-queries` để giới hạn, hoặc
sửa `build_train_pool(...)` cho nhận `exclude=bundle.all_ids` trở lại.

> `train.json` có 7.000 truy vấn nhưng chỉ **1.650** có cache retrieval — gói này
> không có FTS/corpus index để tính cache cho phần còn lại. "Toàn bộ tập train"
> ở đây nghĩa là toàn bộ phần **đã có cache**.

**3. Pooling của vnlegal-lal phải giữ CLS.** Last-token pooling từng cho CV cao
hơn (F2 +0.0127) nhưng giảm recall thật trên leaderboard và đã bị hoàn tác.
`finetune_vnlegal_lal.py` huấn luyện chính biểu diễn CLS để giải quyết mâu thuẫn
đó từ phía huấn luyện, thay vì đổi pooling lúc suy luận. **Đừng đổi lại.**

---

## 11. Gặp lỗi thì làm gì

### `AssertionError: Không thấy /content/drive/MyDrive/ref`
Chưa mount Drive xong, hoặc upload sai chỗ. Kiểm tra bằng
`!ls /content/drive/MyDrive/ref` — phải thấy `fine_tune  results  DSC2026-LegalIR-main  models`.

### `RuntimeError: The attached code is out of date -- missing: ...`
`bc.check_code_version()` phát hiện code bạn attach cũ hơn cái notebook gọi. Trên
Kaggle: mở panel Input, **remove dataset rồi add lại** để nó lấy version mới nhất
(Kaggle pin notebook vào version lúc attach), restart kernel, chạy từ cell 1.
Trên Colab: upload lại `fine_tune/` mới.

### `recall` ở cell 3 không ra 0.9511
Cache thiếu hoặc lẫn phiên bản. Chạy lại mục 3. **Đừng** chạy tiếp.

### `retrieval cache missing N eval queries`
Thiếu file trong `results/burst_large_ltr/` — phải có đủ **4** file:
`retrieval_train1000_tune50_val100.pkl`, `fresh_1251_1350_retrieval.pkl`,
`fresh_1351_1450_retrieval.pkl`, `fresh_1451_1750_retrieval.pkl`.

### `ImportError: cannot import name 'create_position_ids_from_input_ids'`
Đây là code trên hub của jina viết cho transformers cũ. `finetune_jina.py` đã gọi
`tc.patch_hub_code_compat()` để vá; nếu vẫn lỗi thì hạ version:
`!pip install -q "transformers<5"` rồi **Restart runtime**.

### `ImportError` từ `peft.tuners.lora.torchao`
Kaggle/Colab đôi khi ship `torchao` cũ hơn mức `peft` yêu cầu. Cả 2 script LoRA
đã vô hiệu hoá phép thăm dò đó (`is_torchao_available = lambda: False`). Nếu vẫn
lỗi: `!pip install -U "peft>=0.10.0"` rồi Restart.

### CUDA OOM
Thứ tự thử, hiệu quả giảm dần (`torch_common.py` cũng in sẵn gợi ý này khi OOM):

1. Giảm `BATCH_SIZE`, tăng `ACCUM` **giữ nguyên tích** — effective batch không đổi nên kết quả vẫn so sánh được. Ví dụ `4×4` → `2×8`.
2. Giảm `NEGATIVES` (7 → 4 → 3).
3. Tăng `TRAIN_TOP_LAYERS` (chỉ train N block trên cùng — đóng băng bớt). Chỉ áp dụng cho 2 model full-FT.
4. `MAX_GPUS = 1` (tắt DataParallel — mỗi GPU giữ một replica đầy đủ nên đôi khi 1 GPU lại vừa hơn).
5. Giảm `MAX_LENGTH` — **đây là lựa chọn cuối**: nó đổi chính đường inference nên số đo không còn so sánh được với baseline cache.

`report_memory_budget` (cell 4) chỉ kiểm phần cố định (weights + grads + AdamW),
**không mô phỏng được activation** — nên nó có thể báo đủ chỗ mà vẫn OOM ở forward
đầu tiên. Theo dõi dòng "GB left for activations" trong log.

### `no directory .../models/xxx; loading <repo> from HuggingFace`
**Đây không phải lỗi.** `models/` trong gói chỉ có placeholder, cố tình như vậy.
Log này nghĩa là script đang tải trọng số thật từ hub — đúng như mong đợi. Chỉ cần
Internet bật. (Thư mục local chỉ được dùng khi thực sự chứa `*.safetensors`/`*.bin`.)

### `finetune_jina.py`: "burst_pairwise_state.pt not found, initialising from HuggingFace"
Cũng không phải lỗi. Checkpoint jina đã triển khai (197 MB) bị loại khỏi gói, nên
`--init-checkpoint auto` lùi về trọng số gốc và in rõ điều đó ra log.

---

## 12. Kiến trúc: cái gì đọc cache, cái gì tính lại

Đây là lý do một epoch chỉ mất 20 phút – 2 giờ thay vì vài ngày.

**Đọc từ cache, không bao giờ tính lại:**

- **Tầng 1** (sinh ứng viên): BM25 4 nhánh, multistage top-20, dense expansion union-50, corpus dense rank cap=32
- **Tầng 2**: hai kênh **không** phải kênh đang fine-tune, giữ nguyên điểm cache

**Tính lại mỗi epoch:**

- **Tầng 2**: chỉ kênh của model đang fine-tune
- **Tầng 3**: LTR fusion 6 kênh, `LogisticRegression(C=0.15)` — CPU, ~2 giây
- **Tầng 4**: ngưỡng động alpha=0.15

Vì tập ứng viên **cố định qua mọi epoch và mọi model**, các con số so sánh được
với nhau và với baseline gốc. Hard negative lấy trực tiếp từ pool lexical đã cache
của từng truy vấn (48 doc đầu, trừ gold) — tức đúng những văn bản mà retrieval
đang triển khai xếp hạng cao nhưng sai.

### Vì sao pipeline BURST không dùng HNSW

Trong vòng lặp fine-tune này **không có bước tìm kiếm vector nào**: tầng 1 đọc
hoàn toàn từ cache, negative lấy từ pool đã cache, mỗi truy vấn chỉ chấm ~39 ứng
viên. Không có kNN để mà tăng tốc.

Chỗ duy nhất có tìm kiếm vector là `corpus_dense` — quét toàn corpus, và ở đúng
quy mô của nó (273.024 vector × 1.000 truy vấn) matmul thẳng mất **6,5 giây trên
CPU**, dưới 1 giây trên T4. Riêng việc dựng index HNSW cho 273k vector đã tốn 1–2
phút. Con số "67 phút" trong tài liệu gốc là thời gian **mã hoá** corpus, không
phải thời gian tìm kiếm.

Thứ thực sự chiếm thời gian mỗi epoch là số lần forward của reranker (~47.000 đoạn
văn × 512 token) — và đó là thứ thêm GPU sẽ chia đôi.

### Quan sát về kênh `vnlegal_lal`

Chấm riêng từng kênh trên pool ứng viên 600 truy vấn:

```
jina         R@5 = 0.8617
dense        R@5 = 0.8603
e5           R@5 = 0.8467
expansion    R@5 = 0.8419
corpus       R@5 = 0.7975
vnlegal_lal  R@5 = 0.1011      <- hoán vị ngẫu nhiên cùng pool cho 0.113
```

Kênh `vnlegal_lal` với trọng số gốc + CLS pooling gần như **không mang tín hiệu
xếp hạng** (dải điểm [0.53, 1.00]). Nó vẫn được giữ vì đóng góp nhỏ qua LTR — và
đây là kênh **còn nhiều dư địa nhất**, đáng fine-tune trước nếu bạn muốn tìm cải
thiện lớn.

Nhưng lưu ý: **không có notebook nào cho `vnlegal_lal`** trong gói (xem mục 13).

### Lưu ý về kênh `dense` (aiteamvn)

AITeamVN còn nuôi **hai tầng sinh ứng viên khác** (dense expansion union-50 và
corpus dense index). Script cố ý giữ nguyên hai tầng đó từ cache: dựng lại corpus
index tốn ~67 phút GPU, và nếu chạy lại expansion thì tập ứng viên cũng đổi theo,
khiến số đo lẫn hai thay đổi vào nhau. Khi một checkpoint AITeamVN đã chứng minh
được mình ở tầng rerank, **đó mới là lúc** dựng lại cả hai index bằng nó và đo lại
toàn tuyến.

Ngược lại, `qwen3_reranker` và `prism_reranker` **chỉ có thể** thay kênh `jina`,
không bao giờ thay được `dense`: chúng là reranker generative yes/no, chỉ chấm
được cặp (câu hỏi, ứng viên) đã có sẵn, không sinh được vector embedding mà hai
tầng kia cần.

---

## 13. Những gì KHÔNG có trong gói

| Thiếu | Hệ quả | Cách xử lý |
|---|---|---|
| Trọng số thật trong `models/` | Mọi model tải từ HuggingFace | Bật Internet trong notebook |
| `results/jina_reranker/burst_pairwise_state.pt` (197 MB) | `finetune_jina.py --init-checkpoint auto` khởi tạo từ trọng số hub thay vì checkpoint đã triển khai | Không cần làm gì; log in rõ |
| `01_finetune_vnlegal_lal.ipynb` | Không có notebook cho kênh `vnlegal_lal` (cả Kaggle lẫn Colab) | Dùng CLI (mục 7), hoặc `python make_notebooks.py` để sinh lại bản Kaggle |
| Notebook Kaggle cho prism/qwen3 | 2 model LoRA chỉ có bản Colab | Dùng CLI trên Kaggle |
| `chunk-context/`, `warmup.json`, notebook chunking | Pipeline HNSW (mục 8) không chạy được ngay | Xem mục 8.1 |
| FTS index / corpus dense index | Không thể sinh cache retrieval cho 5.350 truy vấn còn lại của `train.json` | Không có cách nào từ gói này |
| `train_clean.json`, `public-official.json` | Không script nào trong `fine_tune/` đọc tới | Không cần |

### ⚠️ `make_notebooks.py` không bao trọn 6 notebook

`make_notebooks.py` sinh notebook từ một template chung, nhưng danh sách `MODELS`
của nó **chỉ có 3 model**: `vnlegal_lal`, `aiteamvn`, `jina` → nó sinh 3 notebook
Kaggle + 2 notebook Colab.

**`colab_finetune_prism_reranker.ipynb` và `colab_finetune_qwen3_reranker.ipynb`
được viết tay, KHÔNG sinh từ template.** Nếu bạn sửa template rồi chạy
`python make_notebooks.py`, hai notebook đó **bị bỏ lại và sẽ lệch** so với 4
notebook kia. Sửa tay cả hai, hoặc thêm chúng vào `MODELS` trước.

(Chạy `make_notebooks.py` cũng tạo lại `01_finetune_vnlegal_lal.ipynb` — xoá đi
nếu bạn không định dùng.)

---

## 14. Bản đồ file

### Code — 7 file `.py`, tất cả đều cần

| File | Vai trò | Cần torch? |
|---|---|---|
| `burst_common.py` | Harness: dựng eval bundle 600 truy vấn từ cache, đặc trưng LTR, LOBO CV, ngưỡng động, ghi kết quả + resume | Không |
| `torch_common.py` | Chấm điểm cross-encoder/bi-encoder đúng đường inference gốc, hàm loss, sampler, quản VRAM | Có |
| `finetune_aiteamvn.py` | AITeamVN/Vietnamese_Embedding → kênh `dense` | Có |
| `finetune_jina.py` | jina-reranker-v2 → kênh `jina` | Có |
| `finetune_vnlegal_lal.py` | darklethelong/vnlegal-lal → kênh `vnlegal_lal` | Có |
| `finetune_qwen3_reranker.py` | Qwen3-Reranker-4B (LoRA) → **thay** kênh `jina` | Có |
| `finetune_prism_reranker.py` | Prism-Qwen3.5-Reranker-2B (LoRA) → **thay** kênh `jina`. Import lại toàn bộ vòng lặp từ file trên | Có |

`make_notebooks.py` chỉ dùng ở local để sinh notebook — không cần upload lên
Colab/Kaggle.

### Cache trong `results/` — 11 file thực sự được đọc

```
results/burst_large_ltr/retrieval_train1000_tune50_val100.pkl    30 MB   qids[0:1150]
results/burst_large_ltr/fresh_1251_1350_retrieval.pkl           2,6 MB   qids[1250:1350]
results/burst_large_ltr/fresh_1351_1450_retrieval.pkl           2,7 MB   qids[1350:1450]
results/burst_large_ltr/fresh_1451_1750_retrieval.pkl           8,0 MB   qids[1450:1750]
results/jina_reranker/holdout_scores_finetuned.pkl              305 KB   kênh jina + thứ tự multistage top-20
results/dense_expansion/union50_scores.pkl                      2,2 MB   kênh expansion
results/e5_dense/holdout_scores.pkl                             305 KB   kênh e5
results/corpus_index/holdout_extended_scores_cap32.pkl           1,1 MB   kênh jina + dense trên pool mở rộng
results/corpus_index/holdout_dense_rank_cap32.pkl                1,8 MB   kênh corpus (ranking + scores)
results/embedding_finetune/vnlegal_lal_cv_scores.pkl             592 KB   kênh vnlegal_lal
```

Các thư mục còn lại trong `results/` (`aiteamvn_dense/`, `expanded_rerank/`,
`burst_expanded_fusion/`, `burst_robust_fusion/`, `burst_gpu_threeview/`,
`burst_vnlegal_extra_fusion/`, `vietnamese_reranker/` — ~35 MB) **không được
script nào ở đây đọc tới**; chúng thuộc các script submission ở `ref/`. Xoá được
nếu cần tiết kiệm dung lượng.

`DSC2026-LegalIR-main/scoring.py` cũng **không được import** — `burst_common` tự
cài lại công thức. File giữ lại chỉ để đối chiếu.

---

## Tóm tắt một trang

```
1. Giải nén ref/, xoá *Zone.Identifier
2. pip install numpy scikit-learn tqdm   -> chạy script mục 3
   -> phải ra recall 0.9511 / f2 0.6144, train pool 1650. Sai thì DỪNG.
3. Upload nguyên ref/ lên Google Drive tại MyDrive/ref
4. Mở colab_finetune_aiteamvn.ipynb, Runtime > GPU, chạy từ cell 1
   -> cell 3 in lại 0.9511 (chốt kiểm tra thứ hai)
   -> cell 7/8/9 = epoch 1/2/3, kết quả tự ghi vào MyDrive/fine_tune_work/
5. Đọc history.json: cột `recall` để chọn epoch, cột `beats_baseline` để biết
   có đáng thay kênh không
6. Lặp lại với colab_finetune_jina / prism / qwen3 (3 ứng viên cùng tranh chỗ `jina`)
7. Chênh lệch < 0.008 recall thì đừng kết luận gì.
```

Bí thì đọc mục 11 (lỗi thường gặp) trước khi hỏi.
