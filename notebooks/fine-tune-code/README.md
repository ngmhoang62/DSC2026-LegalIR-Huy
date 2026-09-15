# Fine-tune 3 model rerank trên khung BURST

Ba script fine-tune, mỗi model một file, chạy tuần tự. Toàn bộ tầng 1 (sinh ứng
viên) được **đọc từ cache** trong `results/` — không dựng lại FTS index, không
dựng lại corpus dense index, không chạy lại retrieval. Chỉ kênh của chính model
đang fine-tune được tính lại sau mỗi epoch.

## Files

Có hai cách chạy cùng một thứ: **notebook** (khuyến nghị trên Kaggle, cắt theo
cell nên mất mát khi đứt session là tối thiểu) hoặc **script CLI**.

| File | Vai trò |
|---|---|
| `01_finetune_vnlegal_lal.ipynb` | Notebook — chạy **trước tiên**, kênh còn nhiều dư địa nhất |
| `02_finetune_aiteamvn.ipynb` | Notebook — kênh `dense` |
| `03_finetune_jina.ipynb` | Notebook — kênh `jina`, chậm nhất |
| `burst_common.py` | Harness dùng chung: dựng lại tập đánh giá 600 truy vấn từ cache, đặc trưng LTR, LOBO CV, ngưỡng động, ghi kết quả + resume. Không cần torch. |
| `torch_common.py` | Phần torch dùng chung: chấm điểm cross-encoder / bi-encoder đúng đường inference gốc, hàm loss, sampler. |
| `finetune_jina.py` | Jina cross-encoder → kênh `jina` |
| `finetune_aiteamvn.py` | AITeamVN/Vietnamese_Embedding → kênh `dense` |
| `finetune_vnlegal_lal.py` | darklethelong/vnlegal-lal → kênh `vnlegal_lal` |
| `make_notebooks.py` | Sinh 3 notebook từ một template chung — sửa template rồi chạy lại để 3 notebook không lệch nhau |

## Harness đã được kiểm chứng

`burst_common.build_eval_bundle` + `lobo_evaluate` chạy trên cache tái tạo
**đúng từng con số** của bản đã nộp:

```
official recall  = 0.9511      (README bản gốc: 0.9511)
official F2      = 0.6144      (README bản gốc: 0.6144)
per-block        a=0.9800  b=0.9600  c=0.9800  d=0.9289
```

Và khi đưa chính điểm số cache trở lại qua đường override kênh, kết quả giống
hệt tới từng chữ số — nghĩa là phần "thay một kênh bằng model mới" không làm
lệch gì khác trong pipeline.

## Tái sử dụng những gì

Đọc từ cache, **không bao giờ tính lại**:

- Tầng 1: BM25 4 nhánh (`results/burst_large_ltr/*.pkl`), multistage top-20
  (thứ tự key của `results/jina_reranker/holdout_scores_finetuned.pkl`), dense
  expansion union-50, corpus dense rank cap=32
- Tầng 2: hai kênh còn lại giữ nguyên điểm cache

Tính lại mỗi epoch:

- Tầng 2: **chỉ kênh của model đang fine-tune**
- Tầng 3: LTR fusion 6 kênh, LogisticRegression(C=0.15) — CPU, ~2 giây
- Tầng 4: ngưỡng động alpha=0.15

Vì tập ứng viên cố định qua mọi epoch và mọi model, các con số so sánh được
với nhau và với baseline gốc.

## Dữ liệu train

1.050 truy vấn có nhãn: `train.json` index 0–749 và 850–1149. Đây là những
truy vấn **đã có sẵn cache retrieval** và **không nằm trong** 600 truy vấn dùng
để đánh giá (index 750–849, 1250–1749). Không rò rỉ — đã kiểm tra.

Hard negative lấy trực tiếp từ pool lexical đã cache của từng truy vấn (48 doc
đầu, trừ gold), tức đúng những văn bản mà retrieval đang triển khai xếp hạng
cao nhưng sai.

## Chạy trên Kaggle

Dataset `nhtclone/fine-tune-ref` mount tại `/kaggle/input/fine-tune-ref/ref`.
Script **tự nhận** đường dẫn này: nếu `/kaggle/input/fine-tune-ref/ref` tồn tại
thì lấy làm `--root`, không thì lùi về thư mục cha của `fine_tune/` (nên vẫn
chạy được ở local mà không sửa gì). Ghi đè bằng `--root` hoặc `BURST_ROOT`.

Notebook settings cần bật: **GPU** và **Internet** (trọng số 3 model tải từ
HuggingFace — thư mục `models/` trong gói ref chỉ có placeholder, cố tình như
vậy để runner gốc bỏ qua việc tải 1,15 GB nó không cần).

### Nên tách làm hai dataset

Code là 164 KB, dữ liệu + cache là ~600 MB. Gộp chung nghĩa là mỗi lần sửa một
dòng code phải upload lại 600 MB.

| Dataset | Nội dung | Tần suất đổi |
|---|---|---|
| `fine-tune-ref` | cả folder `ref/` (results, DSC2026-LegalIR-main, models) | gần như không bao giờ |
| `fine-tune-code` | 5 file `.py` trong `fine_tune/` | mỗi lần sửa code |

Notebook tự dò: nếu `/kaggle/input/fine-tune-code` tồn tại thì lấy code ở đó,
không thì lùi về `REF/fine_tune`. Cả hai kiểu đều chạy được, không phải sửa gì.

5 file cần cho dataset code: `burst_common.py`, `torch_common.py`,
`finetune_jina.py`, `finetune_aiteamvn.py`, `finetune_vnlegal_lal.py`.
(`make_notebooks.py` chỉ dùng ở local, không cần upload.)

Ba file `.ipynb` upload thẳng lên Kaggle dưới dạng **notebook**, không nằm
trong dataset nào.

### Cách 1 — notebook (khuyến nghị)

Upload lần lượt từng file `.ipynb`, chạy xong một model rồi mới sang model sau:

```
01_finetune_vnlegal_lal.ipynb   ->  02_finetune_aiteamvn.ipynb  ->  03_finetune_jina.ipynb
```

Mỗi notebook cắt thành 9 cell code, **mỗi epoch một cell**. Sau mỗi epoch kết
quả ghi ngay xuống `/kaggle/working/<tag>/` và đóng thành `<tag>_results.zip`.

Hai chế độ chạy:

1. **Save Version → Save & Run All (Commit)** — chạy headless, đóng trình duyệt
   được, Kaggle tự lưu `/kaggle/working` thành output. An toàn nhất trước chuyện
   mất mạng hay ngắt server.
2. **Chạy tương tác** — tải `<tag>_results.zip` về sau mỗi epoch qua panel
   Output. Ở chế độ này `/kaggle/working` mất khi session chết, đừng dồn tới
   cuối mới tải.

**Chạy lại sau khi đứt:** cứ chạy lại từ cell 1. Notebook đọc `history.json` cũ,
bỏ qua epoch đã ghi và nạp lại `best_state.pt`, nên không tốn lại GPU cho phần
đã xong. Muốn làm lại sạch thì đặt `RESUME = False`.

Lưu ý: resume khôi phục **trọng số** chứ không khôi phục trạng thái optimizer,
nên epoch chạy tiếp sẽ khởi động lại momentum của Adam. Với 3 epoch thì đây là
đánh đổi chấp nhận được — thay vì phải lưu thêm 2× kích thước model mỗi lần.

### Cách 2 — script CLI

```python
REF = "/kaggle/input/fine-tune-ref/ref/fine_tune"

!python {REF}/finetune_vnlegal_lal.py --epochs 3
!python {REF}/finetune_aiteamvn.py    --epochs 3
!python {REF}/finetune_jina.py        --epochs 3
```

Cũng có resume y hệt (tắt bằng `--no-resume`).

Không cần copy file ra `/kaggle/working` — script chỉ **đọc** từ `--root` và
chỉ **ghi** vào `--work` (mặc định `/kaggle/working`), đã kiểm tra không có
thao tác ghi nào chạm vào thư mục input read-only.

Chạy ở local (không có `/kaggle/input`) thì y hệt, output đổi chỗ:

```bash
python finetune_vnlegal_lal.py --epochs 3 --work ./runs
```

Thêm `--eval-before-training` để chấm luôn model **chưa** fine-tune qua cùng
đường đánh giá — cho một mốc so sánh sạch, tách bạch phần thay đổi do trọng số
với phần thay đổi do đường inference.

Nếu hết VRAM: `--batch-size 2 --accum 8`, hoặc `--negatives 4`.

## Hai GPU T4

Nếu Kaggle cấp 2× T4, model tự được bọc `nn.DataParallel`, không cần đặt gì.

Chọn DataParallel chứ không phải DDP vì notebook chạy theo từng cell, mà DDP
phải dựng process group — phiền hơn nhiều so với cái nó đổi lại ở quy mô này.
Nhược điểm quen thuộc của DataParallel là gom output về GPU 0, nhưng ở đây gần
như không dính: cross-encoder trả về **một logit mỗi chuỗi**, bi-encoder trả về
**một vector 1024 chiều**, nên thứ chạy ngược về GPU 0 chỉ vài KB còn activation
(phần thực sự chiếm VRAM) nằm yên trên GPU của nó.

`scale_for_gpus` nhân `batch_size` lên 2 và chia `accum` xuống 2:

```
1 GPU:  batch_size=4  accum=4  ->  effective batch 16
2 GPU:  batch_size=8  accum=2  ->  effective batch 16   (y hệt)
```

Nghĩa là hai GPU chỉ rút ngắn thời gian, **không làm lệch phép toán huấn luyện**
so với chạy một GPU — số đo giữa hai cấu hình vẫn so sánh được. `eval_batch_size`
thì cứ nhân đôi vì không có bước optimizer nào cần giữ tương đương. Nếu `accum`
không chia hết cho số GPU, script để nguyên `batch_size`/`accum` và báo ra.

Checkpoint lưu ra đã được **strip tiền tố `module.`** mà DataParallel thêm vào,
nên `best_state.pt` từ lần chạy 2 GPU nạp được y hệt vào runner gốc (vốn dùng
`load_state_dict(..., strict=False)` — nếu còn `module.` thì nó sẽ không khớp
key nào và **im lặng** giữ nguyên trọng số gốc).

## Vì sao không dùng HNSW

Đã cân nhắc và **không đưa vào** — nó không chạm được vào chỗ tốn thời gian.

Trong vòng lặp fine-tune này không có bước tìm kiếm vector nào cả: tầng 1 đọc
hoàn toàn từ cache, negative lấy từ pool lexical đã cache, và mỗi truy vấn chỉ
chấm điểm ~39 ứng viên. Không có kNN để mà tăng tốc.

Chỗ duy nhất trong toàn hệ thống có tìm kiếm vector là `corpus_dense` — quét
toàn corpus. Đo thử ở đúng quy mô của nó (8.532 văn bản × tối đa 32 chunk =
273.024 vector, 1024 chiều, 1.000 truy vấn):

```
  8.532 vector x 1000 truy vấn:  0,20 s   (CPU)
100.000 vector x 1000 truy vấn:  3,24 s
273.024 vector x 1000 truy vấn:  6,47 s
```

6,5 giây trên CPU, và dưới 1 giây trên T4. Con số 67 phút trong README gốc là
thời gian **mã hoá** corpus bằng model embedding, không phải thời gian tìm kiếm.
HNSW thay thế cái 6 giây đó, trong khi riêng việc dựng index HNSW cho 273k
vector đã tốn 1–2 phút — tính ra chậm hơn, cộng thêm một dependency và một
nguồn lỗi mới.

Cái thực sự rút ngắn thời gian mỗi epoch là số lần forward của reranker
(~47.000 đoạn văn × 512 token), và đó đúng là thứ hai GPU chia đôi.

Nếu điều bạn muốn là **đào hard negative từ toàn corpus** bằng chính model vừa
fine-tune (kiểu ANCE) thì đó là chuyện khác và đáng làm — nhưng vẫn nên dùng
matmul thẳng chứ không phải HNSW, vì 8.532 văn bản chỉ mất 0,2 giây. Nói nếu
bạn muốn tôi thêm.

### Trọng số model lấy từ đâu

| Script | Thư mục local (nếu có trọng số thật) | Fallback HuggingFace |
|---|---|---|
| `finetune_jina.py` | `models/jina-reranker-v2-base-multilingual` | `jinaai/jina-reranker-v2-base-multilingual` |
| `finetune_aiteamvn.py` | `models/AITeamVN_Vietnamese_Embedding` | `AITeamVN/Vietnamese_Embedding` |
| `finetune_vnlegal_lal.py` | `models/vnlegal-lal` | `darklethelong/vnlegal-lal` |

Thư mục local chỉ được dùng khi thực sự chứa `*.safetensors` hoặc `*.bin` —
placeholder không tính. Trong gói ref hiện tại cả ba đều sẽ tải từ hub.

`finetune_jina.py` còn có `--init-checkpoint auto`: nếu
`results/jina_reranker/burst_pairwise_state.pt` có mặt thì fine-tune **tiếp**
từ checkpoint đã triển khai thay vì từ trọng số gốc. File đó không có trong gói
ref (197 MB, bị loại), nên mặc định sẽ khởi tạo từ trọng số HuggingFace và in
rõ điều đó ra log.

## Output — `<work>/<tag>/`

Chỉ epoch nào **vượt** mốc tốt nhất hiện tại mới được lưu, nên thư mục luôn giữ
đúng một state đáng dùng cộng với lịch sử đầy đủ để giải thích vì sao.

| File | Nội dung |
|---|---|
| `best_state.pt` | `{"state_dict": ...}` — nạp thẳng bằng `torch.load(...)["state_dict"]` + `strict=False`, đúng cách runner gốc nạp checkpoint Jina |
| `best_predictions.json` | `{qid: {"answer": [...]}}` sau ngưỡng động — đúng định dạng submission |
| `best_ranking.json` | top-20 xếp hạng hợp nhất mỗi truy vấn |
| `best_channel_scores.pkl` | `{qid: {doc: score}}` — thả thẳng vào pipeline làm score cache của kênh đó |
| `history.json` | Mọi epoch: recall/precision/F2 tổng, theo từng block, recall của riêng kênh, train loss, và dòng `baseline` từ cache |

Tiêu chí chọn state tốt nhất: `recall` chính thức (sau ngưỡng động), hòa thì
xét `f2`. Đây đúng là công thức trong `DSC2026-LegalIR-main/scoring.py`.

## Ba điều cần nhớ trước khi tin một con số

**Ngưỡng nhiễu của bài này là std 0.008 trên Recall.** Dự án đã có 4 lần
"thắng trên CV nhưng thua trên leaderboard thật". Chênh lệch nhỏ hơn 0.008
không phải bằng chứng của gì cả.

**Pooling của vnlegal-lal phải giữ CLS.** Last-token pooling từng cho CV cao
hơn (F2 +0.0127) nhưng giảm recall thật trên leaderboard và đã bị hoàn tác
ngày 2026-08-21. Script fine-tune vnlegal-lal huấn luyện chính biểu diễn CLS
để giải quyết mâu thuẫn đó từ phía huấn luyện, thay vì đổi pooling lúc suy luận.

**AITeamVN còn nuôi hai tầng sinh ứng viên khác** (dense expansion union-50 và
corpus dense index). Script cố ý giữ nguyên hai tầng đó từ cache: dựng lại
corpus index tốn ~67 phút GPU, và nếu chạy lại expansion thì tập ứng viên cũng
đổi theo, khiến số đo lẫn hai thay đổi vào nhau. Khi một checkpoint AITeamVN
đã chứng minh được mình ở tầng rerank, đó mới là lúc dựng lại cả hai index
bằng nó và đo lại toàn tuyến.

## Một quan sát từ dữ liệu cache

Chấm riêng từng kênh trên pool ứng viên 600 truy vấn:

```
jina         R@5 = 0.8617
dense        R@5 = 0.8603
e5           R@5 = 0.8467
expansion    R@5 = 0.8419
corpus       R@5 = 0.7975
vnlegal_lal  R@5 = 0.1011      <- hoán vị ngẫu nhiên cùng pool cho 0.113
```

Kênh `vnlegal_lal` với trọng số gốc và CLS pooling gần như không mang tín hiệu
xếp hạng (dải điểm [0.53, 1.00]). Nó vẫn được giữ vì đóng góp nhỏ qua LTR, nhưng
đây là kênh còn nhiều dư địa nhất — và là lý do nên fine-tune nó trước.
