# Phương pháp fine-tune (reranker + embedding)

Ghi lại cách 3 script `finetune_jina.py`, `finetune_aiteamvn.py`, `finetune_vnlegal_lal.py`
tạo dữ liệu huấn luyện và tối ưu model, để tra cứu lại khi cần mà không phải đọc lại code.

## 1. Tổng quan

Cả 3 model dùng chung một khung: **fine-tune có giám sát, hard-negative mining trực tiếp
từ retrieval cache đã có sẵn** (không chạy lại retrieval, không sinh dữ liệu tổng hợp).

- **(Cập nhật 2026-09-17)** Train trên toàn bộ 1650 truy vấn có cache retrieval, kể cả
  600 truy vấn của block LOBO (`burst_common.py:build_train_pool` không còn nhận
  `exclude=bundle.all_ids` như trước). `build_eval_bundle` vẫn dựng đúng 600 truy vấn đó
  để chấm điểm mỗi epoch, nhưng vì chúng nay cũng nằm trong train, số đo không còn tách
  bạch khỏi dữ liệu train — không dùng số này làm ước lượng generalization, chỉ dùng để
  chọn `best_state.pt` của chính lần chạy. Xem `README.md#dữ-liệu-train`.
- Không dùng augmentation dưới bất kỳ hình thức nào (không paraphrase, không back-translate,
  không thêm nhiễu, không sinh negative tổng hợp) — đã grep toàn bộ code, không có.

## 2. Positive được tạo ra bằng cách nào

```python
question, gold = queries[q]          # burst_common.py:772 — nhãn gốc trong train.json
```

- `gold` là tập ID tài liệu đúng lấy thẳng từ nhãn đã gán sẵn, không suy luận thêm.
- Nếu 1 câu hỏi có nhiều gold, mỗi **step** huấn luyện chỉ bốc **1 gold ngẫu nhiên**
  (`GroupSampler.group`, `torch_common.py:697`, `self.rng.choice(example.positives)`) —
  không dùng hết tất cả gold cùng lúc trong 1 step.
- Nếu có truyền `documents` (corpus) vào `build_train_pool`, gold bị lọc lại:
  `gold = {d for d in gold if d in corpus}`. Nếu sau lọc gold rỗng → query bị `skip`
  hoàn toàn khỏi train pool.

## 3. Negative được tạo ra bằng cách nào

```python
pool = raw_union(retrieval[q])                          # RRF union các nhánh BM25 đã cache
negs = [d for d in pool if d not in gold][:negatives]    # loại gold, cắt ở negative_depth (mặc định 48)
```

(`burst_common.py:build_train_pool`, dòng ~773-784)

- Negative = tài liệu mà chính retrieval pipeline (BM25 multi-branch + RRF fusion) đang
  **xếp hạng cao nhất mà sai** cho câu hỏi đó — hard negative thật, đúng distractor mà
  model sẽ gặp lúc suy luận thật, không phải negative ngẫu nhiên hay tổng hợp.
- Nếu pool rỗng sau khi trừ gold (không mine được negative nào) → query cũng bị `skip`.
- Mỗi **step** training, `GroupSampler.group()` chỉ bốc ngẫu nhiên `negatives` (mặc định
  7) trong số tối đa 48 đã mine, xáo trộn lại mỗi lần gọi → nhiều epoch sẽ thấy nhiều tổ
  hợp negative khác nhau từ cùng một pool cố định (không tạo thêm negative mới).
- Bi-encoder (aiteamvn/vnlegal-lal) có thêm **in-batch negative** trong loss InfoNCE
  (`in_batch=True` mặc định): documents của các query khác trong cùng batch cũng được
  dùng làm negative "miễn phí" vì đã encode sẵn — đây là tận dụng thêm, không phải sinh
  mới.

## 4. Windowing văn bản (áp dụng cho cả positive lẫn negative)

Không đưa nguyên văn tài liệu vào model — chỉ đưa **1 cửa sổ ~220 từ** chọn theo độ liên
quan tới câu hỏi (`top_passages`, overlap 70 từ), `passages_per_doc=1` lúc train, dùng
`2` cửa sổ (max-pooled) lúc eval để khớp pipeline suy luận thật.

## 5. Kiến trúc & loss theo từng model

### Reranker — Jina (`finetune_jina.py`)
- **Cross-encoder**: input là cặp `(câu hỏi, đoạn văn)` đi thẳng qua model → 1 logit/cặp.
- Loss mặc định: **pairwise RankNet** — `F.softplus(negative - positive).mean()`
  (`torch_common.py:711`, khớp objective của checkpoint gốc `burst_pairwise_state.pt`).
  Có cờ `--loss listwise` để đổi sang **softmax cross-entropy có temperature**
  (`listwise_loss`, coi positive là class đúng trong nhóm `1+negatives` ứng viên).
- Document score lúc eval = max qua các passage-window của tài liệu đó.

### Embedding — AITeamVN / vnlegal-lal (`finetune_aiteamvn.py`, dùng lại cho vnlegal-lal)
- **Bi-encoder**: encode câu hỏi và tài liệu riêng biệt.
- Pooling: **CLS token của `last_hidden_state`, sau đó L2-normalize**.
- Similarity = dot product của 2 vector đã normalize = cosine similarity.
- Loss: **InfoNCE, temperature 0.05** (`infonce_loss`, `torch_common.py:729`) — softmax
  cross-entropy trên ma trận similarity `(groups × tất cả documents trong batch)`.
- **Guard đặc biệt**: `check_pooling_discriminates` (`torch_common.py:269`) thử vài văn
  bản khác nhau trước khi train; nếu CLS-pooling cho ra vector gần giống hệt nhau thì
  **từ chối train**. Đây là lý do `01_finetune_vnlegal_lal` bị bỏ — pooling CLS trên
  model dạng causal-decoder không phân biệt được nội dung (vector chỉ phụ thuộc token
  đầu).

### Chung cho cả 3
- Optimizer: AdamW + linear-warmup scheduler (`make_scheduler`).
- AMP (autocast) cho forward, loss tính ngoài autocast để ổn định số học; loss không hữu
  hạn thì skip backward thay vì lan NaN; gradient accumulation, gradient clipping.
- Tùy chọn `--train-top-layers`: đóng băng các block dưới, chỉ train block trên cùng +
  head (`freeze_lower_layers`).
- Sau mỗi epoch: đánh giá trên 600 truy vấn LOBO holdout; `RunRecorder` lưu **best state
  theo validate của chính run này**, không phụ thuộc việc có vượt baseline cache hay
  không (mỗi history row vẫn ghi `beats_baseline` để biết fine-tune có thật sự vượt
  baseline đã cache hay chưa).

## 6. Những gì KHÔNG được dùng để tạo positive/negative

- Không augmentation văn bản (paraphrase, back-translation, đồng nghĩa, thêm nhiễu...).
- Không negative ngẫu nhiên từ toàn corpus.
- Không tự sinh nhãn (không pseudo-labeling).
- Không train trên truy vấn không có cache retrieval, hoặc không có gold trong corpus,
  hoặc không mine được negative nào — các truy vấn này bị `skip`, tính vào biến đếm
  `skipped` khi build train pool.
