"""Generate the three Kaggle notebooks from one shared template.

The notebooks are the same run the .py scripts do, cut into cells so an
interrupted session loses at most the epoch that was in flight.  Keeping them
generated rather than hand-written stops the three from drifting apart; re-run
this after touching the template:

    python make_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

MODELS = [
    {
        "file": "01_finetune_vnlegal_lal.ipynb",
        "tag": "vnlegal_lal",
        "channel": "vnlegal_lal",
        "title": "1. Fine-tune vnlegal-lal — kênh `vnlegal_lal`",
        "module": "finetune_vnlegal_lal",
        "kind": "bi",
        "lr": 1e-5,
        "eval_batch_size": 32,
        "train_top_layers": 8,
        "max_gpus": 1,
        "why": (
            "Chạy model này **trước**. Chấm riêng trên pool ứng viên, kênh "
            "`vnlegal_lal` với trọng số gốc chỉ đạt Recall@5 = 0.1011 — thấp hơn "
            "cả hoán vị ngẫu nhiên (0.113), trong khi `jina`/`dense` đạt 0.86. "
            "Dải điểm [0.53, 1.00]: CLS token của checkpoint gốc gần như không "
            "mang tín hiệu xếp hạng. Đây là kênh còn nhiều dư địa nhất.\n\n"
            "**Pooling phải giữ CLS.** Last-token pooling từng cho CV cao hơn "
            "(F2 +0.0127) nhưng giảm recall thật trên leaderboard và đã bị hoàn "
            "tác ngày 2026-08-21. Notebook này huấn luyện chính biểu diễn CLS "
            "để giải quyết mâu thuẫn đó từ phía huấn luyện, giữ nguyên dạng "
            "inference mà bản nộp đang chạy."
        ),
    },
    {
        "file": "02_finetune_aiteamvn.ipynb",
        "tag": "aiteamvn",
        "channel": "dense",
        "title": "2. Fine-tune AITeamVN/Vietnamese_Embedding — kênh `dense`",
        "module": "finetune_aiteamvn",
        "kind": "bi",
        "lr": 1e-5,
        "eval_batch_size": 32,
        "train_top_layers": 8,
        "max_gpus": 1,
        "why": (
            "Model này còn nuôi hai tầng sinh ứng viên khác (dense expansion "
            "union-50 và corpus dense index). Notebook **cố ý giữ nguyên** hai "
            "tầng đó từ cache: dựng lại corpus index tốn ~67 phút GPU, và nếu "
            "chạy lại expansion thì tập ứng viên cũng đổi theo, khiến số đo lẫn "
            "hai thay đổi vào nhau.\n\n"
            "Khi một checkpoint ở đây đã chứng minh được mình ở tầng rerank, đó "
            "mới là lúc dựng lại cả hai index bằng nó và đo lại toàn tuyến."
        ),
    },
    {
        "file": "03_finetune_jina.ipynb",
        "tag": "jina",
        "channel": "jina",
        "title": "3. Fine-tune Jina reranker — kênh `jina`",
        "module": "finetune_jina",
        "kind": "cross",
        "lr": 1e-5,
        "eval_batch_size": 16,
        "train_top_layers": 0,
        "max_gpus": 0,
        "why": (
            "Đây là kênh mạnh nhất đang có (R@5 riêng = 0.8617) và **đã từng "
            "được fine-tune một lần** — checkpoint đó (`burst_pairwise_state.pt`, "
            "197 MB) bị loại khỏi gói ref nên notebook khởi tạo lại từ trọng số "
            "HuggingFace gốc. Nghĩa là epoch đầu nhiều khả năng còn thua baseline "
            "cache; đừng hoảng, hãy đọc cột `channel_only` trong history.\n\n"
            "Nếu bạn còn giữ `burst_pairwise_state.pt` ở đâu đó, upload kèm vào "
            "`results/jina_reranker/` rồi đặt `INIT_CHECKPOINT = \"auto\"` — nó sẽ "
            "fine-tune *tiếp* từ đó thay vì làm lại từ đầu, điểm xuất phát tốt "
            "hơn hẳn.\n\n"
            "Đây cũng là notebook chậm nhất: cross-encoder phải chạy qua từng "
            "cặp (câu hỏi, đoạn văn) thay vì mã hoá độc lập."
        ),
    },
]


def md(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source.rstrip("\n").split("\n")}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.rstrip("\n").split("\n")}


def build(model):
    tag, channel, kind = model["tag"], model["channel"], model["kind"]
    scorer = "score_cross_encoder" if kind == "cross" else "score_bi_encoder"
    cells = []

    cells.append(md(f"""\
# {model["title"]}

{model["why"]}

---

## Cách chạy để không mất kết quả

Notebook được cắt thành từng cell nhỏ, **mỗi epoch một cell**. Sau mỗi epoch,
kết quả được ghi ngay xuống `/kaggle/working/{tag}/` và đóng thành
`{tag}_results.zip` — nên nếu session chết ở epoch 3 thì epoch 1–2 vẫn còn
nguyên.

Hai chế độ chạy, chọn theo tình huống:

1. **Save Version → Save & Run All (Commit)** — chạy headless, đóng trình duyệt
   được, Kaggle tự lưu toàn bộ `/kaggle/working` thành output. Đây là cách an
   toàn nhất trước chuyện mất mạng hay ngắt server.
2. **Chạy tương tác** — chạy từng cell, và **tải `{tag}_results.zip` về sau mỗi
   epoch** qua panel Output bên phải. Ở chế độ này `/kaggle/working` sẽ mất khi
   session chết, nên đừng để dồn tới cuối mới tải.

Nếu phải chạy lại: **cứ chạy lại từ cell 1**. Notebook tự đọc `history.json` cũ,
bỏ qua những epoch đã ghi và nạp lại `best_state.pt`, nên không tốn lại thời
gian GPU của phần đã xong. Muốn làm lại sạch thì đặt `RESUME = False`.

**Notebook settings cần bật: GPU và Internet.**"""))

    cells.append(md("""\
## Cell 1 — Cấu hình

`REF` là dataset dữ liệu + cache (~600 MB, gần như không bao giờ đổi).
`CODE` là nơi chứa 5 file `.py`. Nếu bạn tách code thành một dataset nhỏ riêng
(~160 KB) thì mỗi lần sửa code chỉ upload lại chỗ đó; nếu không có, notebook
tự lùi về `REF/fine_tune` — nên cả hai kiểu đều chạy được, không cần sửa gì."""))
    cells.append(code(f'''\
import os, sys, json, time
from pathlib import Path

# Giảm phân mảnh bộ nhớ GPU — chính thông báo OOM của PyTorch gợi ý.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REF  = Path("/kaggle/input/fine-tune-ref/ref")   # dataset nhtclone/fine-tune-ref
CODE = Path("/kaggle/input/fine-tune-code")      # dataset code riêng (tuỳ chọn)
WORK = Path("/kaggle/working")

if not CODE.exists():
    CODE = REF / "fine_tune"                     # code nằm luôn trong dataset ref

EPOCHS          = 3
RESUME          = True     # False = bỏ qua history/checkpoint cũ, chạy lại từ đầu
LR              = {model["lr"]}
BATCH_SIZE      = 4        # số truy vấn mỗi bước
ACCUM           = 4        # gradient accumulation
NEGATIVES       = 7        # hard negative mỗi truy vấn mỗi bước
NEGATIVE_DEPTH  = 48       # lấy negative sâu bao nhiêu trong pool lexical đã cache
MAX_LENGTH      = 512
EVAL_BATCH_SIZE = {model["eval_batch_size"]}
PRECISION       = "auto"   # auto | fp16 | bf16 | fp32 -- xem cell 4
MAX_GPUS        = {model["max_gpus"]}        # 1 = tắt DataParallel (nhẹ VRAM hơn); 0 = dùng hết GPU
TRAIN_TOP_LAYERS = {model["train_top_layers"]}      # N block trên cùng + head; 0 = toàn bộ (xem cell 4)
PASSAGES_PER_DOC_TRAIN = 1   # lúc train; lúc đánh giá luôn là 2 như pipeline gốc
TRAIN_QUERIES   = None     # None = dùng cả 1050 truy vấn
SEED            = 2026
{'INIT_CHECKPOINT = "auto"   # "auto" = tiếp tục từ burst_pairwise_state.pt nếu có' if kind == "cross" else 'TEMPERATURE     = 0.05     # InfoNCE'}
{'LOSS            = "pairwise"' if kind == "cross" else 'IN_BATCH_NEGATIVES = True'}

assert REF.exists(), f"Không thấy {{REF}} — kiểm tra lại dataset dữ liệu đã attach chưa"
assert (CODE / "burst_common.py").exists(), f"Không thấy code trong {{CODE}}"
sys.path.insert(0, str(CODE))
os.environ["BURST_ROOT"] = str(REF)
os.environ["BURST_WORK"] = str(WORK)
WORK.mkdir(parents=True, exist_ok=True)
print("dữ liệu:", REF)
print("code   :", CODE)

import torch
print("torch", torch.__version__, "| CUDA:",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "KHÔNG CÓ GPU")
assert torch.cuda.is_available(), "Bật GPU trong Notebook settings rồi chạy lại"

# Kaggle ghim notebook vào phiên bản dataset lúc attach: upload New Version KHÔNG
# tự cập nhật notebook đang mở. Dòng này bắt lỗi đó ngay thay vì để nó nổ ở cell 4.
import burst_common as _bc, torch_common as _tc
_bc.check_code_version()
'''))

    cells.append(md("## Cell 2 — Dựng lại tập đánh giá 600 truy vấn từ cache\n\n"
                    "Chỉ CPU, khoảng 1–3 phút. Cell này chạy trước phần GPU có "
                    "chủ đích: nếu đường dẫn hay cache có vấn đề thì hỏng ở đây, "
                    "trước khi tốn thời gian GPU.\n\n"
                    "Con số baseline in ra phải đúng bằng "
                    "**recall 0.9511 / F2 0.6144** — đó là bản đã nộp."))
    cells.append(code('''\
import burst_common as bc
import torch_common as tc

documents = bc.DocumentStore(REF / bc.DATA_SUBDIR, preload=True)
bundle    = bc.build_eval_bundle(REF, documents)

baseline, _, _ = bc.lobo_evaluate(bundle)
print(f"\\nBASELINE (toàn bộ 6 kênh từ cache)")
print(f"  recall    = {baseline['recall']:.4f}   <- bản đã nộp: 0.9511")
print(f"  precision = {baseline['precision']:.4f}")
print(f"  f2        = {baseline['f2']:.4f}   <- bản đã nộp: 0.6144")
for name, b in baseline["blocks"].items():
    print(f"  block {name}: recall={b['recall']:.4f} f2={b['f2']:.4f}")
'''))

    cells.append(md("## Cell 3 — Tập huấn luyện\n\n"
                    "1.050 truy vấn có nhãn, đã có sẵn cache retrieval, **không "
                    "giao** với 600 truy vấn đánh giá. Hard negative lấy thẳng "
                    "từ pool lexical đã cache — không chạy lại retrieval."))
    cells.append(code(f'''\
examples = bc.build_train_pool(REF, exclude=bundle.all_ids,
                               negatives=NEGATIVE_DEPTH, limit=TRAIN_QUERIES,
                               seed=SEED, documents=documents)

recorder = bc.RunRecorder(WORK, "{tag}", baseline, resume=RESUME)
print("Ghi kết quả vào:", recorder.dir)
'''))

    cells.append(md("""\
## Cell 4 — Nạp model

Tải trọng số từ HuggingFace (cần bật Internet). Nếu đang resume, trọng số tốt
nhất của lần chạy trước sẽ được nạp đè.

Nếu Kaggle cấp **2× T4**, model được bọc `nn.DataParallel` để mỗi batch chia
đôi cho hai GPU. Chọn DataParallel chứ không phải DDP vì notebook chạy theo
từng cell, mà DDP thì phải dựng process group — phiền hơn nhiều so với cái nó
đổi lại. Nhược điểm quen thuộc của DataParallel (gom output về GPU 0) gần như
không dính ở đây: cross-encoder trả về một logit mỗi chuỗi, bi-encoder trả về
một vector 1024 chiều, nên thứ chạy ngược về chỉ vài KB còn activation thì nằm
yên tại chỗ.

### Kiểm tra độ chính xác số học

T4 (compute capability 7.5) **không có bf16**, nên autocast rơi về fp16 — dải
số chỉ tới 65504. Model được huấn luyện ở bf16 (dải ~1e38) có thể sinh
activation vượt ngưỡng đó, thành `inf`, rồi `F.normalize` biến `inf/inf` thành
`NaN` và loss là NaN ngay từ bước đầu mà log không nói gì.

Cell này chạy thử **một batch** rồi quyết định: nếu fp16 cho kết quả không hữu
hạn, nó tự chuyển sang fp32. fp32 chậm hơn khoảng 2 lần và tốn bộ nhớ hơn,
nhưng đổi lấy một epoch không NaN thì quá rẻ. Ép tay bằng
`PRECISION = "fp32"` ở cell 1."""))
    if kind == "cross":
        load_code = '''\
import finetune_jina as ft

model, tokenizer = ft.load_model(REF, ft.DEFAULT_MODEL, INIT_CHECKPOINT,
                                 gradient_checkpointing=True)'''
    else:
        load_code = f'''\
import {model["module"]} as ft

model, tokenizer = ft.load_model(REF, ft.DEFAULT_MODEL,
                                 gradient_checkpointing=True)'''
    cells.append(code(f'''\
{load_code}

if RESUME:
    recorder.load_best_state(model)

print(f"tham số: {{sum(p.numel() for p in model.parameters())/1e6:.0f}}M")

device, amp_dtype, n_gpus = tc.setup(SEED)
if MAX_GPUS:
    n_gpus = min(n_gpus, MAX_GPUS)
model.to(device)

# Chạy thử 1 batch: fp16 có tràn số trên model này không?
probe_texts = [documents[d] for d in examples[0].negatives[:4]]
amp_dtype = tc.choose_precision(model, tokenizer, probe_texts, device, amp_dtype,
                                MAX_LENGTH, kind="{'cross' if kind == 'cross' else 'bi'}",
                                requested=PRECISION)

# Pooling có phân biệt được hai đoạn văn khác nhau không? Nếu không thì kênh này
# không thể xếp hạng, và train nó chỉ đẩy token embedding ra vô cực.
tc.check_pooling_discriminates(model, tokenizer, probe_texts, device, amp_dtype,
                               MAX_LENGTH, kind="{'cross' if kind == 'cross' else 'bi'}",
                               label="{tag}")
tc.freeze_lower_layers(model, TRAIN_TOP_LAYERS)
tc.report_memory_budget(model, n_gpus, device)

model = tc.wrap_parallel(model, n_gpus)     # 2x T4 -> mỗi batch chia đôi
'''))

    cells.append(md(f"""\
## Cell 5 — Optimizer, scheduler, và hai hàm chạy

`train_one_epoch(n)` huấn luyện một epoch. `evaluate_and_record(n)` chấm lại
**chỉ kênh `{channel}`**, ráp vào LTR fusion 6 kênh, áp ngưỡng động α=0.15,
rồi ghi xuống đĩa ngay.

`scale_for_gpus` nhân `batch_size` lên và chia `accum` xuống theo số GPU, nên
**effective batch không đổi** — hai GPU chỉ rút ngắn thời gian chứ không làm
lệch phép toán huấn luyện so với chạy một GPU. `eval_batch_size` thì cứ nhân
lên vì không có bước optimizer nào cần giữ tương đương."""))
    if kind == "cross":
        extra_args = "args.loss = LOSS\nargs.temperature = 1.0"
    else:
        extra_args = ("args.temperature = TEMPERATURE\n"
                      "args.no_in_batch_negatives = not IN_BATCH_NEGATIVES")
    cells.append(code(f'''\
from types import SimpleNamespace

args = SimpleNamespace(
    root=REF, work=WORK, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE,
    accum=ACCUM, negatives=NEGATIVES, negative_depth=NEGATIVE_DEPTH,
    max_length=MAX_LENGTH, eval_batch_size=EVAL_BATCH_SIZE,
    passages_per_doc=PASSAGES_PER_DOC_TRAIN, train_queries=TRAIN_QUERIES,
    warmup_ratio=.1, weight_decay=.01, max_grad_norm=1.0, seed=SEED,
    gradient_checkpointing=True, keep_every_epoch=False, resume=RESUME,
    eval_before_training=False, precision=PRECISION,
    train_top_layers=TRAIN_TOP_LAYERS, max_gpus=MAX_GPUS,
)
{extra_args}

tc.scale_for_gpus(args, n_gpus)     # giữ nguyên effective batch, chỉ chia việc

sampler = tc.GroupSampler(examples, documents, NEGATIVES,
                          PASSAGES_PER_DOC_TRAIN, seed=SEED)
groups_per_epoch = (len(examples) + args.batch_size - 1) // args.batch_size
total_steps = max(1, EPOCHS * groups_per_epoch // args.accum)

optimizer = tc.make_optimizer(tc.unwrap(model), LR, args.weight_decay)
scheduler = tc.make_scheduler(optimizer, total_steps, args.warmup_ratio)
try:
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
except TypeError:
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))

print(f"{{len(examples)}} truy vấn, {{groups_per_epoch}} nhóm/epoch, "
      f"{{total_steps}} bước optimizer cho {{EPOCHS}} epoch")


def train_one_epoch(epoch):
    if recorder.already_done(epoch):
        print(f"epoch {{epoch}} đã ghi từ lần chạy trước — bỏ qua")
        return None
    started = time.perf_counter()
    loss = ft.train_epoch(model, tokenizer, sampler, args, optimizer, scheduler,
                          scaler, device, amp_dtype, epoch)
    print(f"epoch {{epoch}}: loss={{loss:.4f}} ({{(time.perf_counter()-started)/60:.1f}} phút)")
    return loss


def evaluate_and_record(epoch, loss=None):
    if recorder.already_done(epoch):
        print(f"epoch {{epoch}} đã ghi từ lần chạy trước — bỏ qua")
        return
    started = time.perf_counter()
    questions = {{q: bundle.queries[q][0] for q in bundle.all_ids}}
    scores = tc.{scorer}(
        model, tokenizer, questions, bundle.extended, documents, bundle.all_ids,
        device, amp_dtype, max_length=MAX_LENGTH, batch_size=EVAL_BATCH_SIZE,
        label="{tag}")
    metrics, ranked, predictions = bc.lobo_evaluate(
        bundle, {{"{channel}": scores}},
        {{"{channel}": bc.rank_by(bundle.extended, scores)}})
    metrics["channel_only"] = bc.channel_metrics(bundle, scores)
    improved = recorder.consider(epoch, metrics, tc.unwrap(model).state_dict(),
                                 ranked, predictions, extra={{"train_loss": loss}})
    if improved:
        recorder.save_channel_scores(scores)
    recorder.package()          # zip nhỏ, tải về được ngay
    print(f"  kênh `{channel}` riêng: R@5={{metrics['channel_only']['Recall@5']:.4f}} "
          f"(cache gốc: {{bc.channel_metrics(bundle, bundle.scores['{channel}'])['Recall@5']:.4f}})")
    print(f"  chấm điểm mất {{(time.perf_counter()-started)/60:.1f}} phút")
'''))

    for epoch in (1, 2, 3):
        cells.append(md(f"## Cell {5+epoch} — Epoch {epoch}\n\n"
                        + ("Chạy xong nhớ tải `{}_results.zip` về nếu đang chạy "
                           "tương tác.".format(tag) if epoch < 3 else
                           "Epoch cuối.")))
        cells.append(code(f'''\
loss = train_one_epoch({epoch})
evaluate_and_record({epoch}, loss)
'''))

    cells.append(md("## Cell 9 — Tổng kết\n\n"
                    "Bảng lịch sử đầy đủ và danh sách file để tải về."))
    cells.append(code('''\
history = json.loads((recorder.dir / "history.json").read_text(encoding="utf-8"))

print(f"{'epoch':>9s} {'recall':>8s} {'prec':>7s} {'f2':>7s} {'kênh riêng':>11s}  lưu")
for row in history["history"]:
    channel_only = row.get("channel_only", {}).get("Recall@5")
    print(f"{str(row['epoch']):>9s} {row['recall']:8.4f} {row['precision']:7.4f} "
          f"{row['f2']:7.4f} {(f'{channel_only:.4f}' if channel_only else '-'):>11s}  "
          f"{'YES' if row.get('improved') else ''}")

print(f"\\nEpoch tốt nhất: {history['best_epoch']}")
if history["best_epoch"] is None:
    print("Chưa epoch nào vượt baseline cache — giữ nguyên trọng số gốc.")
print("\\nNgưỡng nhiễu của bài này là std 0.008 trên Recall. Chênh lệch nhỏ hơn "
      "khoảng đó không phải bằng chứng của gì cả — dự án đã có 4 lần "
      "'thắng CV, thua leaderboard thật'.")

print("\\nFile trong /kaggle/working:")
for path in sorted(WORK.rglob("*")):
    if path.is_file():
        print(f"  {path.relative_to(WORK)}  ({path.stat().st_size/2**20:.1f} MB)")
'''))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    for model in MODELS:
        path = HERE / model["file"]
        path.write_text(json.dumps(build(model), ensure_ascii=False, indent=1),
                        encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
