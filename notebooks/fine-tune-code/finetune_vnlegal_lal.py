"""Fine-tune darklethelong/vnlegal-lal -- the `vnlegal_lal` channel of BURST.

Architecture kept identical to the shipped runner
(run_vnlegal_extra_channel_submission.score_vnlegal):

    AutoModel("darklethelong/vnlegal-lal")
    CLS token of last_hidden_state, L2-normalised
    score = passage vector . question vector, over 2 density-selected windows
    per document at max_length 512, document score = max over its windows

POOLING IS CLS AND MUST STAY CLS.  Last-token pooling scored better on CV
(F2 +0.0127 at equal Recall) but regressed real Recall on the CodaBench
leaderboard and was reverted on 2026-08-21.  Do not "fix" it here without an
A/B on the real leaderboard first.

What this script recomputes, and what it reuses
-----------------------------------------------
Reused from results/ (never recomputed, no GPU spent):
    layer 1  BM25 multi-branch retrieval, multistage top-20, dense expansion,
             corpus dense rank cap=32 -> the candidate pool is byte-identical
             every epoch
    layer 2  the `jina` and `dense` channels stay on their cached scores
Recomputed each epoch:
    layer 2  the `vnlegal_lal` channel only -- this model's own output
    layer 3  the 6-channel LTR fusion (CPU, ~2 s)
    layer 4  dynamic threshold alpha=0.15

This is by far the channel with the most headroom, and the measurement is
blunt about why.  Ranked on its own over the 600-query holdout pool, the
cached CLS scores reach Recall@5 = 0.101 -- below the 0.113 a random
permutation of the same pool gets, against 0.86 for `jina` and `dense`.  Their
range is [0.53, 1.00]: the CLS token of the stock checkpoint barely separates
relevant from irrelevant here, which is the same fact the pooling episode above
ran into from the other side.  Switching pooling at inference was tried and
lost on the real leaderboard.  Training the CLS representation to carry the
signal is the other way to resolve it, and it is what this script does, so the
channel keeps the exact inference form the submission already ships.

Expect a large channel-level jump and a small fusion-level one: the LTR ranker
has learned to give this near-noise channel very little weight, and the other
five channels already cover most of what it will start getting right.

Training data is the 1,050 labelled queries whose retrieval is already cached
and which are NOT in the 600-query LOBO evaluation set (train.json indices
0-749 and 850-1149).  Loss is InfoNCE over one gold document against hard
negatives mined from the cached lexical pool, plus in-batch negatives.

Outputs, under --work/vnlegal_lal/ (default /kaggle/working/vnlegal_lal/):
    best_state.pt            {"state_dict": ...} for AutoModel.load_state_dict
    best_predictions.json    {qid: {"answer": [...]}} after the dynamic threshold
    best_ranking.json        top-20 fused ranking per query
    best_channel_scores.pkl  {qid: {doc: score}} -- drops straight into
                             results/embedding_finetune/vnlegal_lal_cv_scores.pkl
                             shape, and into the runner's vnlegal_scores.pkl
    history.json             every epoch's metrics, including the cached baseline

Usage
    python finetune_vnlegal_lal.py --epochs 3
    python finetune_vnlegal_lal.py --epochs 3 --eval-before-training
"""

from __future__ import annotations

from transformers import AutoModel, AutoTokenizer

import burst_common as bc
import torch_common as tc
# The bi-encoder recipe is identical to AITeamVN's -- CLS pooling, InfoNCE over
# mined hard negatives -- so it is imported rather than duplicated.  train_epoch
# and evaluate are re-exported because the notebook drives them cell by cell.
from finetune_aiteamvn import add_bi_encoder_args, evaluate, run, train_epoch

__all__ = ["DEFAULT_MODEL", "HF_REPO", "evaluate", "load_model", "main",
           "run", "train_epoch"]

TAG = "vnlegal_lal"
CHANNEL = "vnlegal_lal"
DEFAULT_MODEL = "models/vnlegal-lal"
HF_REPO = "darklethelong/vnlegal-lal"


def load_model(root, model_path, gradient_checkpointing):
    """models/vnlegal-lal holds only a placeholder in the reproduction package
    (it exists so the runner skips a 1.15 GB download it does not need when the
    score cache is complete), so this normally resolves to the hub."""
    source = tc.resolve_model_source(root, model_path, HF_REPO)
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModel.from_pretrained(source)
    tc.enable_gradient_checkpointing(model, gradient_checkpointing)
    return model, tokenizer


def main():
    ap = bc.common_args(__doc__.splitlines()[0])
    add_bi_encoder_args(ap, DEFAULT_MODEL)
    ap.set_defaults(lr=1e-5, batch_size=4, accum=4, negatives=7, eval_batch_size=32)
    run(ap.parse_args(), TAG, CHANNEL, load_model)


if __name__ == "__main__":
    main()
