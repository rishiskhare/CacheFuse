# SPDX-License-Identifier: Apache-2.0
"""Command-line evaluation of training-free cross-model KV-cache reuse (direct copy).

Run ``python -m cachefuse.evaluate`` (see ``--help``). Per question the receiver answers
from its own cache (``receiver``), from the transferred sharer cache alone
(``replace``), and from blends ``(1-theta) own + theta transferred`` (``theta``) for
each theta; ``sharer`` is the big model alone (the ceiling). A ``t2t`` (text-to-text)
baseline has the sharer *generate* a one-sentence background hint prepended to the
receiver prompt. Accuracy is the next-token argmax over the option-letter tokens after
"Answer:".
"""

# Standard
import gc
import json
import logging
import os
from dataclasses import dataclass
from typing import TextIO

# Third Party
import torch
from transformers import (
    DynamicCache,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    set_seed,
)

# First Party
from cachefuse import core, transfers
from cachefuse.benchmarks import (
    LETTERS,
    MCQuestion,
    build_prompt,
    build_t2t_prompt,
    load_mc,
    render_question,
)
from cachefuse.config import Settings
from cachefuse.transfers import CacheTransfer

logger = logging.getLogger(__name__)

# Sharer instruction for the text-to-text baseline, verbatim from Cache-to-Cache
# (arXiv:2510.03215, App. A.3.6): the sharer writes one background sentence, not the
# answer; that sentence is prepended to the receiver prompt by `build_t2t_prompt`.
T2T_INSTRUCTION = (
    "In one clear sentence, describe the most essential background knowledge needed to "
    "answer the question. Do NOT directly solve or give the answer.\n\n"
)


@dataclass(frozen=True)
class EvalResult:
    """Per-treatment accuracy from a run.

    Accuracy averages over only the successfully scored questions. `failed` holds the
    0-based indices that errored under ``on_error="skip"`` (rerunning with the same
    ``--checkpoint`` retries exactly these).
    """

    accuracy: dict[str, float]  # treatment -> accuracy (t2t added if score_t2t)
    n: int  # questions in the split
    n_scored: int  # questions successfully scored
    failed: list[int]  # indices that errored and were skipped


@torch.no_grad()
def _letter_logits(
    model: PreTrainedModel, ids: torch.Tensor, letter_ids: list[int], past: DynamicCache
) -> torch.Tensor:
    """Next-token logits over `letter_ids` after feeding the last token of `ids`."""
    out = model(input_ids=ids[:, -1:], past_key_values=past, use_cache=False)
    logits = out.logits[0, -1]
    return torch.stack([logits[i] for i in letter_ids])


@torch.no_grad()
def _predict(
    model: PreTrainedModel,
    ids: torch.Tensor,
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
    letter_ids: list[int],
    dev: str,
) -> int:
    """Index of the argmax option letter, decoding from the given per-layer KV cache."""
    past = core.make_cache(keys, values, dev)
    return int(_letter_logits(model, ids, letter_ids, past).argmax().item())


def _layers(cache: DynamicCache) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Per-layer (keys, values) of a prefill cache, all positions kept.

    The cache is built by prefilling the prompt *without* its final token, so it already
    covers ``0:S-2`` and is the decode 'past'; that final token is fed once when scoring
    the option-letter logits.
    """
    return (
        [layer.keys for layer in cache.layers],
        [layer.values for layer in cache.layers],
    )


@torch.no_grad()
def _generate_hint(
    sender: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    item: MCQuestion,
    device: str,
    max_new_tokens: int,
) -> str:
    """Greedily generate the sharer's one-sentence background hint for `item`.

    The sharer is prompted with `T2T_INSTRUCTION` over the question via the tokenizer's
    chat template when present (so an instruct model follows it; thinking is disabled to
    keep the hint short). Returns the continuation past the prompt, stripped.
    """
    user = T2T_INSTRUCTION + render_question(item)
    if tok.chat_template:
        enc = tok.apply_chat_template(
            [{"role": "user", "content": user}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
        )
    else:
        enc = tok(user, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = sender.generate(
        input_ids=ids,
        attention_mask=mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_id,
    )
    return tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()


def _free_memory(dev: str) -> None:
    """Release cached allocations after a failure so the next question can proceed.

    A CUDA OOM leaves the allocator holding fragmented reserved memory; collecting
    Python garbage (which may still own tensors) and emptying the allocator cache hands
    the retry/next question a clean pool instead of compounding the OOM.
    """
    gc.collect()
    if torch.device(dev).type == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def _t2t_cache(
    cfg: Settings,
    sender: PreTrainedModel,
    receiver: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    item: MCQuestion,
    dev: str,
    sdev: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Build the receiver's t2t decode cache: sender hint (`sdev`) + receiver prefill.

    The sender generates the one-sentence hint on its own device `sdev`, the hint is
    prepended to the receiver's prompt, and the receiver prefills it on `dev`. Returns
    ``(keys, values, ids)`` -- the prefill cache over all-but-last token and the full
    t2t prompt ids, used to score t2t accuracy.
    """
    hint = _generate_hint(sender, tok, item, sdev, cfg.t2t_max_new_tokens)
    ids = tok(build_t2t_prompt(item, hint), return_tensors="pt").input_ids.to(dev)
    keys, values = _layers(core.prefill(receiver, ids[:, :-1]))
    return keys, values, ids


@torch.no_grad()
def _run_question(
    cfg: Settings,
    sender: PreTrainedModel,
    receiver: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    transfer: CacheTransfer,
    item: MCQuestion,
    letter_ids: list[int],
) -> dict:
    """Score one question across every treatment.

    Accuracy is cheap single-forward scoring (no generation) for ``receiver``,
    ``replace``, ``sharer``, and ``theta{value}`` at every theta; with
    ``cfg.score_t2t`` the generated ``t2t`` baseline is scored too. Returns
    ``{"correct": {treatment: 0/1}}``.
    Any exception propagates to the caller (see ``cfg.on_error``).
    """
    dev = cfg.device  # receiver's device; fusion happens here
    sdev = cfg.sender_device or cfg.device  # sender's device (may be a second GPU)
    thetas = cfg.thetas
    ids = tok(build_prompt(item), return_tensors="pt").input_ids.to(dev)
    cand = letter_ids[: len(item.options)]
    prompt = ids[:, :-1]  # prefill all-but-last; the last token is fed when scoring
    # The sender runs on `sdev`; feed it the same tokens there. When sdev == dev these
    # `.to()` calls are no-ops, so the single-GPU path is unchanged.
    s_ids, s_prompt = ids.to(sdev), prompt.to(sdev)

    rk, rv = _layers(core.prefill(receiver, prompt))  # on dev
    tk, tv = _layers(transfer(s_prompt))  # sender prefills on sdev; cache moved to dev
    sk, sv = _layers(
        core.prefill(sender, s_prompt)
    )  # on sdev (for the sharer baseline)
    correct = {
        "receiver": int(_predict(receiver, ids, rk, rv, cand, dev) == item.gold),
        "replace": int(_predict(receiver, ids, tk, tv, cand, dev) == item.gold),
        "sharer": int(_predict(sender, s_ids, sk, sv, cand, sdev) == item.gold),
    }
    for theta in thetas:
        fk = [(1 - theta) * a + theta * b for a, b in zip(rk, tk, strict=True)]
        fv = [(1 - theta) * a + theta * b for a, b in zip(rv, tv, strict=True)]
        correct[f"theta{theta:g}"] = int(
            _predict(receiver, ids, fk, fv, cand, dev) == item.gold
        )
    if cfg.score_t2t:
        # Scored after the cheap treatments and by greedy argmax, so it never changes
        # them.
        tk2, tv2, t2t_ids = _t2t_cache(cfg, sender, receiver, tok, item, dev, sdev)
        correct["t2t"] = int(
            _predict(receiver, t2t_ids, tk2, tv2, cand, dev) == item.gold
        )

    return {"correct": correct}


def _checkpoint_config(cfg: Settings) -> dict:
    """The run fingerprint a checkpoint is bound to; a resume must match it exactly."""
    return {
        "sender": cfg.sender,
        "receiver": cfg.receiver,
        "benchmark": cfg.benchmark,
        "split": cfg.split,
        "num_examples": cfg.num_examples,
        "thetas": list(cfg.thetas),
        "score_t2t": cfg.score_t2t,
    }


def _load_checkpoint(path: str, cfg: Settings) -> dict[int, dict]:
    """Load prior per-question records (index -> record) from the JSONL `path`.

    The first line is a ``{"config": ...}`` header; the rest are per-question records.
    Raises ValueError if the header is missing or its fingerprint differs from `cfg`
    (reusing one path across incompatible configs would corrupt the index-keyed data).
    """
    records: dict[int, dict] = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        if not header.strip():
            return records
        stored = json.loads(header).get("config")
        if stored != _checkpoint_config(cfg):
            raise ValueError(
                f"checkpoint {path!r} was written for a different run config "
                f"({stored}); use a fresh --checkpoint path"
            )
        for line in f:
            if line.strip():
                rec = json.loads(line)
                records[rec["i"]] = rec
    return records


def _open_checkpoint(cfg: Settings) -> TextIO | None:
    """Open the JSONL checkpoint for append, writing the config header if it is new.

    Returns ``None`` when no checkpoint path is configured. The file is opened in append
    mode so a resume adds to (never truncates) the prior run's records.
    """
    if not cfg.checkpoint:
        return None
    fresh = not os.path.exists(cfg.checkpoint)
    if fresh:
        os.makedirs(os.path.dirname(cfg.checkpoint) or ".", exist_ok=True)
    ck = open(cfg.checkpoint, "a", encoding="utf-8")
    if fresh:
        ck.write(json.dumps({"config": _checkpoint_config(cfg)}) + "\n")
        ck.flush()
    return ck


def _aggregate(
    records: dict[int, dict],
    cache_treatments: list[str],
    n: int,
) -> EvalResult:
    """Reduce per-question records into the run's accuracy.

    Accuracy averages over only the **successfully scored** questions; failed indices
    are reported separately so they can be retried.
    """
    ok = [r for r in records.values() if r["ok"]]
    failed = sorted(i for i, r in records.items() if not r["ok"])
    n_scored = len(ok)
    accuracy = (
        {t: sum(r["correct"][t] for r in ok) / n_scored for t in cache_treatments}
        if n_scored
        else {}
    )
    return EvalResult(accuracy, n, n_scored, failed)


@torch.no_grad()
def evaluate(
    cfg: Settings,
    sender: PreTrainedModel,
    receiver: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    transfer: CacheTransfer,
    data: list[MCQuestion],
    letter_ids: list[int],
) -> EvalResult:
    """Score every treatment over `data` and return an `EvalResult`.

    Each question is scored by `_run_question` (single-forward accuracy for every
    treatment). With ``cfg.checkpoint`` set, one JSONL line is flushed per question, so
    a crash loses at most the in-flight question and rerunning with the same path
    resumes and retries failures. With ``cfg.on_error="skip"`` a question that raises
    (e.g. a CUDA OOM) is logged, its CUDA memory freed, and the run continues, its index
    returned in `EvalResult.failed`. Running accuracies print every 25 rows.
    """
    n = len(data)
    if n == 0:
        raise ValueError(
            f"no questions to evaluate (benchmark {cfg.benchmark!r}); nothing to score"
        )
    cache_treatments = [
        "receiver",
        *(f"theta{theta:g}" for theta in cfg.thetas),
        "replace",
        "sharer",
    ]
    if cfg.score_t2t:
        cache_treatments.append("t2t")

    # Resume: a prior `ok` record means "done" (skip it); a failed one is retried.
    records: dict[int, dict] = {}
    if cfg.checkpoint and os.path.exists(cfg.checkpoint):
        records = _load_checkpoint(cfg.checkpoint, cfg)

    # Running accuracy counters, seeded from any resumed `ok` records.
    correct = dict.fromkeys(cache_treatments, 0)
    for r in records.values():
        if r["ok"]:
            for t in cache_treatments:
                correct[t] += r["correct"][t]
    n_done = sum(1 for r in records.values() if r["ok"])
    if records:
        logger.info(
            "resuming from %s: %d already scored, %d to run/retry",
            cfg.checkpoint,
            n_done,
            n - n_done,
        )

    ck = _open_checkpoint(cfg)
    try:
        for i, item in enumerate(data):
            prior = records.get(i)
            if prior is not None and prior["ok"]:
                continue  # already scored on an earlier run
            try:
                rec = _run_question(
                    cfg,
                    sender,
                    receiver,
                    tok,
                    transfer,
                    item,
                    letter_ids,
                )
                rec["i"], rec["ok"] = i, True
            except Exception as exc:
                if cfg.on_error == "fail":
                    raise
                _free_memory(cfg.device)
                logger.warning(
                    "question %d failed (%s) -- skipping: %s",
                    i,
                    type(exc).__name__,
                    exc,
                )
                rec = {"i": i, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            records[i] = rec
            if ck is not None:
                ck.write(json.dumps(rec) + "\n")
                ck.flush()
            if rec["ok"]:
                n_done += 1
                for t in cache_treatments:
                    correct[t] += rec["correct"][t]
                if n_done % 25 == 0:
                    running = " ".join(
                        f"{t}={correct[t] / n_done:.3f}" for t in cache_treatments
                    )
                    print(f"  [{n_done}] {running}", flush=True)
    finally:
        if ck is not None:
            ck.close()

    return _aggregate(records, cache_treatments, n)


def _write_result(path: str, cfg: Settings, result: EvalResult) -> None:
    """Write the run config and per-treatment results to `path` as JSON."""
    payload = {
        "sender": cfg.sender,
        "receiver": cfg.receiver,
        "device": cfg.device,
        "sender_device": cfg.sender_device,
        "transfer": "direct",
        "benchmark": cfg.benchmark,
        "split": cfg.split,
        "num_examples": cfg.num_examples,
        "thetas": list(cfg.thetas),
        "score_t2t": cfg.score_t2t,
        "seed": cfg.seed,
        "on_error": cfg.on_error,
        "checkpoint": cfg.checkpoint,
        "n": result.n,
        "n_scored": result.n_scored,
        "failed": result.failed,
        "t2t_max_new_tokens": cfg.t2t_max_new_tokens,
        "accuracy": result.accuracy,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _check_device(device: str) -> None:
    """Validate `device` before multi-GB models are loaded onto it.

    Surfaces a clear, early RuntimeError (CUDA unavailable, or the index exceeds the
    visible devices) instead of a deep traceback after both models are partially placed.
    """
    dev = torch.device(device)
    if dev.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"device {device!r} requests CUDA, but torch.cuda.is_available() is False "
            f"-- pass --device cpu or run on a GPU host"
        )
    count = torch.cuda.device_count()
    index = dev.index or 0
    if index >= count:
        raise RuntimeError(
            f"device {device!r} requests CUDA index {index}, but only {count} CUDA "
            f"device(s) are visible"
        )


@torch.no_grad()
def main(argv: list[str] | None = None) -> None:
    """Parse settings, build the direct transfer, evaluate the benchmark, print results.

    Configures logging once here (the application entry point) so the diagnostic
    messages the modules emit via their loggers are shown; the accuracy table is written
    to stdout directly as the program's result.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    cfg = Settings.from_cli(argv)
    if cfg.seed is not None:
        set_seed(cfg.seed)
    _check_device(cfg.device)
    if cfg.sender_device:
        _check_device(cfg.sender_device)
    tok, sender, receiver = core.load_models(
        cfg.sender, cfg.receiver, cfg.device, cfg.sender_device
    )

    transfer = transfers.build_transfer(cfg, sender, receiver, tok)
    letter_ids = [tok(" " + L, add_special_tokens=False).input_ids[-1] for L in LETTERS]
    # Accuracy is the argmax over these ids, so a tokenizer that maps " A" to a token
    # that does not decode back to "A" would silently corrupt every score. Fail loudly
    # with a real exception (an `assert` would be stripped under `python -O`).
    for lid, L in zip(letter_ids, LETTERS, strict=True):
        decoded = tok.decode([lid]).strip()
        if decoded != L:
            raise ValueError(
                f"option letter {L!r} tokenizes to id {lid} which decodes to "
                f"{decoded!r}; option-letter scoring is invalid for this tokenizer "
                f"({cfg.receiver})"
            )
    data = load_mc(cfg.benchmark, cfg.split)
    if cfg.num_examples is not None:
        data = data[: cfg.num_examples]
    result = evaluate(cfg, sender, receiver, tok, transfer, data, letter_ids)

    receiver_slug, sender_slug = cfg.receiver.split("/")[-1], cfg.sender.split("/")[-1]
    print(
        f"\nCacheFuse [{cfg.benchmark}/{cfg.split}] transfer=direct "
        f"receiver={receiver_slug} sharer={sender_slug} "
        f"(scored {result.n_scored}/{result.n}):"
    )
    print("  treatment    accuracy")
    for treatment, acc in result.accuracy.items():
        print(f"  {treatment:>10} {acc:.3f}", flush=True)
    if result.failed:
        shown = result.failed[:20]
        logger.warning(
            "%d question(s) failed and were skipped: %s%s",
            len(result.failed),
            shown,
            " ..." if len(result.failed) > len(shown) else "",
        )
        if cfg.checkpoint:
            logger.warning(
                "rerun the same command with --checkpoint %s to retry the failures",
                cfg.checkpoint,
            )
        else:
            logger.warning(
                "pass --checkpoint PATH to record per-question results and retry "
                "failures on a rerun"
            )
    if cfg.out:
        _write_result(cfg.out, cfg, result)
        logger.info("wrote results JSON to %s", cfg.out)


if __name__ == "__main__":
    main()
