# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for evaluate helpers and orchestration (mocked models)."""

import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from cachefuse.benchmarks import MCQuestion
from cachefuse.config import Settings
from cachefuse.evaluate import (
    EvalResult,
    _check_device,
    _free_memory,
    _generate_hint,
    _layers,
    _letter_logits,
    _predict,
    _run_question,
    _t2t_cache,
    _write_result,
    evaluate,
    main,
)


def _item(gold=0):
    return MCQuestion(question="Q?", options=["a", "b"], gold=gold)


def _fake_cache(n_layers=1, seq=2, heads=1, dim=2):
    layers = []
    for _ in range(n_layers):
        layers.append(
            SimpleNamespace(
                keys=torch.zeros(1, heads, seq, dim),
                values=torch.zeros(1, heads, seq, dim),
                device=torch.device("cpu"),
            )
        )
    return SimpleNamespace(layers=layers)


def test_letter_logits_and_predict(monkeypatch):
    logits = torch.tensor([0.0, 5.0, 1.0, 2.0])
    model = MagicMock()
    model.return_value = SimpleNamespace(logits=logits.view(1, 1, -1))
    past = _fake_cache()
    out = _letter_logits(model, torch.zeros(1, 2, dtype=torch.long), [1, 3], past)
    assert out.tolist() == [5.0, 2.0]

    monkeypatch.setattr(
        "cachefuse.evaluate.core.make_cache",
        lambda keys, values, dev: past,
    )
    pred = _predict(
        model,
        torch.zeros(1, 2, dtype=torch.long),
        [torch.zeros(1, 1, 1, 1)],
        [torch.zeros(1, 1, 1, 1)],
        [1, 3],
        "cpu",
    )
    assert pred == 0


def test_layers_reads_keys_and_values():
    cache = _fake_cache(n_layers=2)
    keys, values = _layers(cache)
    assert len(keys) == 2 and len(values) == 2
    assert keys[0] is cache.layers[0].keys


def test_generate_hint_with_and_without_chat_template():
    item = _item()

    # With chat template.
    tok = MagicMock()
    tok.chat_template = "tmpl"
    tok.apply_chat_template.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    tok.pad_token_id = 0
    tok.decode.return_value = "  hint text  "
    sender = MagicMock()
    sender.generate.return_value = torch.tensor([[1, 2, 3, 9, 9]])
    assert _generate_hint(sender, tok, item, "cpu", 8) == "hint text"
    assert sender.generate.call_args.kwargs["pad_token_id"] == 0

    # Without chat template; pad falls back to eos.
    tok2 = MagicMock()
    tok2.chat_template = None
    tok2.return_value = {
        "input_ids": torch.tensor([[4, 5]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }
    tok2.pad_token_id = None
    tok2.eos_token_id = 7
    tok2.decode.return_value = "other"
    sender2 = MagicMock()
    sender2.generate.return_value = torch.tensor([[4, 5, 8]])
    assert _generate_hint(sender2, tok2, item, "cpu", 4) == "other"
    assert sender2.generate.call_args.kwargs["pad_token_id"] == 7


def test_free_memory_cpu_and_cuda(monkeypatch):
    emptied = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(True))
    _free_memory("cpu")
    assert emptied == []
    _free_memory("cuda:0")
    assert emptied == [True]


def test_t2t_cache_prefills_hint_prompt(monkeypatch):
    cfg = Settings(device="cpu", t2t_max_new_tokens=8)
    monkeypatch.setattr(
        "cachefuse.evaluate._generate_hint", lambda *a, **k: "background"
    )
    tok = MagicMock()
    tok.return_value = SimpleNamespace(input_ids=torch.tensor([[1, 2, 3, 4]]))
    cache = _fake_cache()
    monkeypatch.setattr("cachefuse.evaluate.core.prefill", lambda model, ids: cache)
    keys, values, ids = _t2t_cache(
        cfg, MagicMock(), MagicMock(), tok, _item(), "cpu", "cpu"
    )
    assert ids.shape[1] == 4
    assert len(keys) == 1 and len(values) == 1


def test_run_question_scores_treatments_with_and_without_t2t(monkeypatch):
    cfg = Settings(device="cpu", thetas=(0.5,), score_t2t=True, sender_device="cpu")
    item = _item(gold=0)
    tok = MagicMock()
    tok.return_value = SimpleNamespace(input_ids=torch.tensor([[10, 11, 12]]))
    cache = _fake_cache()
    monkeypatch.setattr("cachefuse.evaluate.core.prefill", lambda model, ids: cache)
    monkeypatch.setattr(
        "cachefuse.evaluate._predict",
        lambda model, ids, keys, values, letter_ids, dev: 0,
    )
    monkeypatch.setattr(
        "cachefuse.evaluate._t2t_cache",
        lambda *a, **k: (
            [torch.zeros(1, 1, 1, 1)],
            [torch.zeros(1, 1, 1, 1)],
            torch.tensor([[1, 2, 3]]),
        ),
    )

    def transfer(ids):
        return cache

    rec = _run_question(
        cfg,
        MagicMock(),
        MagicMock(),
        tok,
        transfer,
        item,
        [1, 2],
    )
    assert rec["correct"]["receiver"] == 1
    assert rec["correct"]["theta0.5"] == 1
    assert rec["correct"]["t2t"] == 1

    cfg2 = Settings(device="cpu", thetas=(0.5,), score_t2t=False)
    rec2 = _run_question(
        cfg2,
        MagicMock(),
        MagicMock(),
        tok,
        transfer,
        item,
        [1, 2],
    )
    assert "t2t" not in rec2["correct"]
    assert rec2["correct"]["replace"] == 1


def test_evaluate_empty_data_raises():
    with pytest.raises(ValueError, match="no questions"):
        evaluate(
            Settings(device="cpu"),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            lambda x: x,
            [],
            [1, 2],
        )


def test_evaluate_resumes_skips_failures_and_prints(monkeypatch, tmp_path, capsys):
    path = str(tmp_path / "ck.jsonl")
    cfg = Settings(
        device="cpu",
        thetas=(0.5,),
        checkpoint=path,
        on_error="skip",
        score_t2t=False,
    )
    # Seed a prior ok record for index 0 so evaluate resumes over it.
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "config": {
                        "sender": cfg.sender,
                        "receiver": cfg.receiver,
                        "benchmark": cfg.benchmark,
                        "split": cfg.split,
                        "num_examples": cfg.num_examples,
                        "thetas": list(cfg.thetas),
                        "score_t2t": cfg.score_t2t,
                    }
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "i": 0,
                    "ok": True,
                    "correct": {
                        "receiver": 1,
                        "theta0.5": 1,
                        "replace": 0,
                        "sharer": 1,
                    },
                }
            )
            + "\n"
        )

    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        # Fail the first new question, succeed thereafter.
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return {
            "correct": {
                "receiver": 1,
                "theta0.5": 0,
                "replace": 1,
                "sharer": 1,
            },
        }

    monkeypatch.setattr("cachefuse.evaluate._run_question", fake_run)
    monkeypatch.setattr("cachefuse.evaluate._free_memory", lambda dev: None)

    # 26 questions so the 25th success prints a running line (1 resumed + 24 new ok
    # after one failure -> need enough successes). With index 0 already ok, indices
    # 1..25 run: first fails, next 24 succeed => n_done reaches 25.
    data = [_item() for _ in range(26)]
    result = evaluate(
        cfg, MagicMock(), MagicMock(), MagicMock(), lambda x: x, data, [1, 2]
    )
    assert 1 in result.failed
    assert result.n_scored == 25
    out = capsys.readouterr().out
    assert "[25]" in out


def test_evaluate_on_error_fail_reraises(monkeypatch):
    cfg = Settings(device="cpu", on_error="fail", thetas=(0.5,))
    monkeypatch.setattr(
        "cachefuse.evaluate._run_question",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")),
    )
    with pytest.raises(RuntimeError, match="x"):
        evaluate(
            cfg,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            lambda x: x,
            [_item()],
            [1, 2],
        )


def test_evaluate_score_t2t_and_no_checkpoint(monkeypatch):
    cfg = Settings(device="cpu", thetas=(0.5,), score_t2t=True)
    monkeypatch.setattr(
        "cachefuse.evaluate._run_question",
        lambda *a, **k: {
            "correct": {
                "receiver": 1,
                "theta0.5": 1,
                "replace": 1,
                "sharer": 1,
                "t2t": 0,
            },
        },
    )
    result = evaluate(
        cfg,
        MagicMock(),
        MagicMock(),
        MagicMock(),
        lambda x: x,
        [_item()],
        [1, 2],
    )
    assert "t2t" in result.accuracy
    assert result.accuracy["t2t"] == 0.0


def test_evaluate_resume_retries_failed_prior(monkeypatch, tmp_path):
    path = str(tmp_path / "ck.jsonl")
    cfg = Settings(
        device="cpu",
        thetas=(0.5,),
        checkpoint=path,
        on_error="skip",
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "config": {
                        "sender": cfg.sender,
                        "receiver": cfg.receiver,
                        "benchmark": cfg.benchmark,
                        "split": cfg.split,
                        "num_examples": cfg.num_examples,
                        "thetas": list(cfg.thetas),
                        "score_t2t": cfg.score_t2t,
                    }
                }
            )
            + "\n"
        )
        f.write(json.dumps({"i": 0, "ok": False, "error": "old"}) + "\n")

    monkeypatch.setattr(
        "cachefuse.evaluate._run_question",
        lambda *a, **k: {
            "correct": {
                "receiver": 1,
                "theta0.5": 1,
                "replace": 1,
                "sharer": 1,
            },
        },
    )
    result = evaluate(
        cfg, MagicMock(), MagicMock(), MagicMock(), lambda x: x, [_item()], [1, 2]
    )
    assert result.n_scored == 1
    assert result.failed == []


def test_write_result_creates_parent(tmp_path):
    out = tmp_path / "nested" / "result.json"
    cfg = Settings(device="cpu", out=str(out))
    result = EvalResult(
        accuracy={"receiver": 1.0},
        n=1,
        n_scored=1,
        failed=[],
    )
    _write_result(str(out), cfg, result)
    payload = json.loads(out.read_text())
    assert payload["accuracy"]["receiver"] == 1.0
    assert payload["n"] == 1


def test_check_device_cpu_ok_and_cuda_errors(monkeypatch):
    _check_device("cpu")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        _check_device("cuda:0")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(RuntimeError, match="only 1 CUDA"):
        _check_device("cuda:3")

    # index None treated as 0; available.
    _check_device("cuda")


def test_load_checkpoint_blank_file_and_blank_lines(tmp_path):
    from cachefuse.evaluate import _load_checkpoint

    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert _load_checkpoint(str(empty), Settings()) == {}

    path = tmp_path / "run.jsonl"
    cfg = Settings(checkpoint=str(path), thetas=(0.5,))
    from cachefuse.evaluate import _checkpoint_config, _open_checkpoint

    ck = _open_checkpoint(cfg)
    ck.write("\n")
    ck.write(json.dumps({"i": 0, "ok": True, "correct": {"receiver": 1}}) + "\n")
    ck.close()
    # reopen append path already exists (not fresh)
    ck2 = _open_checkpoint(cfg)
    assert ck2 is not None
    ck2.close()
    assert _load_checkpoint(str(path), cfg)[0]["ok"] is True
    assert _checkpoint_config(cfg)["thetas"] == [0.5]


def test_open_checkpoint_flat_filename(tmp_path, monkeypatch):
    from cachefuse.evaluate import _open_checkpoint

    monkeypatch.chdir(tmp_path)
    cfg = Settings(checkpoint="flat.jsonl")
    ck = _open_checkpoint(cfg)
    ck.close()
    assert (tmp_path / "flat.jsonl").exists()


def test_main_happy_path_with_failures_and_outputs(
    monkeypatch, tmp_path, capsys, caplog
):
    out = tmp_path / "out.json"
    argv = [
        "--device",
        "cpu",
        "--theta",
        "0.5",
        "--seed",
        "7",
        "--out",
        str(out),
        "--checkpoint",
        str(tmp_path / "ck.jsonl"),
        "--score-t2t",
    ]

    tok = MagicMock()

    # " A" etc. tokenize to ids that decode back to letters.
    def tok_call(text, add_special_tokens=False):
        letter = text.strip()
        return SimpleNamespace(input_ids=[ord(letter)])

    tok.side_effect = tok_call
    tok.decode.side_effect = lambda ids: chr(ids[0])

    sender = MagicMock()
    receiver = MagicMock()
    monkeypatch.setattr(
        "cachefuse.evaluate.core.load_models",
        lambda *a, **k: (tok, sender, receiver),
    )
    monkeypatch.setattr(
        "cachefuse.evaluate.transfers.build_transfer",
        lambda *a, **k: lambda ids: _fake_cache(),
    )
    monkeypatch.setattr(
        "cachefuse.evaluate.load_mc",
        lambda bench, split: [_item(), _item()],
    )
    monkeypatch.setattr("cachefuse.evaluate.set_seed", lambda s: None)
    monkeypatch.setattr("cachefuse.evaluate._check_device", lambda d: None)

    result = EvalResult(
        accuracy={"receiver": 0.5, "t2t": 0.5},
        n=2,
        n_scored=1,
        failed=[1] + list(range(2, 25)),  # >20 to hit truncation branch
    )
    monkeypatch.setattr("cachefuse.evaluate.evaluate", lambda *a, **k: result)

    with caplog.at_level(logging.WARNING):
        main(argv)
    printed = capsys.readouterr().out
    assert "CacheFuse" in printed
    assert "receiver" in printed
    assert "0.500" in printed
    assert out.exists()
    assert "failed and were skipped" in caplog.text
    assert "rerun the same command with --checkpoint" in caplog.text


def test_main_failed_without_checkpoint_and_letter_mismatch(monkeypatch, caplog):
    tok = MagicMock()
    tok.side_effect = lambda text, add_special_tokens=False: SimpleNamespace(
        input_ids=[1]
    )
    tok.decode.return_value = "Z"  # mismatch
    monkeypatch.setattr(
        "cachefuse.evaluate.core.load_models",
        lambda *a, **k: (tok, MagicMock(), MagicMock()),
    )
    monkeypatch.setattr(
        "cachefuse.evaluate.transfers.build_transfer", lambda *a, **k: MagicMock()
    )
    monkeypatch.setattr("cachefuse.evaluate._check_device", lambda d: None)
    monkeypatch.setattr("cachefuse.evaluate.set_seed", lambda s: None)
    with pytest.raises(ValueError, match="option letter"):
        main(["--device", "cpu", "--theta", "0.5"])

    # Failure warning without checkpoint.
    tok2 = MagicMock()
    tok2.side_effect = lambda text, add_special_tokens=False: SimpleNamespace(
        input_ids=[ord(text.strip())]
    )
    tok2.decode.side_effect = lambda ids: chr(ids[0])
    monkeypatch.setattr(
        "cachefuse.evaluate.core.load_models",
        lambda *a, **k: (tok2, MagicMock(), MagicMock()),
    )
    monkeypatch.setattr("cachefuse.evaluate.load_mc", lambda bench, split: [_item()])
    monkeypatch.setattr(
        "cachefuse.evaluate.evaluate",
        lambda *a, **k: EvalResult(
            accuracy={"receiver": 0.0},
            n=1,
            n_scored=0,
            failed=[0],
        ),
    )
    with caplog.at_level(logging.WARNING):
        main(["--device", "cpu", "--theta", "0.5", "--seed", "0"])
    assert "pass --checkpoint PATH" in caplog.text


def test_main_num_examples_truncates(monkeypatch, capsys):
    tok = MagicMock()
    tok.side_effect = lambda text, add_special_tokens=False: SimpleNamespace(
        input_ids=[ord(text.strip())]
    )
    tok.decode.side_effect = lambda ids: chr(ids[0])
    monkeypatch.setattr(
        "cachefuse.evaluate.core.load_models",
        lambda *a, **k: (tok, MagicMock(), MagicMock()),
    )
    monkeypatch.setattr(
        "cachefuse.evaluate.transfers.build_transfer", lambda *a, **k: MagicMock()
    )
    monkeypatch.setattr("cachefuse.evaluate._check_device", lambda d: None)
    monkeypatch.setattr("cachefuse.evaluate.set_seed", lambda s: None)
    monkeypatch.setattr(
        "cachefuse.evaluate.load_mc",
        lambda bench, split: [_item() for _ in range(5)],
    )
    seen = {}

    def fake_eval2(cfg, sender, receiver, tok, transfer, data, letter_ids):
        seen["n"] = len(data)
        return EvalResult(
            {"receiver": 1.0},
            n=len(data),
            n_scored=len(data),
            failed=[],
        )

    monkeypatch.setattr("cachefuse.evaluate.evaluate", fake_eval2)
    main(
        [
            "--device",
            "cpu",
            "--theta",
            "0.5",
            "--num-examples",
            "2",
            "--sender-device",
            "cpu",
        ]
    )
    assert seen["n"] == 2
    assert "CacheFuse" in capsys.readouterr().out


def test_main_skips_set_seed_when_none(monkeypatch, capsys):
    tok = MagicMock()
    tok.side_effect = lambda text, add_special_tokens=False: SimpleNamespace(
        input_ids=[ord(text.strip())]
    )
    tok.decode.side_effect = lambda ids: chr(ids[0])
    seeded = []
    monkeypatch.setattr("cachefuse.evaluate.set_seed", lambda s: seeded.append(s))
    monkeypatch.setattr(
        "cachefuse.evaluate.Settings.from_cli",
        lambda argv=None: Settings(device="cpu", thetas=(0.5,), seed=None),
    )
    monkeypatch.setattr(
        "cachefuse.evaluate.core.load_models",
        lambda *a, **k: (tok, MagicMock(), MagicMock()),
    )
    monkeypatch.setattr(
        "cachefuse.evaluate.transfers.build_transfer", lambda *a, **k: MagicMock()
    )
    monkeypatch.setattr("cachefuse.evaluate._check_device", lambda d: None)
    monkeypatch.setattr("cachefuse.evaluate.load_mc", lambda bench, split: [_item()])
    monkeypatch.setattr(
        "cachefuse.evaluate.evaluate",
        lambda *a, **k: EvalResult({"receiver": 1.0}, n=1, n_scored=1, failed=[]),
    )
    main([])
    assert seeded == []
    assert "CacheFuse" in capsys.readouterr().out
