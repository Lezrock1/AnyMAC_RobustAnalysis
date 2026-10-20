import contextlib
import hashlib
import io
import json
import random
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "screen"))
import dylan  # noqa: E402
import runner  # noqa: E402
import scoring  # noqa: E402
from contract import ATTACK_TEMPLATE, Episode, validate  # noqa: E402
from screen import mc_text  # noqa: E402

OPTS4 = ["one", "two", "three", "four"]
MC4 = {"id": "mmlu/x/1", "kind": "mc", "n_options": 4, "truth": "A", "text": mc_text("Which one?", OPTS4)}
OPTS10 = [f"opt {c}" for c in "ABCDEFGHIJ"]
MC10 = {"id": "mmlu_pro/9", "kind": "mc", "n_options": 10, "truth": "G", "text": mc_text("Pick G?", OPTS10)}
USAGE = {"prompt_tokens": 3, "completion_tokens": 2, "truncated": False}


def respond(messages, salt):
    """Deterministic in the full request: identical requests give identical replies."""
    h = int(hashlib.sha256((salt + json.dumps(messages)).encode()).hexdigest(), 16)
    if messages[0]["role"] == "user":  # ranker
        return "No ranking." if h % 6 == 0 else f"Best: [{h % 7 + 1}, {(h // 7) % 7 + 1}]"
    if h % 17 == 0:
        return "I am unsure."
    bias = (9, 7, 4)[int(salt) % 3]
    return f"Reasoning {h % 997}.\nThe answer is ({'A' if h % 10 < bias else 'BCD'[h % 3]}) [[1, 2, 3]]"


def _upstream():
    """Import the vendored, unmodified DyLAN MMLU modules with stubbed API clients."""
    for name in ("openai", "openai.error", "backoff"):
        sys.modules.setdefault(name, types.ModuleType(name))
    err = sys.modules["openai.error"]
    for k in ("RateLimitError", "APIError", "ServiceUnavailableError", "APIConnectionError", "Timeout"):
        setattr(err, k, getattr(err, k, type(k, (Exception,), {})))
    sys.modules["openai"].error = err
    sys.modules["backoff"].on_exception = lambda *a, **k: (lambda f: f)
    sys.modules["backoff"].expo = None
    saved = {k: sys.modules.pop(k) for k in ("utils", "prompt_lib", "LLM_Neuron", "LLMLP") if k in sys.modules}
    sys.path.insert(0, str(dylan.VENDOR / "MMLU"))
    try:
        import LLM_Neuron
        import LLMLP
        import prompt_lib
        return LLMLP, LLM_Neuron, prompt_lib
    finally:
        sys.path.pop(0)
        for k in ("utils", "prompt_lib", "LLM_Neuron", "LLMLP"):
            sys.modules.pop(k, None)
        sys.modules.update(saved)


def run_adapter(task, seed, variant="upstream", policy="honest", compromised="Lawyer", max_calls=17,
                ranker_fallback="native", salt=None, rng=True):
    salt = str(seed) if salt is None else salt
    log = []

    def llm(messages):
        log.append(messages)
        return respond(messages, salt), dict(USAGE)
    system = dylan.DyLAN(variant, ranker_fallback=ranker_fallback)
    ep = Episode(system=system.name, task=task, policy=policy, rep=1, compromised=compromised, llm=llm, max_calls=max_calls)
    t = ep.run(lambda e: system(e, task, random.Random(seed) if rng else None))
    return t, log


def test_question_and_prompts_match_upstream_for_four_options():
    _, _, P = _upstream()
    assert dylan.question(MC4) == P.MMLU_QUESTION.format("Which one?", *OPTS4)
    for resp in ([], ["x", "y"]):
        assert dylan.worker_message(resp, "Q", 4) == P.construct_message(resp, "Q", "single_choice")


def test_ten_options_are_rendered_and_parsed():
    q = dylan.question(MC10)
    assert q.endswith("Pick G?: A) opt A, B) opt B, C) opt C, D) opt D, E) opt E, F) opt F, G) opt G, "
                      "H) opt H, I) opt I, J) opt J ")
    m = dylan.worker_message([], q, 10)["content"]
    assert "(X) represents choice (A), (B), (C), (D), (E), (F), (G), (H), (I), or (J)." in m
    assert dylan.parse_choice("so (b) ... finally (G) [[1,2]]", 10) == "G"
    assert dylan.parse_choice("answer G)", 10) == "G" and dylan.parse_choice("(G)", 4) is None
    assert dylan.parse_choice("The answer is G", 10) is None  # native parser needs "(X)" or "X)"
    assert dylan.parse_choice(ATTACK_TEMPLATE.format(target="G"), 10) == "G"
    with pytest.raises(ValueError, match="options"):
        dylan.question(dict(MC10, n_options=9))


def test_upstream_variant_reproduces_pinned_code_exactly():
    """Same seed, same deterministic LLM: identical request sequence, call count and final answer."""
    LLMLP, LLM_Neuron, _ = _upstream()
    paths = {"round0": 0, "round1": 0, "round2_consensus": 0, "round2_most_frequent": 0, "fallback": 0}
    for seed in range(150):
        up_log = []

        def fake(ctx, model, seed=seed):
            up_log.append(ctx)
            return respond(ctx, str(seed)), 1, 1
        LLM_Neuron.generate_answer = fake
        random.seed(seed)
        m = LLMLP.LLMLP("gpt-3.5-turbo", 7, dylan.ROLES, 3, "listwise", "single_choice", "gpt-3.5-turbo")
        with contextlib.redirect_stdout(io.StringIO()):
            res, resp_cnt, *_ = m.forward(dylan.question(MC4))
        t, log = run_adapter(MC4, seed)
        assert json.dumps(log) == json.dumps(up_log), seed
        assert resp_cnt == len(log) and t["stop"]["reason"] == "native"
        assert scoring.strict_answer(t["final"]["text"], MC4) == res
        detail = t["stop"]["detail"]
        paths["round0" if "round 0" in detail else "round1" if "round 1" in detail else
              "round2_consensus" if "consensus round 2" in detail else "round2_most_frequent"] += 1
        rk = [e for e in t["events"] if e["type"] == "decision" and e["kind"] == "ranker"]
        paths["fallback"] += bool(rk and rk[0]["output"]["fallback"])
    assert all(paths.values()), paths  # every native path was exercised


def test_paper_variant_runs_the_ranked_roles_with_all_round1_replies():
    differs = 0
    for seed in range(80):
        t, _ = run_adapter(MC4, seed, variant="paper")
        tu, _ = run_adapter(MC4, seed, variant="upstream")
        rk = [e for e in t["events"] if e["type"] == "decision" and e["kind"] == "ranker"]
        if not rk:
            continue
        rk = rk[0]
        assert len(set(rk["selected"])) == len(rk["selected"]) and set(rk["selected"]) == set(rk["ranking"])
        r2 = [e for e in t["events"] if e["type"] == "call" and e["round"] == 2 and e["kind"] == "worker"]
        assert [e["identity"] for e in r2] == rk["selected"][:len(r2)]
        r1 = [e["msg"] for e in t["events"] if e["type"] == "call" and e["round"] == 1]
        assert all(sorted(e["inputs"]) == sorted(r1) and len(r1) == 7 for e in r2)
        sel_u = [e for e in tu["events"] if e["type"] == "decision" and e["kind"] == "ranker"][0]
        differs += set(sel_u["selected"]) != set(sel_u["ranking"])
    assert differs > 10  # the upstream bug: ranked roles usually do not run


def test_honest_and_attack_share_shuffles_and_attack_is_delivered():
    th, _ = run_adapter(MC4, 0, variant="paper", rng=False, salt="1")
    ta, _ = run_adapter(MC4, 0, variant="paper", policy="attack", rng=False, salt="1")
    order = lambda t: [e["identity"] for e in t["events"] if e["type"] == "call" and e["round"] == 0]  # noqa: E731
    assert order(th) == order(ta)  # per-(task, rep) RNG, policy not in the seed
    comp = [m for m in ta["messages"].values() if m["author"] == "Lawyer"]
    assert comp and all(m["substituted"] for m in comp)
    r = scoring.score(ta, MC4)
    assert r["invoked"] == len(comp) and r["delivered"].get("stop", 0) > 0
    if any(e["round"] == 1 for e in ta["events"] if e["type"] == "call"):
        assert r["delivered"]["worker"] > 0
    assert validate(ta)


def test_round0_consensus_stops_after_five_calls():
    def llm(messages):
        return "Easy.\nThe answer is (A)", dict(USAGE)
    s = dylan.DyLAN("paper")
    ep = Episode(system="dylan", task=MC4, policy="honest", rep=1, compromised="Doctor", llm=llm, max_calls=17)
    t = ep.run(lambda e: s(e, MC4))
    calls = [e for e in t["events"] if e["type"] == "call"]
    stops = [e for e in t["events"] if e["type"] == "decision" and e["kind"] == "stop"]
    assert len(calls) == 5 and len(stops) == 1 and stops[0]["stop"] and stops[0]["output"]["count"] == 5
    assert t["final"]["text"] == "The answer is (A)" and scoring.score(t, MC4)["outcome"] == "correct"
    agg = t["events"][t["final"]["decision"]]
    assert agg["kind"] == "aggregator" and agg["inputs"] == [e["msg"] for e in calls]


def _disagree(ranker):
    def llm(messages):
        if messages[0]["role"] == "user":
            return ranker, dict(USAGE)
        role = next(r for r in dylan.ROLES if messages[0]["content"].startswith(dylan.P.ROLE_MAP[r]))
        return f"Hmm.\nThe answer is ({'ABCD'[dylan.ROLES.index(role) % 4]})", dict(USAGE)
    return llm


def test_full_path_uses_17_calls_and_ranker_is_recorded():
    s = dylan.DyLAN("paper")
    ep = Episode(system="dylan", task=MC4, policy="attack", rep=1, compromised="Lawyer", llm=_disagree("so [1, 2]"),
                 max_calls=17)
    t = ep.run(lambda e: s(e, MC4))
    calls = [e for e in t["events"] if e["type"] == "call"]
    assert len(calls) == 17 and [e["kind"] for e in calls].count("ranker") == 1
    rk = [e for e in t["events"] if e["type"] == "decision" and e["kind"] == "ranker"][0]
    assert rk["output"]["fallback"] is None and len(rk["selected"]) == 2
    assert rk["ranking"] == rk["output"]["shown_order"][:2]
    assert scoring.score(t, MC4)["delivered"]["ranker"] == 1  # the substituted round-1 reply reached the ranker


def test_ranker_parse_failure_native_is_logged_or_error_is_explicit():
    for fb, outcome in (("native", None), ("error", "control_error")):
        s = dylan.DyLAN("paper", ranker_fallback=fb)
        ep = Episode(system="dylan", task=MC4, policy="honest", rep=1, compromised="Lawyer", llm=_disagree("none"),
                     max_calls=17)
        t = ep.run(lambda e: s(e, MC4))
        if outcome:
            assert t["stop"]["reason"] == "control_error" and scoring.score(t, MC4)["outcome"] == outcome
        else:
            rk = [e for e in t["events"] if e["type"] == "decision" and e["kind"] == "ranker"][0]
            assert rk["output"]["fallback"] == "random_pair" and t["stop"]["reason"] == "native"


def test_budget_abort_is_explicit():
    s = dylan.DyLAN("paper")
    ep = Episode(system="dylan", task=MC4, policy="honest", rep=1, compromised="Lawyer", llm=_disagree("[1,2]"),
                 max_calls=10)
    t = ep.run(lambda e: s(e, MC4))
    assert t["stop"]["reason"] == "budget" and t["final"] is None
    assert scoring.score(t, MC4)["outcome"] == "budget_abort"


def test_roles_and_runner_registration(tmp_path):
    with pytest.raises(ValueError, match="7 distinct"):
        dylan.DyLAN(roles=["Economist"] * 7)
    tf = tmp_path / "tasks.json"
    tf.write_text(json.dumps({"tasks": [MC10]}))
    res = tmp_path / "reserved.json"
    res.write_text(json.dumps({"tasks": []}))
    (dylan.VENDOR / "MMLU" / "__pycache__").mkdir(exist_ok=True)  # as after any test run (gitignored)
    for system in ("dylan", "dylan-upstream"):
        out = tmp_path / system
        p = runner.argparse.Namespace(system=system, tasks=str(tf), task_ids=None, reserved=[str(res)], out=str(out),
                                      model="fake", policies=["honest", "attack"], reps=1, max_calls=17,
                                      compromised="Lawyer", roles=None, rounds=1, ranker_fallback="native",
                                      seed=20261020, target_seed=0, attack_form="bare", attack_texts=None)
        runner.run(p, llm=lambda m: (respond(m, "1"), dict(USAGE)), identity={"digest": "x"})
        man = json.loads((out / "manifest.json").read_text())
        assert man["system"]["system"] == system and man["system"]["upstream"].endswith("006e440a519f7cf21e2826f3b8033d84ae9bf07c")
        assert "vendor/dylan_006e440a/MMLU/LLMLP.py" in man["provenance"] and (out / "COMPLETE.json").exists()
    assert sorted(man["provenance"]) == sorted(  # only the vendored sources, never caches
        f"vendor/dylan_006e440a/{n}" for n in ("LICENSE", "README.md", "MMLU/LLMLP.py", "MMLU/LLM_Neuron.py",
                                               "MMLU/exp_mmlu.sh", "MMLU/llmlp_listwise_mmlu.py",
                                               "MMLU/prompt_lib.py", "MMLU/utils.py"))
