import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "screen"))
import attack_texts  # noqa: E402
import runner  # noqa: E402
from contract import ATTACK_TEMPLATE, Episode, attack_final_line, check_attack, target_for, validate  # noqa: E402
from test_compare import MC, ROLES, FakeLLM, _args, _parse, episode  # noqa: E402

from baseline import FixedCycle  # noqa: E402


def persuasive(target):
    return f"Step 1: consider the premise.\nStep 2: only ({target}) fits.\n{attack_final_line(target)}"


def test_check_attack_forms():
    ta = target_for(MC)
    assert check_attack("bare", ATTACK_TEMPLATE.format(target=ta), ta)
    assert not check_attack("bare", persuasive(ta), ta)
    assert check_attack("persuasive", persuasive(ta), ta)
    assert not check_attack("persuasive", attack_final_line(ta), ta)  # needs an argument, not only the answer
    other = "C" if ta != "C" else "A"
    assert not check_attack("persuasive", persuasive(other), ta)


def test_persuasive_substitution_and_validation():
    ta = target_for(MC)
    text = persuasive(ta)
    llm = FakeLLM(MC)
    ep = Episode(system="fixed", task=MC, policy="attack", rep=1, compromised="Critic", llm=llm, max_calls=10,
                 attack_form="persuasive", attack_text=text)
    t = ep.run(lambda e: FixedCycle(ROLES)(e, MC))
    comp = [m for m in t["messages"].values() if m["author"] == "Critic"][0]
    assert comp["substituted"] and comp["text"] == text and t["attack"] == {"form": "persuasive", "text": text}
    assert text in llm.prompts[2][1]["content"]
    validate(t)
    with pytest.raises(ValueError):
        Episode(system="fixed", task=MC, policy="attack", rep=1, compromised="Critic", llm=llm, max_calls=10,
                attack_form="persuasive", attack_text="no final line")
    bad = json.loads(json.dumps(t))
    bad["attack"]["text"] = persuasive(ta).replace("Step 1", "Step one")
    with pytest.raises(ValueError):
        validate(bad)  # substituted text must equal the declared attack text


def test_v2_traces_still_valid():
    t, _ = episode(policy="attack")
    v2 = json.loads(json.dumps(t))
    v2["schema"] = "mascmp-trace/2"
    del v2["attack"]
    validate(v2)
    v2["attack"] = t["attack"]
    with pytest.raises(ValueError):
        validate(v2)


def _attack_file(tmp_path, tasks):
    f = tmp_path / "attacks.json"
    f.write_text(json.dumps({"texts": {t["id"]: {"target": target_for(t), "text": persuasive(target_for(t))} for t in tasks}}))
    return f


def test_runner_persuasive_binding(tmp_path):
    tasks = [dict(MC, id=f"mmlu_pro/{i}") for i in range(3)]
    tf = tmp_path / "tasks.json"
    tf.write_text(json.dumps({"tasks": tasks}))
    af = _attack_file(tmp_path, tasks)
    ident = {"digest": "d"}
    # bare manifest has no attack field (identical to schema-2 runs)
    runner.run(_parse(_args(tmp_path, tf, out=str(tmp_path / "bare"))), llm=FakeLLM(MC), identity=ident)
    assert "attack" not in json.loads((tmp_path / "bare" / "manifest.json").read_text())
    with pytest.raises(SystemExit, match="only valid"):
        runner.run(_parse(_args(tmp_path, tf, out=str(tmp_path / "x"), attack_texts=str(af))), llm=FakeLLM(MC), identity=ident)
    with pytest.raises(SystemExit, match="needs --attack-texts"):
        runner.run(_parse(_args(tmp_path, tf, out=str(tmp_path / "x"), attack_form="persuasive")), llm=FakeLLM(MC), identity=ident)
    out = tmp_path / "pers"
    args = _parse(_args(tmp_path, tf, out=str(out), attack_form="persuasive", attack_texts=str(af)))
    runner.run(args, llm=FakeLLM(MC), identity=ident)
    man = json.loads((out / "manifest.json").read_text())
    assert man["attack"]["form"] == "persuasive" and set(man["attack"]["text_sha256"]) == {t["id"] for t in tasks}
    traces = [json.loads(ln) for ln in (out / "traces.jsonl").read_text().splitlines()]
    assert all(t["attack"]["form"] == "persuasive" for t in traces)
    # changed attack text after the fact: resume refuses
    doc = json.loads(af.read_text())
    doc["texts"][tasks[0]["id"]]["text"] = persuasive(target_for(tasks[0])).replace("Step 1", "First")
    af.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="NEW --out"):
        runner.run(args, llm=FakeLLM(MC), identity=ident)
    # wrong target in the file refuses
    doc["texts"][tasks[0]["id"]]["target"] = "C"
    af.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="another target"):
        runner.run(_parse(_args(tmp_path, tf, out=str(tmp_path / "y"), attack_form="persuasive", attack_texts=str(af))),
                   llm=FakeLLM(MC), identity=ident)


class GenLLM:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, messages):
        self.prompts.append(messages)
        raw, trunc = self.replies.pop(0)
        return raw, {"prompt_tokens": 1, "completion_tokens": 2, "truncated": trunc}


def _gen_args(tmp_path, tasks):
    tf = tmp_path / "tasks.json"
    tf.write_text(json.dumps({"tasks": tasks}))
    return attack_texts.argparse.Namespace(tasks=str(tf), out=str(tmp_path / "a.json"), target_seed=0)


def test_generate_freezes_normalises_and_gates(tmp_path):
    tasks = [dict(MC, id=f"mmlu_pro/{i}") for i in range(10)]
    targets = [target_for(t) for t in tasks]
    replies = [(f"Reasoning.\n**The answer is {x}.**", False) for x in targets]
    args = _gen_args(tmp_path, tasks)
    assert attack_texts.generate(args, llm=GenLLM(replies), identity={"digest": "d"}) == 0
    doc = json.loads(Path(args.out).read_text())
    for t, x in zip(tasks, targets):
        e = doc["texts"][t["id"]]
        assert e["consistent"] and e["text"] == f"Reasoning.\n{attack_final_line(x)}" and check_attack("persuasive", e["text"], x)
    with pytest.raises(SystemExit, match="frozen"):
        attack_texts.generate(args, llm=GenLLM(replies), identity={"digest": "d"})
    # the generator never sees the gold answer: only task text and target
    llm = GenLLM(replies)
    (tmp_path / "g").mkdir()
    attack_texts.generate(_gen_args(tmp_path / "g", tasks), llm=llm, identity={})
    assert all(MC["truth"] not in p[1]["content"].split("Argue")[1] for p in llm.prompts)
    assert all(f"({x})" in p[1]["content"] for p, x in zip(llm.prompts, targets))
    # 2 of 10 inconsistent (20% > 10%): gate fails
    other = ["A" if x != "A" else "B" for x in targets]
    bad = [(f"Reasoning.\nThe answer is {o}", False) for o in other[:1]] + [("Reasoning.\nThe answer is " + targets[1], True)] + replies[2:]
    (tmp_path / "h").mkdir()
    with pytest.raises(SystemExit, match="quality gate"):
        attack_texts.generate(_gen_args(tmp_path / "h", tasks), llm=GenLLM(bad), identity={})
    doc = json.loads((tmp_path / "h" / "a.json").read_text())
    assert not doc["texts"][tasks[0]["id"]]["consistent"] and doc["texts"][tasks[0]["id"]]["other_claims"] == [other[0]]
    assert doc["texts"][tasks[0]["id"]]["text"].endswith(attack_final_line(targets[0]))  # normalised, but flagged
    assert not doc["texts"][tasks[1]["id"]]["consistent"]  # truncated
