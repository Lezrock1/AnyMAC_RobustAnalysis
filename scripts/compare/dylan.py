"""DyLAN (Liu et al., COLM 2024) adapter for the common contract, MMLU path of SALT-NLP/DyLAN@006e440a.

Native inference (vendor/dylan_006e440a/MMLU/LLMLP.py, `LLMLP(..., 7 roles, rounds=3, "listwise")`):
  round 0  the 7 roles answer in shuffled order without inputs; from the 5th answer on, stop if more than
           floor(2/3*7)=4 parsed answers agree (`check_consensus`, None answers can agree too).
  round 1  every role receives all round-0 replies (shuffled per agent) and answers again; same stop rule.
  round 2  an LLM ranker (user-only prompt) sees the shuffled round-1 replies and names the best two
           `[i,j]`; the selected roles answer again with all round-1 replies; stop if both agree, otherwise
           `most_frequent` of the round-2 answers (ties -> first in activation order).
At most 7 + 7 + 1 + 2 = 17 calls. Prompts, role texts and control constants are taken verbatim from the
vendored upstream files; only the documented adaptations below differ.

Variants (separate system labels, never pooled):
  `dylan`           paper semantics (Sec. 3.3.2, Eq. 7): the two ranked roles run in round 2.
  `dylan-upstream`  the pinned code as-is. `LLMLP.forward` compares the shuffled loop position with the
                    selected role indices (`if idx in idx_mask`), so the roles that run in round 2 are
                    effectively random, not the ranked ones. Reproduced exactly (see test_dylan.py).
Adaptations (both variants):
  - options A-J: question text and the "(X) represents choice ..." sentence list the task's options, and
    the answer parser accepts the same letters (upstream hard-codes 4 options / [ABCD]);
  - decoding follows the common transport (temperature 0, common token limit) instead of 1.0/2048;
  - shuffles use a per-(task, repetition) RNG instead of one global `random.seed(0)` per dataset, so the
    honest and attack run of a pair start from identical shuffles (the policy never enters the seed);
  - a ranker reply without `[i,j]` falls back to the native random pair (`ranker_fallback="native"`) and is
    recorded in the ranker decision, or becomes a ControlError (`ranker_fallback="error"`); never silent;
  - the programmatic final answer is written as "The answer is (X)" for the common strict scorer.
"""
from __future__ import annotations

import importlib.util
import math
import random
import re
from pathlib import Path

from contract import ControlError

UPSTREAM = "SALT-NLP/DyLAN@006e440a519f7cf21e2826f3b8033d84ae9bf07c"
VENDOR = Path(__file__).resolve().parent / "vendor" / "dylan_006e440a"
LETTERS = "ABCDEFGHIJ"
ROLES = ["Economist", "Doctor", "Lawyer", "Mathematician", "Psychologist", "Programmer", "Historian"]  # exp_mmlu.sh
ROUNDS = 3
RANKER = "Ranker"
CONSENSUS = "Consensus"


def _load_prompt_lib():
    spec = importlib.util.spec_from_file_location("dylan_prompt_lib_006e440a", VENDOR / "MMLU" / "prompt_lib.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load_prompt_lib()
_CHOICE_4 = "(X) represents choice (A), (B), (C), or (D)."
_QUESTION_PREFIX, _QUESTION_4 = P.MMLU_QUESTION.split("{}", 1)[0], "{}: A) {}, B) {}, C) {}, D) {} "
assert P.MMLU_QUESTION == _QUESTION_PREFIX + _QUESTION_4, "vendored MMLU_QUESTION changed"


def split_options(task):
    """Recover (question, options) from the common `mc_text` format 'question\\n\\nA. ..\\nB. ..'."""
    n = task["n_options"]
    head, sep, block = task["text"].rpartition("\n\nA. ")
    parts = re.split(r"\n(?=[B-J]\. )", "A. " + block) if sep else []
    if len(parts) != n or any(not p.startswith(f"{LETTERS[i]}. ") for i, p in enumerate(parts)):
        raise ValueError(f"task {task['id']}: cannot recover {n} options from the task text")
    return head, [p[3:] for p in parts]


def question(task):
    """Generalisation of upstream `parse_question_answer` (MMLU_QUESTION) to n options."""
    q, opts = split_options(task)
    return _QUESTION_PREFIX + q + ": " + ", ".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(opts)) + " "


def _choice_sentence(n):
    letters = [f"({c})" for c in LETTERS[:n]]
    return f"(X) represents choice {', '.join(letters[:-1])}, or {letters[-1]}."


def worker_message(responses, q, n):
    msg = P.construct_message(responses, q, "single_choice")
    if _CHOICE_4 not in msg["content"]:
        raise AssertionError("vendored construct_message changed")
    return {"role": "user", "content": msg["content"].replace(_CHOICE_4, _choice_sentence(n))}


def parse_choice(reply, n):
    """Upstream `parse_single_choice` with the letter range of the task: last '(X)', else last 'X)'."""
    letters = LETTERS[:n] + LETTERS[:n].lower()
    for pattern in (rf"\(([{letters}])\)", rf"([{letters}])\)"):
        matches = re.findall(pattern, reply or "")
        if matches:
            return matches[-1].upper()
    return None


def most_frequent(clist):
    """Upstream `most_frequent` with equality (single_choice)."""
    counter, num = 0, clist[0]
    for i in clist:
        f = sum(i == item for item in clist)
        if f > counter:
            counter, num = f, i
    return num, counter


def parse_ranks(completion, max_num):
    """Upstream `parse_ranks` without the silent fallback: None if no `[i,j]` pair is found."""
    matches = re.findall(r'\[([1234567]),\s*([1234567])\]', completion or "")
    if not matches:
        return None
    return [min(max(int(x) - 1, 0), max_num - 1) for x in matches[-1]]


class DyLAN:
    VARIANTS = ("paper", "upstream")

    def __init__(self, variant="paper", roles=None, ranker_fallback="native", seed=20261020):
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}")
        if ranker_fallback not in ("native", "error"):
            raise ValueError("ranker_fallback must be 'native' or 'error'")
        self.roles = list(roles or ROLES)
        if len(self.roles) != 7 or len(set(self.roles)) != 7 or not set(self.roles) <= set(P.ROLE_MAP):
            raise ValueError(f"DyLAN MMLU uses 7 distinct roles from ROLE_MAP (ranker regex is [1-7]); got {self.roles}")
        self.variant, self.ranker_fallback, self.seed = variant, ranker_fallback, seed
        self.name = "dylan" if variant == "paper" else "dylan-upstream"

    def config(self):
        return {"system": self.name, "variant": self.variant, "upstream": UPSTREAM, "roles": self.roles,
                "rounds": ROUNDS, "activation": "listwise", "top_k": 2, "ranker_fallback": self.ranker_fallback,
                "shuffle_seed": self.seed, "max_calls_native": 2 * len(self.roles) + 3}

    def provenance(self):
        from runner import file_hashes
        root = VENDOR.parent.parent
        files = sorted(f for f in VENDOR.rglob("*") if f.is_file() and "__pycache__" not in f.parts)
        return {str(f.relative_to(root)): h for f, h in zip(files, file_hashes(files).values())}

    def worker_identities(self):
        return list(self.roles)

    def rng(self, ep):
        return random.Random(f"{self.seed}:{ep.trace['task_id']}:{ep.trace['rep']}")

    def __call__(self, ep, task, rng=None):
        rng = rng or self.rng(ep)
        q, n, agents = question(task), task["n_options"], len(self.roles)
        sysmsg = {r: P.ROLE_MAP[r] + "\n" + P.SYSTEM_PROMPT_MMLU for r in self.roles}
        answers = {}

        def activate(role, rnd, formers):
            formers = list(formers)
            rng.shuffle(formers)  # LLMNeuron.activate shuffles its inputs (no-op for round 0)
            msgs = [{"role": "system", "content": sysmsg[role]},
                    worker_message([ep.text(m) for m in formers], q, n)]
            mid = ep.call("worker", role, msgs, inputs=formers, round=rnd)
            answers[mid] = parse_choice(ep.text(mid), n)
            return mid

        def finish(voters, value, how):
            text = f"The answer is ({value})" if value else "No parseable answer"
            d = ep.decide("aggregator", CONSENSUS, inputs=voters, candidates=[ep.trace["messages"][m]["author"] for m in voters],
                          output=text, stop=True)
            ep.finish(final_text=text, decision=d, detail=how)

        def consensus(mids, mask_len, rnd):
            value, cnt = most_frequent([answers[m] for m in mids])
            reached = cnt > math.floor(2 / 3 * mask_len)
            ep.decide("stop", CONSENSUS, inputs=mids, round=rnd, stop=reached,
                      output={"answers": [answers[m] for m in mids], "value": value, "count": cnt,
                              "threshold": math.floor(2 / 3 * mask_len)})
            return reached, value

        previous = []  # replies of the previous round in role order (LLMNeuron.get_context edge order)
        for rnd in (0, 1):
            order = list(range(agents))
            rng.shuffle(order)
            done = {}
            for pos, a in enumerate(order):
                done[a] = activate(self.roles[a], rnd, previous)
                if pos >= math.floor(2 / 3 * agents):
                    mids = list(done.values())
                    reached, value = consensus(mids, agents, rnd)
                    if reached:
                        return finish(mids, value, f"consensus round {rnd}")
            previous = [done[a] for a in range(agents)]

        # round 2: listwise ranker over the shuffled round-1 replies
        indices = list(range(len(previous)))
        rng.shuffle(indices)
        shown = [previous[i] for i in indices]
        rmsg = P.construct_ranking_message([ep.text(m) for m in shown], q, "single_choice")
        rid = ep.call("ranker", RANKER, [rmsg], inputs=shown, round=2)
        tops, fallback = parse_ranks(ep.text(rid), len(shown)), None
        if tops is None:
            if self.ranker_fallback == "error":
                raise ControlError("ranker reply contains no [i,j] ranking")
            tops, fallback = rng.sample(list(range(len(shown))), 2), "random_pair"
        idx_mask = [indices[x] for x in tops]  # role indices, may repeat (e.g. "[3,3]")
        order = list(range(agents))
        rng.shuffle(order)
        if self.variant == "paper":
            run = [a for a in order if a in idx_mask]
        else:  # upstream: shuffled loop position compared with role indices
            run = [order[pos] for pos in range(agents) if pos in idx_mask]
        ranked = [self.roles[a] for a in idx_mask]
        ep.decide("ranker", RANKER, inputs=[rid], round=2, candidates=list(self.roles), ranking=ranked,
                  selected=[self.roles[a] for a in run],
                  output={"shown_order": [self.roles[i] for i in indices], "parsed": tops, "fallback": fallback})
        mids = []
        for a in run:
            mids.append(activate(self.roles[a], 2, previous))
            if len(mids) > math.floor(2 / 3 * len(idx_mask)):
                reached, value = consensus(mids, len(idx_mask), 2)
                if reached:
                    return finish(mids, value, "consensus round 2")
        return finish(mids, most_frequent([answers[m] for m in mids])[0], "most_frequent round 2")

