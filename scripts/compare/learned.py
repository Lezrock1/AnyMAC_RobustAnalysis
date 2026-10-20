"""Frozen learned routers. G-Designer is an independent, qualified paper reconstruction.

See docs/LEARNED_COMPARISON.md for equation interpretations and reproduction limits.
No upstream G-Designer code is imported. Historical AnyMAC checkpoints are not accepted.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical
from torch.nn import functional as F

from baseline import history_block, system_prompt

ROLES = ["Knowledgeable Expert", "Critic", "Mathematician", "Psychologist", "Historian",
         "Doctor", "Lawyer", "Economist", "Programmer"]
DECIDER = "Decider"
SYSTEMS = ("anymac-corrected", "gdesigner-paper")
ARTIFACT_SCHEMA = "mascmp-learned/1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def environment():
    return {name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "sentence-transformers")}


def runtime_sources():
    here = Path(__file__).resolve().parent
    return {name: digest(here / name) for name in ("learned.py", "baseline.py", "contract.py", "scoring.py")}


def tree_digest(path):
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"local model directory does not exist: {path}")
    h = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file() and ".cache" not in p.relative_to(path).parts)
    if not files:
        raise ValueError(f"empty model directory: {path}")
    for p in files:
        h.update(p.relative_to(path).as_posix().encode())
        h.update(bytes.fromhex(digest(p)))
    return h.hexdigest()


def worker_identity(identity):
    # CPU threading is execution metadata, not a worker-model or decoding change.
    value = json.loads(json.dumps(identity))
    value.get("options", {}).pop("num_thread", None)
    return value


class Embedder:
    def __init__(self, path):
        from sentence_transformers import SentenceTransformer
        self.sha256 = tree_digest(path)
        self.model = SentenceTransformer(str(Path(path).resolve()), device="cpu")
        self.cache = {}

    def __call__(self, text):
        if text not in self.cache:
            value = torch.as_tensor(self.model.encode(text), dtype=torch.float32).detach().clone()
            if value.shape != (384,) or not torch.isfinite(value).all():
                raise ValueError("expected a finite 384-dimensional MiniLM embedding")
            self.cache[text] = value
        return self.cache[text]


class AnyMACPolicy(nn.Module):
    """Task, roles, history, NAP, NHP. Corrected role/history offsets and terminal gradient."""

    def __init__(self, config, role_embeddings, backbone=None):
        super().__init__()
        from transformers import GPT2Config, GPT2Model
        self.config = config
        self.register_buffer("role_embeddings", role_embeddings.clone())
        self.transformer = (GPT2Model.from_pretrained(str(backbone), local_files_only=True) if backbone else
                            GPT2Model(GPT2Config.from_dict(config["gpt2_config"])))
        width = self.transformer.config.n_embd
        self.norm = nn.LayerNorm(width)
        self.task = nn.Linear(384, width)
        self.role = nn.Linear(384, width)
        self.history = nn.Linear(768, width)
        self.nap = nn.Linear(384, width)
        self.nhp = nn.Linear(384, width)

    def scores(self, task, history, authors):
        roles = self.role(self.role_embeddings)
        parts = [self.task(task).unsqueeze(0), roles]
        if history:
            h = torch.stack([self.history(torch.cat((self.role_embeddings[a], x)))
                             for a, x in zip(authors, history)])
            parts.append(h)
        parts += [self.nap(task).unsqueeze(0), self.nhp(task).unsqueeze(0)]
        x = torch.cat(parts)
        positions = torch.arange(len(x))
        prefix = len(roles) + 1
        positions[:prefix] = 0
        encoded = self.norm(self.transformer(inputs_embeds=x.unsqueeze(0),
                                           position_ids=positions.unsqueeze(0),
                                           attention_mask=torch.ones(1, len(x))).last_hidden_state[0])
        agents = (encoded[1:prefix] * encoded[-2]).sum(-1)
        agents = (agents - agents.mean()) / (agents.std() + 1e-8)
        hints = F.cosine_similarity(encoded[prefix:prefix + len(history)], encoded[-1].unsqueeze(0), dim=-1)
        return agents, hints * 2

    def objective(self, replay):
        history, authors, terms = [], [], []
        for step in replay["steps"]:
            agents, hints = self.scores(replay["task"], history, authors)
            if not step["forced"]:
                terms.append(Categorical(logits=agents * 1.5).log_prob(torch.tensor(step["role"])))
            if history:
                eligible = min(len(history), self.config["max_context"])
                terms.append(Bernoulli(logits=hints[-eligible:] * 1.5).log_prob(
                    torch.tensor(step["mask"][-eligible:], dtype=torch.float32)).sum())
            if step["role"] < len(self.config["roles"]):
                authors.append(step["role"])
                history.append(step["embedding"])
        return torch.stack(terms).sum(), torch.tensor(0.0)


def normalized_adjacency(adjacency):
    adjacency = adjacency + torch.eye(len(adjacency))
    inv = adjacency.sum(-1).rsqrt()
    return inv[:, None] * adjacency * inv[None, :]


class GCN(nn.Module):
    def __init__(self, hidden, latent):
        super().__init__()
        self.first = nn.Linear(384, hidden)
        self.second = nn.Linear(hidden, latent)

    def forward(self, x, adjacency):
        return self.second(adjacency @ F.relu(self.first(adjacency @ x)))


def refine(sketch, anchor, rank, zeta):
    """Minimizing interpretation of Eq. 14, solved in the fixed top-left singular subspace."""
    z = torch.linalg.svd(sketch, full_matrices=False).U[:, :rank]
    # Freeze the subspace in the inner minimization; gradients still reach sketch via the target.
    z = z.detach()
    target = z.T @ ((sketch + anchor) / 2) @ z
    u, s, vh = torch.linalg.svd(target, full_matrices=False)
    w = (u * (s - zeta / 2).clamp_min(0)) @ vh
    return z @ w @ z.T, w


class GDesignerPolicy(nn.Module):
    def __init__(self, config, role_embeddings):
        super().__init__()
        self.config = config
        self.register_buffer("role_embeddings", role_embeddings.clone())
        n = len(config["roles"])
        anchor = torch.zeros(n, n)
        for i in range(n - 1):
            anchor[i, i + 1] = 1
        self.register_buffer("anchor", anchor)
        adjacency = torch.zeros(n + 1, n + 1)
        adjacency[:n, :n] = anchor + anchor.T
        adjacency[-1, :-1] = adjacency[:-1, -1] = 1
        self.register_buffer("adjacency", normalized_adjacency(adjacency))
        self.mu = GCN(config["hidden"], config["latent"])
        self.log_sigma = GCN(config["hidden"], config["latent"])
        self.decoder = nn.Sequential(nn.Linear(3 * config["latent"], config["hidden"]), nn.ReLU(),
                                     nn.Linear(config["hidden"], 1))

    def distribution(self, task, noise):
        x = torch.cat((self.role_embeddings, task.unsqueeze(0)))
        mu = self.mu(x, self.adjacency)
        log_sigma = self.log_sigma(x, self.adjacency).clamp(-8, 4)
        h = mu + log_sigma.exp() * noise["normal"]
        n = len(self.anchor)
        features = torch.cat((h[:n, None, :].expand(n, n, -1),
                              h[None, :n, :].expand(n, n, -1),
                              h[-1].expand(n, n, -1)), dim=-1)
        logits = self.decoder(features).squeeze(-1)
        sketch = torch.sigmoid((logits + noise["logistic"]) / self.config["tau"])
        refined, w = refine(sketch, self.anchor, self.config["rank"], self.config["zeta"])
        # Nuclear shrinkage does not create entry-wise zeros; explicit probabilistic DAG projection.
        probs = refined.clamp(1e-6, 1 - 1e-6)
        regularizer = (0.5 * (sketch - refined).square().sum()
                       + 0.5 * (self.anchor - refined).square().sum()
                       + self.config["zeta"] * torch.linalg.svdvals(w).sum())
        return probs, regularizer

    def objective(self, replay):
        probs, regularizer = self.distribution(replay["task"], replay["noise"])
        eligible = torch.triu(torch.ones_like(probs, dtype=torch.bool), diagonal=1)
        likelihood = Bernoulli(probs=probs[eligible]).log_prob(replay["edges"][eligible]).sum()
        return likelihood, regularizer


def policy_from_config(config, embeddings, backbone=None):
    if config["system"] == "anymac-corrected":
        return AnyMACPolicy(config, embeddings, backbone)
    if config["system"] == "gdesigner-paper":
        return GDesignerPolicy(config, embeddings)
    raise ValueError(f"unknown learned system {config['system']}")


def new_config(system, roles=None, backbone=None):
    config = {"system": system, "roles": roles or list(ROLES), "schema": ARTIFACT_SCHEMA,
              "prompt": "common-answer-spec/1", "seed": 0}
    if system == "anymac-corrected":
        from transformers import GPT2Config
        config.update(max_workers=10, max_context=10,
                      gpt2_config=GPT2Config.from_pretrained(str(backbone), local_files_only=True).to_dict())
    elif system == "gdesigner-paper":
        config.update(rounds=3, hidden=32, latent=16, rank=min(4, len(config["roles"])), zeta=0.1, tau=0.01,
                      interpretation="minimize-eq14-fixed-subspace-bernoulli-forward-dag/1")
    else:
        raise ValueError(system)
    return config


def call_agent(ep, task, role, inputs, round=None, final=False):
    user = f"Task:\n{task['text']}"
    if inputs:
        user += "\n\nPrevious agents:\n\n" + history_block(ep, inputs)
    user += "\n\nWeigh the evidence and give your answer."
    return ep.call("aggregator" if final else "worker", role,
                   [{"role": "system", "content": system_prompt(role, task)},
                    {"role": "user", "content": user}], inputs=inputs, round=round)


class LearnedSystem:
    def __init__(self, policy, embedder, artifact=None, metadata=None):
        self.policy, self.embedder = policy, embedder
        self.settings = policy.config
        self.name = self.settings["system"]
        self.roles = list(self.settings["roles"])
        self.artifact, self.metadata = artifact, metadata
        self.last_replay = None
        self.policy.eval()

    def worker_identities(self):
        return list(self.roles)

    def config(self):
        return json.loads(json.dumps(self.settings))

    def provenance(self):
        return {"artifact_sha256": digest(self.artifact) if self.artifact else None,
                "embedding_sha256": self.embedder.sha256,
                "qualification": "corrected-reimplementation" if self.name == "anymac-corrected" else
                "paper-derived-reconstruction-not-verified-reproduction"}

    def validate_model(self, identity):
        if self.metadata and worker_identity(identity) != self.metadata["worker"]:
            raise ValueError("router was trained with another worker model/decoding/serving version")

    def __call__(self, ep, task, sample=False, generator=None):
        generator = generator or torch.Generator().manual_seed(
            random.Random(f"{self.settings['seed']}:{task['id']}:{ep.trace['rep']}").randrange(2**63))
        self.policy.eval()  # no dropout mismatch between rollout and likelihood recomputation
        with torch.no_grad():
            if self.name == "anymac-corrected":
                self._anymac(ep, task, sample, generator)
            else:
                self._gdesigner(ep, task, generator)

    def _anymac(self, ep, task, sample, generator):
        embedding = self.embedder(task["text"])
        history, authors, mids, steps = [], [], [], []
        choices = self.roles + [DECIDER]
        for step in range(self.settings["max_workers"] + 1):
            start = time.perf_counter()
            agents, hints = self.policy.scores(embedding, history, authors)
            forced = step == self.settings["max_workers"]
            role = len(self.roles) if forced else (
                int(torch.multinomial((agents * 1.5).softmax(0), 1, generator=generator)) if sample else
                int(agents.argmax()))
            mask = ((torch.rand(len(hints), generator=generator) < (hints * 1.5).sigmoid()).int() if sample else
                    (hints > 0).int())
            if len(mask) > self.settings["max_context"]:
                mask[:-self.settings["max_context"]] = 0
            delivered = [m for m, keep in zip(mids, mask) if keep]
            ep.decide("router", "AnyMAC", inputs=mids, round=step, candidates=choices, selected=[choices[role]],
                      edges=[[m, choices[role]] for m in delivered], compute_s=time.perf_counter() - start)
            ep.decide("router", "AnyMAC-context", inputs=mids, round=step, candidates=mids, selected=delivered)
            mid = call_agent(ep, task, choices[role], delivered, round=step, final=role == len(self.roles))
            action = {"role": role, "mask": mask.tolist(), "forced": forced}
            steps.append(action)
            if role == len(self.roles):
                ep.finish(final_msg=mid)
                break
            action["embedding"] = self.embedder(ep.text(mid))
            history.append(action["embedding"])
            authors.append(role)
            mids.append(mid)
        self.last_replay = {"task": embedding, "steps": steps}

    def _gdesigner(self, ep, task, generator):
        embedding = self.embedder(task["text"])
        n, latent = len(self.roles), self.settings["latent"]
        uniform = torch.rand(n, n, generator=generator).clamp(1e-6, 1 - 1e-6)
        noise = {"normal": torch.randn(n + 1, latent, generator=generator),
                 "logistic": uniform.log() - (-uniform).log1p()}
        start = time.perf_counter()
        probs, _ = self.policy.distribution(embedding, noise)
        edges = torch.triu((torch.rand(n, n, generator=generator) < probs).float(), diagonal=1)
        connections = [[self.roles[i], self.roles[j]] for i, j in edges.nonzero().tolist()]
        ep.decide("router", "GDesigner-task-graph", candidates=self.roles, selected=self.roles, edges=connections,
                  compute_s=time.perf_counter() - start)
        for r in range(self.settings["rounds"]):
            current = []
            for j, role in enumerate(self.roles):
                inputs = [current[i] for i in range(j) if edges[i, j]]
                current.append(call_agent(ep, task, role, inputs, round=r))
            final = call_agent(ep, task, DECIDER, current, round=r, final=True)
        ep.finish(final_msg=final)
        self.last_replay = {"task": embedding, "noise": noise, "edges": edges}


def load_system(path, embedding_path, require_gate=True):
    """Only load a locally produced, hash-bound tensor artifact, never a pickled historical graph."""
    path = Path(path)
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema") != ARTIFACT_SCHEMA or digest(path) != metadata["artifact_sha256"]:
        raise ValueError("unrecognized or changed learned checkpoint")
    if metadata["runtime_sources"] != runtime_sources() or metadata["environment"] != environment():
        raise ValueError("implementation or dependency versions changed since training")
    if require_gate:
        if metadata["purpose"] != "study":
            raise ValueError("runtime-smoke checkpoint cannot be used for evaluation")
        gate = json.loads(path.with_suffix(".gate.json").read_text())
        if (not gate["passed"] or gate["artifact_sha256"] != digest(path)
                or gate["metadata_sha256"] != digest(metadata_path)
                or gate["runtime_sources"] != runtime_sources() or gate["environment"] != environment()):
            raise ValueError("missing, stale or failed development gate")
    embedder = Embedder(embedding_path)
    if embedder.sha256 != metadata["embedding_sha256"]:
        raise ValueError("embedding model changed since training")
    state = torch.load(path, map_location="cpu", weights_only=True)
    policy = policy_from_config(metadata["config"], state["role_embeddings"])
    policy.load_state_dict(state, strict=True)
    return LearnedSystem(policy, embedder, path, metadata)
