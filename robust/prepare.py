"""Download/prepare data (and report model identity). Safe to run on a login node; no inference except a 1-token check."""
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Dict

from robust import PROTOCOL_VERSION
from robust.common import data_dir, get_logger, read_json, write_json
from robust.data import load_or_create_splits, parquet_path
from robust import llm

log = get_logger()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_masbench(cfg: Dict) -> Dict:
    dest = parquet_path(cfg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        local = os.getenv("MASBENCH_LOCAL_PARQUET")  # offline alternative: pre-copied file
        if local:
            shutil.copy(local, dest)
        else:
            url = (f"https://huggingface.co/datasets/{cfg['repo']}/resolve/{cfg['revision']}/"
                   f"{cfg['config']}/{cfg['source_split']}-00000-of-00001.parquet")
            log.info("downloading %s", url)
            tmp = dest.with_suffix(".tmp")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dest)
    return {"path": str(dest), "sha256": sha256(dest), "bytes": dest.stat().st_size}


def prepare_data(profile: Dict) -> Dict:
    info = fetch_masbench(profile["dataset"])
    sp = load_or_create_splits(profile["dataset"], profile["split_sizes"], profile["data_seed"], profile["name"])
    return {"parquet": info, "splits_sizes": sp["sizes"], "split_ids_sha256": {
        k: hashlib.sha256(",".join(v).encode()).hexdigest() for k, v in sp["ids"].items()}}


def prepare_embedding() -> Dict:
    from GDesigner.llm import profile_embedding as pe
    v = pe.get_sentence_embedding("check")
    return {"model": pe._MODEL_NAME, "dim": len(v), "HF_HOME": os.getenv("HF_HOME")}


def pull_via_api(tag: str) -> None:
    """Pull through the same HTTP endpoint the pipeline uses (avoids CLI/host mismatches)."""
    import json as _json
    import requests
    last = None
    with requests.post(f"{llm.base_url()}/api/pull", json={"model": tag, "stream": True}, stream=True, timeout=(30, 600)) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            ev = _json.loads(line)
            if ev.get("error"):
                raise RuntimeError(f"ollama pull {tag} failed: {ev['error']}")
            st = ev.get("status")
            if st != last and not ev.get("total"):
                log.info("pull %s: %s", tag, st)
                last = st
            elif ev.get("total") and ev.get("completed") is not None and int(ev["completed"]) % (256 << 20) < (4 << 20):
                log.info("pull %s: %.0f%%", tag, 100 * ev["completed"] / ev["total"])


def pull_models(profile: Dict, models) -> Dict:
    out = {}
    for key in models:
        m = profile["models"][key]
        tag = m["ollama_tag"]
        if llm.model_info(tag) is None:
            log.info("ollama pull %s", tag)
            pull_via_api(tag)
        info = llm.model_info(tag)
        out[key] = {**m, **(info or {}), "hf_origin": m.get("hf_origin")}
    return out


def run(profile: Dict, out_path: Path, with_models: bool = True, models=("qwen",), with_embedding: bool = True) -> Dict:
    man = {"protocol_version": PROTOCOL_VERSION, "profile": profile["name"],
           "profile_sha256": hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest(),
           "dataset_cfg": profile["dataset"], "data": prepare_data(profile)}
    if with_embedding:
        man["embedding"] = prepare_embedding()
    if with_models:
        if not llm.server_ready():
            raise RuntimeError("Ollama server not reachable (prepare.sh starts one)")
        man["models"] = pull_models(profile, models)
    write_json(out_path, man)
    return man
