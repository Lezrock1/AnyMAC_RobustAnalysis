import os
from functools import lru_cache

from sentence_transformers import SentenceTransformer

_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'


@lru_cache(maxsize=1)
def _get_model():
    # Loaded once instead of once per call; device is configurable (default: cpu).
    return SentenceTransformer(_MODEL_NAME, device=os.getenv('EMBED_DEVICE', 'cpu'))


@lru_cache(maxsize=20000)
def _encode_cached(sentence):
    emb = _get_model().encode(sentence)
    emb.setflags(write=False)
    return emb


def get_sentence_embedding(sentence):
    if isinstance(sentence, str):
        return _encode_cached(sentence).copy()
    return _get_model().encode(sentence)
