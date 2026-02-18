from functools import lru_cache
from sentence_transformers import SentenceTransformer

@lru_cache(maxsize=1)
def get_embedder(model_name: str) -> SentenceTransformer:
    # Cached singleton (loads once)
    return SentenceTransformer(model_name)

def embed_texts(model_name: str, texts: list[str]) -> list[list[float]]:
    model = get_embedder(model_name)
    emb = model.encode(texts, normalize_embeddings=True)
    return emb.tolist()

def embed_query(model_name: str, text: str) -> list[float]:
    return embed_texts(model_name, [text])[0]
