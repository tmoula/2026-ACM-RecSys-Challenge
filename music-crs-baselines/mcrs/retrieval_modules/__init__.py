from .bm25 import BM25_MODEL
from .dense import BERT_MODEL, DenseRetriever
from .hybrid import HybridRetriever
from .fullstack import FullStackRetriever
from .cf_retrieve import CFRetriever

def load_retrieval_module(
        retrieval_type: str,
        dataset_name: str,
        track_split_types: list[str],
        corpus_types: list[str] = ["track_name", "artist_name", "album_name"],
        cache_dir: str = "./cache",
        device: str | None = None,
        retrieval_kwargs: dict | None = None,
    ):
    retrieval_kwargs = retrieval_kwargs or {}
    if retrieval_type == "bm25":
        return BM25_MODEL(dataset_name, track_split_types, corpus_types, cache_dir)
    if retrieval_type in ("bert", "dense"):
        return DenseRetriever(
            dataset_name,
            track_split_types,
            corpus_types,
            cache_dir,
            device=device,
            **retrieval_kwargs,
        )
    if retrieval_type == "hybrid":
        hybrid_kwargs = dict(retrieval_kwargs)
        pool_topk = hybrid_kwargs.pop("pool_topk", 100)
        rrf_k = hybrid_kwargs.pop("rrf_k", 60)
        dense_kwargs = hybrid_kwargs.pop("dense_kwargs", hybrid_kwargs)
        sparse = BM25_MODEL(dataset_name, track_split_types, corpus_types, cache_dir)
        dense = DenseRetriever(
            dataset_name,
            track_split_types,
            corpus_types,
            cache_dir,
            device=device,
            **dense_kwargs,
        )
        return HybridRetriever(sparse, dense, rrf_k=rrf_k, pool_topk=pool_topk)
    if retrieval_type == "fullstack":
        fs_kwargs = dict(retrieval_kwargs)
        pool_topk = fs_kwargs.pop("pool_topk", 500)
        rrf_k = fs_kwargs.pop("rrf_k", 60)
        cf_kwargs = fs_kwargs.pop("cf_kwargs", {})
        dense_kwargs = fs_kwargs.pop("dense_kwargs", fs_kwargs)
        sparse = BM25_MODEL(dataset_name, track_split_types, corpus_types, cache_dir)
        dense = DenseRetriever(
            dataset_name,
            track_split_types,
            corpus_types,
            cache_dir,
            device=device,
            **dense_kwargs,
        )
        cf = CFRetriever(cache_dir=cache_dir, **cf_kwargs)
        return FullStackRetriever(
            dense_retriever=dense,
            sparse_retriever=sparse,
            cf_retriever=cf,
            rrf_k=rrf_k,
            pool_topk=pool_topk,
        )
    raise ValueError(f"Unsupported retrieval type: {retrieval_type}")
