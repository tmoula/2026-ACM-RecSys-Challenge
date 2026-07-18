from .embedding_reranker import EmbeddingReranker
from .cross_encoder_reranker import CrossEncoderReranker
from .lgbm_reranker import LGBMReranker
from .lambdamart_reranker import LambdaMARTReRanker


def load_reranker_module(reranker_type: str, **kwargs):
    cache_dir = kwargs.pop("cache_dir", "./cache")
    kwargs.pop("device", None)
    if reranker_type == "embedding":
        kwargs.setdefault("cache_dir", cache_dir)
        return EmbeddingReranker(**kwargs)
    if reranker_type == "cross_encoder":
        return CrossEncoderReranker(**kwargs)
    if reranker_type == "lgbm":
        kwargs.setdefault("cache_dir", cache_dir)
        return LGBMReranker(**kwargs)
    if reranker_type == "lambdamart":
        kwargs.setdefault("cache_dir", cache_dir)
        return LambdaMARTReRanker(**kwargs)
    raise ValueError(f"Unsupported reranker type: {reranker_type}")
