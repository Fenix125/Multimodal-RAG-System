import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union

class TextEmbedder:
    """
    Wrapper around sentence-transformers models.

    Supports:
      - "sentence-transformers/all-mpnet-base-v2"
      - "intfloat/multilingual-e5-base"
      - "intfloat/multilingual-e5-large"

    For E5-family models, we follow the recommended formatting:
      - queries:  "query: <text>"
      - passages: "passage: <text>"
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def ensure_list(self, texts: Union[str, List[str]]) -> List[str]:
        if isinstance(texts, str):
            return [texts]
        return list(texts)

    def format_texts_for_e5(self, texts: List[str], purpose: str) -> List[str]:
        """
        Apply E5-style prefixes if needed
        """
        if "e5" not in self.model_name.lower():
            return texts

        if purpose == "query":
            prefix = "query: "
        else:
            prefix = "passage: "

        return [prefix + t for t in texts]

    def embed_documents(self, docs: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Embed documents/passages for indexing in the vector store.
        Returns a numpy array of shape (B, D).
        """
        docs_list = self.ensure_list(docs)
        docs_list = self.format_texts_for_e5(docs_list, purpose="passage")

        embeddings = self.model.encode(
            docs_list,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return embeddings

    def embed_queries(self, queries: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Embed queries for retrieval against the vector store.
        Returns a numpy array of shape (B, D).
        """
        queries_list = self.ensure_list(queries)
        queries_list = self.format_texts_for_e5(queries_list, purpose="query")

        embeddings = self.model.encode(
            queries_list,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return embeddings

    def embed_documents_to_list(self, docs: Union[str, List[str]], normalize: bool = True) -> List[List[float]]:
        return self.embed_documents(docs, normalize=normalize).tolist()

    def embed_queries_to_list(self, queries: Union[str, List[str]], normalize: bool = True) -> List[List[float]]:
        return self.embed_queries(queries, normalize=normalize).tolist()
