"""Módulo 04: embeddings.

Un embedding es un vector de números que representa el significado de un texto: textos
con significados parecidos producen vectores que apuntan en direcciones parecidas.
La similitud entre dos textos se mide con el coseno del ángulo entre sus vectores.

Se ofrecen dos implementaciones con la misma interfaz:

- `LSAEmbedder`: análisis semántico latente (TF-IDF + SVD) escrito desde cero con NumPy.
  No necesita descargar nada y se entrena con los propios documentos.
- `SentenceTransformerEmbedder`: un modelo neuronal multilingüe preentrenado que se
  ejecuta en local. Requiere el extra `[semantic]`.
- `OnnxEmbedder`: el mismo modelo, cuantizado a 8 bits y ejecutado con ONNX Runtime.
  Ocupa cuatro veces menos y no necesita PyTorch, por lo que cabe en servidores con
  poca memoria. Requiere el extra `[onnx]`.
"""

from __future__ import annotations

import math
import warnings
from collections import Counter
from collections.abc import Callable
from typing import Protocol

import numpy as np

from fundamenta.text import tokenize


class Embedder(Protocol):
    """Interfaz común: los documentos y las consultas pueden codificarse distinto."""

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Escala cada fila a longitud 1, de modo que el producto escalar sea el coseno."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Coseno del ángulo entre dos vectores: 1 = misma dirección, 0 = sin relación."""
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator else 0.0


class LSAEmbedder:
    """Embeddings por análisis semántico latente (Latent Semantic Analysis).

    1. Construye la matriz TF-IDF de los documentos (una fila por documento, una
       columna por término).
    2. La descompone con SVD y se queda con las `dimensions` direcciones principales.
       Los términos que suelen aparecer juntos acaban en direcciones parecidas, lo que
       captura algo de "significado" más allá de la coincidencia exacta de palabras.
    """

    def __init__(
        self, dimensions: int = 100, tokenizer: Callable[[str], list[str]] = tokenize
    ) -> None:
        self.dimensions = dimensions
        self.tokenizer = tokenizer
        self.vocabulary: dict[str, int] = {}
        self.idf = np.zeros(0)
        self.components = np.zeros((0, 0))

    def _tfidf(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), len(self.vocabulary)))
        for row, text in enumerate(texts):
            for term, count in Counter(self.tokenizer(text)).items():
                column = self.vocabulary.get(term)
                if column is not None:
                    matrix[row, column] = 1 + math.log(count)
        return normalize_rows(matrix * self.idf)

    def fit(self, texts: list[str]) -> LSAEmbedder:
        """Aprende el vocabulario, el IDF y las direcciones principales del corpus."""
        tokenized = [set(self.tokenizer(text)) for text in texts]
        terms = sorted(set().union(*tokenized))
        self.vocabulary = {term: i for i, term in enumerate(terms)}
        document_frequency = np.array([sum(term in doc for doc in tokenized) for term in terms])
        self.idf = np.log((1 + len(texts)) / (1 + document_frequency)) + 1

        _, _, vt = np.linalg.svd(self._tfidf(texts), full_matrices=False)
        self.components = vt[: self.dimensions]
        return self

    def _project(self, texts: list[str]) -> np.ndarray:
        if not self.vocabulary:
            raise RuntimeError("LSAEmbedder no está entrenado: llame primero a fit()")
        return normalize_rows(self._tfidf(texts) @ self.components.T)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._project(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._project([text])[0]


def _silence_model_logs() -> None:
    """Oculta barras de progreso y avisos informativos al cargar el modelo."""
    from huggingface_hub.utils import disable_progress_bars
    from huggingface_hub.utils import logging as hub_logging
    from transformers.utils import logging as transformers_logging

    disable_progress_bars()
    hub_logging.set_verbosity_error()
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()


class SentenceTransformerEmbedder:
    """Embeddings con un modelo neuronal preentrenado (sentence-transformers).

    El modelo por defecto, `intfloat/multilingual-e5-small`, es multilingüe (incluye
    español), ocupa unos 470 MB y funciona en CPU. Los modelos de la familia E5 esperan
    los prefijos "query: " y "passage: " para distinguir preguntas de documentos.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        query_prefix: str = "query: ",
        document_prefix: str = "passage: ",
        batch_size: int = 32,
        quiet: bool = True,
    ) -> None:
        try:
            with warnings.catch_warnings():  # aviso de tqdm irrelevante fuera de Jupyter
                warnings.simplefilter("ignore")
                from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError(
                "SentenceTransformerEmbedder requiere el extra [semantic]: "
                'pip install -e ".[semantic]"'
            ) from error
        if quiet:
            _silence_model_logs()
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.document_prefix + text for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_prefix + text])[0]


class OnnxEmbedder:
    """El modelo E5 multilingüe cuantizado, ejecutado con ONNX Runtime en lugar de PyTorch.

    Replica lo que hace sentence-transformers con este modelo: tokeniza con SentencePiece,
    obtiene un vector por token, los promedia ignorando el relleno (mean pooling) y
    normaliza. Los pesos cuantizados a 8 bits ocupan 118 MB y el proceso completo cabe
    en unos 300 MB de memoria, por lo que funciona en los planes gratuitos de hosting.
    """

    # Identificadores especiales del vocabulario de XLM-RoBERTa, en el que se basa E5
    BOS, PAD, EOS, UNK = 0, 1, 2, 3

    def __init__(
        self,
        model_repo: str = "Xenova/multilingual-e5-small",
        model_file: str = "onnx/model_int8.onnx",
        tokenizer_repo: str = "intfloat/multilingual-e5-small",
        query_prefix: str = "query: ",
        document_prefix: str = "passage: ",
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        try:
            import onnxruntime
            import sentencepiece
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import disable_progress_bars
        except ImportError as error:
            raise ImportError(
                'OnnxEmbedder requiere el extra [onnx]: pip install -e ".[onnx]"'
            ) from error
        disable_progress_bars()
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = sentencepiece.SentencePieceProcessor(
            model_file=hf_hub_download(tokenizer_repo, "sentencepiece.bpe.model")
        )
        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3
        options.enable_cpu_mem_arena = False  # devuelve la memoria tras cada lote
        self.session = onnxruntime.InferenceSession(
            hf_hub_download(model_repo, model_file), options, providers=["CPUExecutionProvider"]
        )
        self.input_names = {model_input.name for model_input in self.session.get_inputs()}

    def _token_ids(self, text: str) -> list[int]:
        # El vocabulario del modelo desplaza en 1 los identificadores de SentencePiece
        pieces = [self.UNK if i == 0 else i + 1 for i in self.tokenizer.encode(text)]
        return [self.BOS, *pieces[: self.max_length - 2], self.EOS]

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = [self._token_ids(text) for text in texts[start : start + self.batch_size]]
            width = max(len(ids) for ids in batch)
            ids = np.full((len(batch), width), self.PAD, dtype=np.int64)
            mask = np.zeros((len(batch), width), dtype=np.int64)
            for row, tokens in enumerate(batch):
                ids[row, : len(tokens)] = tokens
                mask[row, : len(tokens)] = 1
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self.session.run(None, feed)[0]
            weights = mask[..., None].astype(np.float32)
            vectors.append((hidden * weights).sum(axis=1) / weights.sum(axis=1))
        return normalize_rows(np.vstack(vectors))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.document_prefix + text for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_prefix + text])[0]


def get_embedder(name: str) -> Embedder:
    """Crea un embedder por nombre: "lsa", "semantic" u "onnx"."""
    if name == "lsa":
        return LSAEmbedder()
    if name == "semantic":
        return SentenceTransformerEmbedder()
    if name == "onnx":
        return OnnxEmbedder()
    raise ValueError(f"Embedder desconocido: {name!r} (opciones: lsa, semantic, onnx)")
