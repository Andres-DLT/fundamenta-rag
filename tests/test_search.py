"""Tests de los módulos 03 a 06: texto, BM25, embeddings, almacén vectorial y recuperación."""

import numpy as np
import pytest

from fundamenta.bm25 import BM25Index
from fundamenta.embeddings import LSAEmbedder, cosine_similarity, get_embedder
from fundamenta.loaders import Document
from fundamenta.retriever import Retriever, mmr, reciprocal_rank_fusion
from fundamenta.text import light_stem, strip_accents, tokenize
from fundamenta.vector_store import VectorStore

# --- Módulo 03: texto y BM25 ---------------------------------------------------


def test_strip_accents():
    assert strip_accents("Información, pingüino, año") == "Informacion, pinguino, ano"


def test_tokenize_normalizes_and_removes_stopwords():
    assert tokenize("¿Cuántos DÍAS de vacaciones tengo?", stem=False) == ["dias", "vacaciones"]


@pytest.mark.parametrize(
    "variants",
    [("vacación", "vacaciones"), ("permiso", "permisos"), ("viaje", "viajes"), ("día", "días")],
)
def test_light_stem_groups_variants(variants):
    stems = {light_stem(strip_accents(word)) for word in variants}
    assert len(stems) == 1


def test_bm25_ranks_the_most_specific_document_first():
    texts = [
        "Las vacaciones se solicitan en NovaHR.",
        "La dieta diaria en España es de 45 euros.",
        "Política general de la empresa sobre horarios.",
    ]
    index = BM25Index().fit(texts)

    assert index.search("¿Cómo solicito las vacaciones?")[0][0] == 0
    assert index.search("dieta España")[0][0] == 1


def test_bm25_gives_more_weight_to_rare_terms():
    index = BM25Index().fit(["empresa vacaciones", "empresa horario", "empresa dietas"])

    assert index.idf["vacacio"] > index.idf["empres"]


def test_bm25_ignores_documents_without_query_terms():
    index = BM25Index().fit(["uno", "dos"])

    assert index.search("tres") == []


# --- Módulo 04: embeddings -----------------------------------------------------


def test_cosine_similarity():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([2.0, 0.0])) == pytest.approx(1.0)
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 3.0])) == pytest.approx(0.0)
    assert cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 0.0


def test_lsa_embedder_produces_normalized_vectors_and_groups_topics():
    texts = [
        "vacaciones días libres solicitud",
        "vacaciones días festivos calendario",
        "portátil contraseña VPN soporte",
        "portátil soporte incidencia equipo",
    ]
    embedder = LSAEmbedder(dimensions=2).fit(texts)
    vectors = embedder.embed_documents(texts)

    assert vectors.shape == (4, 2)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    query = embedder.embed_query("días de vacaciones")
    assert query @ vectors[0] > query @ vectors[2]


def test_lsa_embedder_requires_fit():
    with pytest.raises(RuntimeError):
        LSAEmbedder().embed_query("hola")


def test_get_embedder_rejects_unknown_names():
    with pytest.raises(ValueError):
        get_embedder("desconocido")


@pytest.mark.semantic
def test_semantic_embedder_understands_paraphrases():
    pytest.importorskip("sentence_transformers")
    embedder = get_embedder("semantic")
    docs = embedder.embed_documents(
        ["Si pierdes el portátil o te lo roban, avisa a IT.", "El plan Starter cuesta 15 €."]
    )
    query = embedder.embed_query("Me han robado el ordenador")

    assert query @ docs[0] > query @ docs[1]


@pytest.mark.semantic
def test_onnx_embedder_matches_the_full_model():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("sentence_transformers")
    texts = ["Si pierdes el portátil o te lo roban, avisa a IT.", "El plan Starter cuesta 15 €."]
    onnx = get_embedder("onnx")
    full = get_embedder("semantic")

    vectors = onnx.embed_documents(texts)
    query = onnx.embed_query("Me han robado el ordenador")

    assert vectors.shape == (2, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    assert query @ vectors[0] > query @ vectors[1]
    # La cuantización apenas altera los vectores del modelo original
    assert np.all(np.sum(vectors * full.embed_documents(texts), axis=1) > 0.95)


# --- Módulo 05: almacén vectorial ---------------------------------------------


def _store() -> VectorStore:
    store = VectorStore()
    store.add(
        np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]]),
        [
            Document("a", {"source": "uno.md"}),
            Document("b", {"source": "dos.md"}),
            Document("c", {"source": "uno.md"}),
        ],
    )
    return store


def test_vector_store_returns_most_similar_first():
    results = _store().search(np.array([1.0, 0.1]), k=2)

    assert [i for i, _ in results] == [0, 1]
    assert results[0][1] > results[1][1]


def test_vector_store_filters_by_metadata():
    results = _store().search(np.array([1.0, 0.1]), k=3, where={"source": "uno.md"})

    assert [i for i, _ in results] == [0, 2]


def test_vector_store_rejects_wrong_dimensions():
    with pytest.raises(ValueError):
        _store().add(np.array([[1.0, 0.0, 0.0]]), [Document("d")])


def test_vector_store_save_and_load(tmp_path):
    store = _store()
    store.save(tmp_path / "indice")

    loaded = VectorStore.load(tmp_path / "indice")

    assert len(loaded) == 3
    assert loaded.documents[1] == store.documents[1]
    assert loaded.search(np.array([0.0, 1.0]), k=1)[0][0] == 2


# --- Módulo 06: recuperación ----------------------------------------------------


def test_reciprocal_rank_fusion():
    fused = dict(reciprocal_rank_fusion([[1, 2], [2, 3]], k=60))

    assert fused[2] == pytest.approx(1 / 62 + 1 / 61)
    assert max(fused, key=fused.get) == 2


def test_reciprocal_rank_fusion_weights():
    fused = reciprocal_rank_fusion([[1], [2]], k=60, weights=[0.5, 1.0])

    assert [i for i, _ in fused] == [2, 1]


def test_mmr_avoids_near_duplicates():
    vectors = np.array([[1.0, 0.0], [0.99, 0.14], [0.7, 0.71]])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    selected = mmr(np.array([1.0, 0.0]), vectors, [0, 1, 2], k=2, diversity=0.7)

    assert selected == [0, 2]


def test_retriever_requires_embedder_for_semantic_modes(novatech_chunks):
    with pytest.raises(ValueError):
        Retriever(novatech_chunks, mode="hybrid")


def test_lexical_retriever_uses_section_context(novatech_chunks):
    retriever = Retriever(novatech_chunks, mode="lexical")

    [top] = retriever.search("días de permiso por mudanza", k=1)

    assert "Mudanza" in top.document.text
    assert top.rank == 1


def test_hybrid_retriever_with_filters(novatech_chunks):
    retriever = Retriever(novatech_chunks, embedder=LSAEmbedder(), mode="hybrid")

    results = retriever.search("vacaciones", k=3, where={"source": "manual-del-empleado.md"})

    assert results
    assert all(r.document.source == "manual-del-empleado.md" for r in results)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_retriever_with_diversity(novatech_chunks):
    retriever = Retriever(novatech_chunks, embedder=LSAEmbedder(), mode="semantic")

    results = retriever.search("vacaciones", k=3, diversity=0.5)

    assert len({id(r.document) for r in results}) == 3
