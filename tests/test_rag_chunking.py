import pytest

from app.backend.rag_service import _chunk_text, _collection_name


def test_chunk_respects_max_size():
    text = "a" * 2000
    chunks = _chunk_text(text, chunk_size=800, overlap=120)
    assert all(len(c) <= 800 for c in chunks)
    assert len(chunks) >= 2


def test_chunk_overlap_matches():
    text = "a" * 2000
    chunks = _chunk_text(text, chunk_size=800, overlap=120)
    assert chunks[0][-120:] == chunks[1][:120]


def test_chunk_empty_text():
    assert _chunk_text("   \n\t  ") == []


def test_chunk_whitespace_normalized():
    text = "hello   \n\n  world"
    chunks = _chunk_text(text, chunk_size=800, overlap=120)
    assert chunks == ["hello world"]


def test_chunk_size_must_exceed_overlap():
    with pytest.raises(ValueError):
        _chunk_text("some text", chunk_size=100, overlap=100)


def test_collection_name_sanitizes_special_chars():
    name = _collection_name("weird user!! name")
    assert name == "user_weird_user___name_docs"


def test_collection_name_empty_user_falls_back_to_anon():
    assert _collection_name("") == "user_anon_docs"
