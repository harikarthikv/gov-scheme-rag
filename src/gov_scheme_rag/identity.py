"""Portable source identities shared by ingestion, retrieval and evaluation."""

import hashlib
import re


def canonical_filename(filename: str) -> str:
    return re.sub(r"[ _-]+copy(?:[ _-]*\d+)?(?=\.pdf$)", "", filename.lower())


def normalized_text(text: str) -> str:
    return " ".join(text.split())


def content_id(text: str) -> str:
    return hashlib.sha256(normalized_text(text).encode()).hexdigest()


def chunk_id(text: str, metadata: dict) -> str:
    source = metadata.get("document_id", canonical_filename(metadata.get("filename", "")))
    payload = f"{source}:{metadata.get('page')}:{normalized_text(text)}"
    return hashlib.sha256(payload.encode()).hexdigest()
