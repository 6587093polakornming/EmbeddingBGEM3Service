# services/embedding_service.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch

from langchain_core.documents import Document  # updated import
from langchain_text_splitters import TextSplitter  # base class for custom splitter
# from langchain_text_splitters import RecursiveCharacterTextSplitter  # <- removed
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
)

# -------------------------
# Small config
# -------------------------
MODEL_NAME = "BAAI/bge-m3"  # multilingual BGE-M3
# Auto-pick device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NORMALIZE = True  # cosine similarity works best with normalized vecs
BATCH_SIZE = 32  # embedding batch size

CHUNK_SIZE = 800  # characters per chunk
CHUNK_OVERLAP = 150  # characters


# -------------------------
# Size-only splitter
# -------------------------
class PureSizeTextSplitter(TextSplitter):
    """
    Split text purely by character count with fixed overlap.
    - No separator heuristics
    - Deterministic chunking
    """

    def __init__(
        self,
        *,
        chunk_size: int,
        chunk_overlap: int = 0,
        add_start_index: bool = True,
    ):
        # Keep our own attrs and also pass through to the base for LC compatibility
        self._ps_chunk_size = int(chunk_size)
        self._ps_chunk_overlap = max(0, int(chunk_overlap))
        super().__init__(
            chunk_size=self._ps_chunk_size,
            chunk_overlap=self._ps_chunk_overlap,
            add_start_index=add_start_index,
        )

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []

        n = len(text)
        size = self._ps_chunk_size
        # Cap overlap at half the chunk size to ensure forward progress
        overlap = min(self._ps_chunk_overlap, size // 2)

        chunks: List[str] = []
        start = 0
        step = max(1, size - overlap)

        while start < n:
            end = min(start + size, n)
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start += step

        return chunks


# -------------------------
# Output model
# -------------------------
@dataclass
class ChunkEmbedding:
    chunk_text: str
    vector: List[float]
    meta: Dict[str, Any]


# -------------------------
# Core service
# -------------------------
class EmbeddingService:
    """
    Embedding service using LangChain:

    - PDF  -> PyPDFLoader
    - TXT  -> TextLoader
    - CSV  -> CSVLoader (row = doc)
    - Splitter: PureSizeTextSplitter (size-only)
    - Embeddings: HuggingFaceBgeEmbeddings (BAAI/bge-m3)
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = DEVICE,
        normalize: bool = NORMALIZE,
        batch_size: int = BATCH_SIZE,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        # HuggingFace BGE-M3 via LangChain adapter
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device} if device else {},
            encode_kwargs={
                "normalize_embeddings": normalize,
                "batch_size": batch_size,
            },
        )

        # Replace RecursiveCharacterTextSplitter with our size-only splitter
        self.splitter = PureSizeTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ---------- Public API ----------
    def embed_query(self, text: str) -> List[float]:
        """Embed a single user query string."""
        return self.embeddings.embed_query((text or "").strip())

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of raw strings."""
        safe_texts = [t.strip() for t in texts]
        return self.embeddings.embed_documents(safe_texts)

    def embed_file(
        self,
        file_path: str | Path,
        *,
        csv_delimiter: str = ",",
        csv_encoding: str = "utf-8",
    ) -> List[ChunkEmbedding]:
        """
        Load a file (pdf/txt/csv), split to chunks, and return embeddings per chunk.
        CSV: one row = one document (CSVLoader default behavior).
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(path)

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            docs = self._load_pdf(path)
        elif suffix == ".txt":
            docs = self._load_txt(path)
        elif suffix == ".csv":
            docs = self._load_csv(path, delimiter=csv_delimiter, encoding=csv_encoding)
        else:
            raise ValueError(f"Unsupported file type: {suffix} (only .pdf, .txt, .csv)")

        # Split documents into smaller chunks
        split_docs = self.splitter.split_documents(docs)
        if not split_docs:
            return []

        # Embed all chunks
        vectors = self.embeddings.embed_documents([d.page_content for d in split_docs])

        # Package results
        results: List[ChunkEmbedding] = []
        total = len(split_docs)
        for i, (doc, vec) in enumerate(zip(split_docs, vectors)):
            meta = dict(doc.metadata or {})
            meta.update(
                {
                    "chunk_index": i,
                    "total_chunks": total,
                    "chunk_size_chars": self.chunk_size,
                    "chunk_overlap_chars": self.chunk_overlap,
                }
            )
            results.append(
                ChunkEmbedding(chunk_text=doc.page_content, vector=vec, meta=meta)
            )
        return results

    # ---------- Loaders ----------
    def _load_pdf(self, path: Path) -> List[Document]:
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        for d in docs:
            d.metadata.setdefault("source", str(path))
        return docs

    def _load_txt(self, path: Path) -> List[Document]:
        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
        for d in docs:
            d.metadata.setdefault("source", str(path))
        return docs

    def _load_csv(self, path: Path, *, delimiter: str, encoding: str) -> List[Document]:
        loader = CSVLoader(
            str(path),
            csv_args={"delimiter": delimiter},  # per API
            encoding=encoding,
        )
        docs = loader.load()
        for d in docs:
            d.metadata.setdefault("source", str(path))
        return docs
