# services/embedding_service.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import uuid

from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter
from FlagEmbedding import BGEM3FlagModel
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
)


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


def generate_chunk_id(filename: str, chunk_index: int) -> str:
    """
    Deterministic unique ID for a chunk.
    Uses UUID5 (SHA-1 hash) with filename + chunk_index.
    """
    base = f"{filename}-{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, base))


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
        model_name: str,
        device: Optional[str],
        normalize: bool,
        batch_size: int,
        chunk_size: int,
        chunk_overlap: int,
    ):
        # Use BGEM3FlagModel instead of HuggingFaceEmbeddings
        self.embeddings = BGEM3FlagModel(
            model_name_or_path=model_name,
            devices=device,
            use_fp16=True,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            return_dense=True,
        )

        self.splitter = PureSizeTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ---------- Public API ----------
    def embed_query(self, text: str) -> List[float]:
        out = self.embeddings.encode(
            (text or "").strip(),
            return_dense=True,
        )
        vec = out["dense_vecs"]
        # Defensive: handle ndarray (D,), ndarray (1,D), or list
        return (
            vec[0].tolist()
            if getattr(vec, "ndim", 1) > 1
            else (vec.tolist() if hasattr(vec, "tolist") else list(vec))
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        safe_texts = [t.strip() for t in texts]
        out = self.embeddings.encode(
            safe_texts,
            return_dense=True,
        )
        return out["dense_vecs"].tolist()

    def embed_file(
        self,
        file_path: str | Path,
        *,
        csv_delimiter: str = ",",
        csv_encoding: str = "utf-8",
    ) -> List[ChunkEmbedding]:
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
        elif suffix == ".docx":
            docs = self._load_word(path)
        elif suffix == ".xlsx":
            docs = self._load_excel(path)
        else:
            raise ValueError(f"Unsupported file type: {suffix} (only .pdf, .txt, .csv)")

        split_docs = self.splitter.split_documents(docs)
        if not split_docs:
            return []

        texts = [d.page_content.strip() for d in split_docs]

        # Encode chunks with BGEM3 and extract dense vectors
        out = self.embeddings.encode(
            texts,
            return_dense=True,
        )
        vectors = out["dense_vecs"].tolist()

        results: List[ChunkEmbedding] = []
        total = len(split_docs)
        for i, (doc, vec) in enumerate(zip(split_docs, vectors)):
            meta = dict(doc.metadata or {})
            meta.update(
                {
                    "id": generate_chunk_id(path.name, i),
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
            d.metadata.setdefault("source", path.name)
        return docs

    def _load_txt(self, path: Path) -> List[Document]:
        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
        for d in docs:
            d.metadata.setdefault("source", path.name)
        return docs

    def _load_csv(self, path: Path, *, delimiter: str, encoding: str) -> List[Document]:
        loader = CSVLoader(
            str(path),
            csv_args={"delimiter": delimiter},
            encoding=encoding,
        )
        docs = loader.load()
        for d in docs:
            d.metadata.setdefault("source", path.name)
        return docs

    def _load_word(self, path: Path) -> List[Document]:
        loader = Docx2txtLoader(str(path))
        docs = loader.load()
        for d in docs:
            d.metadata.setdefault("source", path.name)
        return docs

    def _load_excel(self, path: Path, mode: str = "elements") -> List[Document]:
        loader = UnstructuredExcelLoader(str(path), mode=mode)
        docs = loader.load()
        for d in docs:
            d.metadata.setdefault("source", path.name)
        return docs
