# services/embedding_service.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import uuid
import os

from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
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
    - Embeddings: HuggingFaceEmbeddings (BAAI/bge-m3)
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
        # --- NEW: offline-aware resolution (repo id -> HF_HOME cache path) ---
        resolved_model_name = self._resolve_model_prefer_cache(model_name)
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=resolved_model_name,
            cache_folder=os.getenv("HF_HOME"),        # no hardcode, no dup in model_kwargs
            model_kwargs={**({"device": device} if device else {})},
            encode_kwargs={"normalize_embeddings": normalize, "batch_size": batch_size},
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
        return self.embeddings.embed_query((text or "").strip())

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        safe_texts = [t.strip() for t in texts]
        return self.embeddings.embed_documents(safe_texts)

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

        vectors = self.embeddings.embed_documents([d.page_content for d in split_docs])

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

    # ---------- Loaders (unchanged) ----------
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

    def _resolve_model_prefer_cache(self, name_or_path: str) -> str:
        """
        Prefer a cached HF_HOME snapshot when a repo id is given, regardless of env vars.
        - If a local path is passed and exists -> use it
        - Else if name looks like 'org/name' and a cached snapshot exists in HF_HOME -> use that path
        - Else -> return the original name (may trigger network if not cached)
        """
        p = Path(name_or_path)
        if p.exists():
            return str(p.resolve())

        # Heuristic: repo id looks like "org/name"
        is_repo_id = ("/" in name_or_path) and not name_or_path.startswith(
            (".", "/", "\\")
        )
        if is_repo_id:
            cached = self._hf_cached_snapshot_path(name_or_path)
            if cached is not None:
                return str(cached)
        return name_or_path


    def _hf_cached_snapshot_path(self, repo_id: str) -> Optional[Path]:
        if "/" not in repo_id:
            return None
        org, model = repo_id.split("/", 1)

        # if you don't want to depend on env vars, hardcode your HF_HOME here:
        # hf_home = Path(r"D:\huggingface_cache")
        from pathlib import Path
        import os
        hf_home = Path(os.getenv("HF_HOME") or (Path.home() / ".cache" / "huggingface"))

        base = hf_home / "hub" / f"models--{org}--{model}" / "snapshots"
        if not base.exists():
            return None

        snapshots = [d for d in base.iterdir() if d.is_dir()]
        if not snapshots:
            return None

        # newest first
        snapshots.sort(key=lambda d: d.stat().st_mtime, reverse=True)

        # Only accept snapshots that look like a proper SentenceTransformers folder
        required_files = {"modules.json"}  # strongest signal for sbert layout
        for d in snapshots:
            if all((d / f).exists() for f in required_files):
                return d

        # If no snapshot has modules.json, return None -> we will fall back to repo id (online)
        return None

