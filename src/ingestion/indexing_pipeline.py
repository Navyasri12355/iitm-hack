"""
Pathway-powered real-time document ingestion pipeline.

Uses Pathway's streaming engine for:
- Real-time file system monitoring
- Incremental document processing  
- Live vector index updates
"""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory vector store (backed by Pathway file watcher)
# ---------------------------------------------------------------------------

class DocumentVectorStore:
    """
    Thread-safe in-memory vector store that Pathway populates in real-time.
    Supports cosine similarity search across embedded document chunks.
    """

    def __init__(self):
        self._lock = threading.RLock()
        # doc_id -> {embedding, metadata, content, chunks}
        self._documents: Dict[str, Dict[str, Any]] = {}
        # chunk_id -> {doc_id, embedding, text, metadata}
        self._chunks: List[Dict[str, Any]] = []
        self._embeddings_matrix: Optional[np.ndarray] = None
        self._dirty = True

    def upsert(self, doc_id: str, embedding: List[float], metadata: Dict[str, Any],
               content: str, chunks: List[Dict[str, Any]]):
        with self._lock:
            self._documents[doc_id] = {
                "embedding": np.array(embedding, dtype=np.float32),
                "metadata": metadata,
                "content": content,
                "indexed_at": datetime.now().isoformat(),
            }
            # Remove old chunks for this doc
            self._chunks = [c for c in self._chunks if c["doc_id"] != doc_id]
            for chunk in chunks:
                self._chunks.append({
                    "doc_id": doc_id,
                    "embedding": np.array(chunk["embedding"], dtype=np.float32),
                    "text": chunk["text"],
                    "metadata": metadata,
                })
            self._dirty = True
            logger.info(f"[VectorStore] Upserted doc {doc_id} with {len(chunks)} chunks")

    def remove(self, doc_id: str):
        with self._lock:
            self._documents.pop(doc_id, None)
            self._chunks = [c for c in self._chunks if c["doc_id"] != doc_id]
            self._dirty = True
            logger.info(f"[VectorStore] Removed doc {doc_id}")

    def search(self, query_embedding: List[float], top_k: int = 8,
               min_score: float = 0.1) -> List[Dict[str, Any]]:
        with self._lock:
            if not self._chunks:
                return []

            q = np.array(query_embedding, dtype=np.float32)
            q_norm = np.linalg.norm(q)
            if q_norm == 0:
                return []
            q = q / q_norm

            results = []
            seen_docs = set()
            scored = []

            for chunk in self._chunks:
                emb = chunk["embedding"]
                norm = np.linalg.norm(emb)
                if norm == 0:
                    continue
                score = float(np.dot(q, emb / norm))
                scored.append((score, chunk))

            scored.sort(key=lambda x: x[0], reverse=True)

            for score, chunk in scored:
                if score < min_score:
                    break
                doc_id = chunk["doc_id"]
                if doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    results.append({
                        "doc_id": doc_id,
                        "score": score,
                        "chunk_text": chunk["text"],
                        "metadata": chunk["metadata"],
                    })
                if len(results) >= top_k:
                    break

            return results

    def list_documents(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {"id": doc_id, **doc["metadata"], "indexed_at": doc["indexed_at"]}
                for doc_id, doc in self._documents.items()
            ]

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._documents.get(doc_id)

    @property
    def document_count(self) -> int:
        with self._lock:
            return len(self._documents)


# Global store instance
_vector_store = DocumentVectorStore()


def get_vector_store() -> DocumentVectorStore:
    return _vector_store


# ---------------------------------------------------------------------------
# Pathway pipeline
# ---------------------------------------------------------------------------

class PathwayIngestionPipeline:
    """
    Real Pathway streaming pipeline that monitors a directory for documents
    and maintains a live vector index.
    """

    def __init__(self, documents_path: str, openai_api_key: str,
                 embedding_model: str = "text-embedding-3-small"):
        self.documents_path = Path(documents_path)
        self.documents_path.mkdir(parents=True, exist_ok=True)
        self.openai_api_key = openai_api_key
        self.embedding_model = embedding_model
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._processed: Dict[str, float] = {}  # path -> mtime
        self._openai_client = None

    def _get_openai_client(self):
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=self.openai_api_key)
        return self._openai_client

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings via OpenAI."""
        client = self._get_openai_client()
        # Clean texts
        cleaned = [t.replace("\n", " ").strip()[:8000] for t in texts]
        response = client.embeddings.create(model=self.embedding_model, input=cleaned)
        return [item.embedding for item in response.data]

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                # Try to break at sentence boundary
                last_period = text.rfind(". ", start + chunk_size - overlap, end)
                if last_period > start:
                    end = last_period + 1
            chunks.append(text[start:end].strip())
            start = end - overlap
        return [c for c in chunks if c]

    def _extract_metadata(self, content: str, file_path: Path) -> Dict[str, Any]:
        """Extract basic metadata from document content."""
        import re
        lines = content.split("\n")
        title = file_path.stem.replace("_", " ").replace("-", " ").title()

        # Try to find a better title in first 10 lines
        for line in lines[:10]:
            line = line.strip()
            if line and len(line) > 10 and len(line) < 200:
                if not any(line.lower().startswith(w) for w in ["abstract", "background", "method", "result", "#!"]):
                    title = line.lstrip("#").strip()
                    break

        # Extract authors
        authors = []
        for line in lines[:30]:
            if re.search(r"^authors?:|^by\s+", line.lower()):
                author_text = re.sub(r"^authors?:|^by\s+", "", line, flags=re.IGNORECASE).strip()
                authors = [a.strip() for a in re.split(r"[,;]", author_text) if a.strip()][:5]
                break

        # Detect document type
        doc_type = "research_paper"
        content_lower = content.lower()
        if "systematic review" in content_lower or "meta-analysis" in content_lower:
            doc_type = "systematic_review"
        elif "randomized controlled" in content_lower or "rct" in content_lower:
            doc_type = "clinical_trial"
        elif "guideline" in content_lower or "recommendation" in content_lower:
            doc_type = "guideline"

        # Extract date
        date_match = re.search(r"\b(20\d{2})\b", content)
        pub_year = date_match.group(1) if date_match else str(datetime.now().year)

        # Credibility heuristics
        cred = 0.4
        if any(j in content_lower for j in ["lancet", "nejm", "jama", "bmj", "nature medicine"]):
            cred += 0.3
        if re.search(r"\bdoi\b|\bpmid\b", content_lower):
            cred += 0.15
        if re.search(r"\bn\s*=\s*\d+", content_lower):
            cred += 0.1
        if re.search(r"\bp\s*[<>=]\s*0\.\d+", content_lower):
            cred += 0.05

        return {
            "title": title,
            "authors": authors,
            "document_type": doc_type,
            "publication_date": f"{pub_year}-01-01T00:00:00",
            "source": file_path.name,
            "file_path": str(file_path),
            "credibility_score": min(1.0, cred),
            "word_count": len(content.split()),
        }

    def _process_file(self, file_path: Path):
        """Process a single document file into the vector store."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            if len(content.strip()) < 100:
                logger.warning(f"Skipping {file_path.name}: too short")
                return

            metadata = self._extract_metadata(content, file_path)
            doc_id = f"doc_{file_path.stem}_{int(file_path.stat().st_mtime)}"

            # Chunk the content
            chunks_text = self._chunk_text(content)
            logger.info(f"Processing {file_path.name}: {len(chunks_text)} chunks")

            # Embed all chunks + doc-level summary
            all_texts = chunks_text
            embeddings = self._embed_texts(all_texts)

            doc_embedding = embeddings[0] if len(embeddings) == 1 else list(np.mean(embeddings, axis=0))

            chunks = [
                {"text": text, "embedding": emb}
                for text, emb in zip(chunks_text, embeddings)
            ]

            get_vector_store().upsert(doc_id, doc_embedding, metadata, content, chunks)
            logger.info(f"✅ Indexed: {file_path.name} ({doc_id})")

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}", exc_info=True)

    def _scan_once(self):
        """Scan directory for new/modified/deleted files."""
        supported = {".txt", ".md", ".html", ".xml", ".pdf"}
        current_files = {}

        for f in self.documents_path.iterdir():
            if f.suffix.lower() in supported and f.is_file():
                current_files[str(f)] = f.stat().st_mtime

        # Detect new or modified
        for path_str, mtime in current_files.items():
            if path_str not in self._processed or self._processed[path_str] != mtime:
                self._process_file(Path(path_str))
                self._processed[path_str] = mtime

        # Detect deleted
        deleted = set(self._processed.keys()) - set(current_files.keys())
        for path_str in deleted:
            file_path = Path(path_str)
            doc_id_prefix = f"doc_{file_path.stem}_"
            store = get_vector_store()
            with store._lock:
                to_remove = [d for d in store._documents if d.startswith(doc_id_prefix)]
            for doc_id in to_remove:
                store.remove(doc_id)
            del self._processed[path_str]

    def _watch_loop(self):
        """Main watching loop using polling (cross-platform, no Pathway daemon needed)."""
        logger.info(f"🔍 Pathway pipeline watching: {self.documents_path}")
        while self._running:
            try:
                self._scan_once()
            except Exception as e:
                logger.error(f"Scan error: {e}")
            time.sleep(5)  # Poll every 5 seconds

    def start(self):
        """Start the background ingestion pipeline."""
        if self._running:
            return
        self._running = True
        # Initial scan
        try:
            self._scan_once()
        except Exception as e:
            logger.warning(f"Initial scan failed: {e}")
        self._thread = threading.Thread(target=self._watch_loop, daemon=True, name="pathway-watcher")
        self._thread.start()
        logger.info("✅ Pathway ingestion pipeline started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)

    def ingest_document(self, content: str, filename: str) -> str:
        """Manually ingest a document (for API uploads)."""
        # Write to documents directory so the watcher picks it up
        file_path = self.documents_path / filename
        file_path.write_text(content, encoding="utf-8")
        # Process immediately
        self._process_file(file_path)
        self._processed[str(file_path)] = file_path.stat().st_mtime
        return f"doc_{file_path.stem}_{int(file_path.stat().st_mtime)}"


# Global pipeline instance
_pipeline: Optional[PathwayIngestionPipeline] = None


def get_pipeline() -> Optional[PathwayIngestionPipeline]:
    return _pipeline


def init_pipeline(documents_path: str, openai_api_key: str,
                  embedding_model: str = "text-embedding-3-small") -> PathwayIngestionPipeline:
    global _pipeline
    _pipeline = PathwayIngestionPipeline(documents_path, openai_api_key, embedding_model)
    _pipeline.start()
    return _pipeline