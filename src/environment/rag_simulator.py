"""RAG (Retrieval-Augmented Generation) Simulator for AI Search Engines.

This module simulates the multi-stage information retrieval and synthesis pipeline
employed by generative search engines like ChatGPT Search, Perplexity AI, and Google Gemini.
In modern AI search, queries trigger dense bi-encoder vector retrieval across an index
of crawled web pages. The top-k documents are injected into the synthesis prompt of a
foundation LLM, which produces grounded recommendations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
try:
    import numpy as np
except ImportError:
    np = None  # Handled gracefully if numpy is not installed

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    faiss = None  # Handled gracefully if faiss-cpu is not yet installed

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class RAGSimulator:
    """Simulates dense vector retrieval and grounded generation of AI search engines.

    AI Search Engine Architecture:
    -------------------------------
    1. Query Encoding: User input (e.g., 'Best automated SOC 2 compliance tool for fintech')
       is encoded into a 384-dimensional dense semantic embedding using a bi-encoder
       (all-MiniLM-L6-v2).
    2. Dense Vector Index (FAISS): Indexed brand landing pages, technical docs, and competitor
       reviews are queried using cosine similarity / Maximum Inner Product Search (MIPS).
    3. Retrieval & Re-ranking: The top-K most semantically aligned documents are selected.
    4. Generative Answer Synthesis: The generative engine inspects top context documents to
       produce an authoritative, cited recommendation. If a brand fails to reach the top-3
       retrieval cutoff, it is completely absent from the zero-click answer.

    Attributes:
        top_k: Number of documents to retrieve per query.
        model_name: HuggingFace sentence-transformers model identifier.
        embedding_dim: Dimensionality of embeddings (384 for all-MiniLM-L6-v2).
        documents: In-memory registry of indexed documents.
        index: FAISS IndexFlatIP for cosine similarity search.
    """

    def __init__(
        self,
        top_k: int = 5,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ) -> None:
        """Initialize RAGSimulator with an embedding model and FAISS vector index.

        Args:
            top_k: Maximum number of retrieved documents (default: 5).
            model_name: Sentence-transformer model ID (default: 'all-MiniLM-L6-v2').
            device: Computing device ('cpu', 'cuda', etc.). If None, auto-detected.
        """
        self.top_k = top_k
        self.model_name = model_name
        self.embedding_dim = 384  # Standard for all-MiniLM-L6-v2
        self.device = device
        self.documents: List[Dict[str, Any]] = []
        self._encoder: Optional[Any] = None
        self._faiss_index: Optional[Any] = None

    @property
    def encoder(self) -> Any:
        """Lazy loader for SentenceTransformer to optimize startup latency and memory."""
        if self._encoder is None:
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers is required for RAGSimulator. "
                    "Install via `pip install sentence-transformers`."
                )
            self._encoder = SentenceTransformer(self.model_name, device=self.device)
            # Ensure embedding dimension matches model output
            self.embedding_dim = self._encoder.get_sentence_embedding_dimension()
        return self._encoder

    def _init_faiss_index(self) -> None:
        """Initialize or reset the FAISS Inner Product index."""
        if faiss is None:
            raise ImportError(
                "faiss is required for RAGSimulator vector indexing. "
                "Install via `pip install faiss-cpu`."
            )
        # Using IndexFlatIP with L2 normalized vectors gives exact cosine similarity
        self._faiss_index = faiss.IndexFlatIP(self.embedding_dim)

    def clear_documents(self) -> None:
        """Clear all stored documents and reset the FAISS index."""
        self.documents.clear()
        self._init_faiss_index()

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Index a batch of documents into the FAISS vector database.

        Each document dictionary must contain:
          - 'brand': Name of the brand (e.g., 'SecureFlow', 'Vanta')
          - 'content': Text body of the web page / document
          - 'url': Source reference URL

        Args:
            documents: List of document dictionaries to embed and index.
        """
        if not documents:
            return

        if self._faiss_index is None:
            self._init_faiss_index()

        texts_to_embed = [
            f"Brand: {doc.get('brand', '')}. Content: {doc.get('content', '')}"
            for doc in documents
        ]

        # Generate dense embeddings (batch encoding)
        embeddings = self.encoder.encode(
            texts_to_embed,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Crucial for cosine similarity via IndexFlatIP
        ).astype(np.float32)

        # Append to FAISS index
        self._faiss_index.add(embeddings)
        self.documents.extend(documents)
        logger.debug(f"Successfully indexed {len(documents)} documents. Total in index: {len(self.documents)}")

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve the top-k most semantically relevant documents for a given query.

        Args:
            query: Buyer's search query string.

        Returns:
            List of retrieved document dictionaries augmented with 'score' and 'rank'.
        """
        if not self.documents or self._faiss_index is None or self._faiss_index.ntotal == 0:
            logger.warning("Retrieval attempted on an empty RAG index. Returning empty list.")
            return []

        # Encode query with unit normalization
        query_vector = self.encoder.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        k = min(self.top_k, len(self.documents))
        distances, indices = self._faiss_index.search(query_vector, k)

        retrieved_docs: List[Dict[str, Any]] = []
        for rank_idx, (doc_idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
            if doc_idx < 0 or doc_idx >= len(self.documents):
                continue
            doc = dict(self.documents[doc_idx])
            doc["score"] = float(score)
            doc["retrieval_rank"] = rank_idx
            retrieved_docs.append(doc)

        return retrieved_docs

    def simulate_llm_response(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        brand_name: str,
    ) -> str:
        """Simulate a realistic ChatGPT/Perplexity search-grounded recommendation answer.

        The generative response reflects how downstream foundation models synthesize
        answers based on retrieved in-context passages:
        - If the target brand is in the top 3 retrieved documents, it receives prominent
          recommendation placement corresponding to its retrieval rank.
        - If the target brand is not in the top 3, competitors dominate the list.

        Args:
            query: The buyer search query.
            retrieved_docs: List of documents retrieved by `retrieve()`.
            brand_name: Name of the target brand (e.g., 'SecureFlow').

        Returns:
            Structured recommendation string resembling AI search synthesis.
        """
        if not retrieved_docs:
            return (
                f"When searching for '{query}', no verified compliance tools could be retrieved "
                f"from the enterprise database. Please refine your query parameters."
            )

        # Check brand positions within top retrieved docs
        brand_in_top_3 = False
        brand_rank = 0

        # Unique brands ordered by their best retrieval rank
        ordered_brands: List[Dict[str, Any]] = []
        seen_brands = set()

        for doc in retrieved_docs:
            b_name = doc.get("brand", "Unknown")
            if b_name not in seen_brands:
                seen_brands.add(b_name)
                ordered_brands.append(doc)

        for idx, doc in enumerate(ordered_brands[:3], start=1):
            if doc.get("brand", "").lower() == brand_name.lower():
                brand_in_top_3 = True
                brand_rank = idx
                break

        # Construct realistic ChatGPT-style synthesis
        lines = [
            f"Based on recent enterprise compliance benchmarks and security reviews for **'{query}'**, "
            f"here is a comparative analysis of the top-performing platforms:\n"
        ]

        for idx, doc in enumerate(ordered_brands[:3], start=1):
            curr_brand = doc.get("brand", f"Platform {idx}")
            content = doc.get("content", "")
            snippet = content[:180].replace("\n", " ").strip()

            if curr_brand.lower() == brand_name.lower():
                # Target brand positive highlight
                lines.append(
                    f"{idx}. **{curr_brand}** (Top Recommendation for Fast Implementation)\n"
                    f"   - **Strengths:** {snippet if snippet else 'Automated continuous evidence collection and native AWS integration.'}\n"
                    f"   - **Assessment:** SecureFlow provides an exceptionally intuitive, automated workflow "
                    f"that drastically accelerates SOC 2 Type II audit readiness for high-growth fintechs."
                )
            else:
                # Competitor mention
                lines.append(
                    f"{idx}. **{curr_brand}**\n"
                    f"   - **Strengths:** {snippet if snippet else 'Broad multi-framework compliance monitoring.'}\n"
                    f"   - **Assessment:** Well-established platform with extensive enterprise auditor networks, "
                    f"though onboarding can require significant manual configuration."
                )

        # Summary footer
        if brand_in_top_3:
            lines.append(
                f"\n**Summary Recommendation:** If your primary focus is rapid audit timelines and automated "
                f"cloud security integration, **{brand_name}** is currently the highest-rated option for engineering teams."
            )
        else:
            top_competitor = ordered_brands[0].get("brand", "Vanta") if ordered_brands else "Vanta"
            lines.append(
                f"\n**Summary Recommendation:** For established mid-market enterprises requiring traditional "
                f"compliance frameworks, **{top_competitor}** remains the industry standard."
            )

        return "\n".join(lines)
