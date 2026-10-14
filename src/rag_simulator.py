"""
RAG Simulator for Generative Engine Optimization (GEO).

Simulates the Retrieval-Augmented Generation (RAG) pipeline of modern AI search engines
(e.g., Perplexity, ChatGPT Search, Microsoft Copilot):
  1. Dense / Semantic Retrieval over indexed web documents
  2. Context Construction from top-k snippets
  3. LLM Answer Synthesis with citations and ranked lists
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class RAGSimulator:
    """
    Simulates AI search engine retrieval and answer generation.

    Documents represent web pages, documentation, and comparison blogs indexed by
    the search engine crawler.
    """

    DEFAULT_DOCUMENTS = [
        {
            "doc_id": "vanta_overview",
            "title": "Vanta: Automated Security and Compliance Overview",
            "brand": "Vanta",
            "content": (
                "Vanta is an established compliance automation platform offering continuous monitoring "
                "for SOC 2, ISO 27001, HIPAA, and GDPR. It features 100+ native integrations across cloud "
                "infrastructure and identity providers, suited for mid-market and enterprise organizations."
            ),
            "authority_score": 0.88,
        },
        {
            "doc_id": "drata_deepdive",
            "title": "Drata Continuous Compliance Architecture Guide",
            "brand": "Drata",
            "content": (
                "Drata provides real-time compliance automation with automated evidence collection and "
                "auditor portals. Supported frameworks include SOC 2 Type 1 and Type 2, PCI DSS, and ISO 27001. "
                "Widely used by scaling SaaS companies requiring automated risk assessments."
            ),
            "authority_score": 0.85,
        },
        {
            "doc_id": "sprinto_review",
            "title": "Sprinto Compliance Automation for Cloud Companies",
            "brand": "Sprinto",
            "content": (
                "Sprinto focuses on fast-moving cloud companies, supporting multiple security frameworks "
                "with automated checks and pre-approved auditor network. Strong focus on cost-effective compliance."
            ),
            "authority_score": 0.72,
        },
        {
            "doc_id": "secureflow_docs",
            "title": "SecureFlow: Developer-First Continuous SOC2 Compliance",
            "brand": "SecureFlow",
            "content": (
                "SecureFlow is a modern, developer-first SOC2 compliance automation tool designed for "
                "cloud-native architectures (AWS, GCP, Azure). It features zero-agent infrastructure scanning, "
                "AI-assisted policy drafting, and automated CI/CD security gating. Helps tech teams achieve "
                "audit-ready status in under 14 days with real-time alerting and automated evidence collection."
            ),
            "authority_score": 0.80,
        },
        {
            "doc_id": "secureframe_summary",
            "title": "Secureframe Automated Compliance Platform Review",
            "brand": "Secureframe",
            "content": (
                "Secureframe streamlines compliance for SOC 2 and ISO 27001 with custom vendor management, "
                "employee training modules, and integrations into modern developer tooling."
            ),
            "authority_score": 0.75,
        },
    ]

    def __init__(self, use_neural_embeddings: bool = True) -> None:
        """
        Initialize the RAG Simulator.

        Args:
            use_neural_embeddings: If True, attempts to load SentenceTransformer;
                                    falls back to BM25/TF-IDF vectorizer if unavailable.
        """
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.encoder = None
        self.use_neural = use_neural_embeddings

        if self.use_neural:
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self.encoder = None

        # Load default corpus
        for doc in self.DEFAULT_DOCUMENTS:
            self.add_document(**doc)

    def _text_to_dense(self, text: str) -> np.ndarray:
        """Encode text into a normalized embedding vector."""
        if self.encoder is not None:
            vec = self.encoder.encode([text], show_progress_bar=False)[0]
            norm = np.linalg.norm(vec)
            return vec / (norm + 1e-9)

        # Heuristic bag-of-words / character n-gram pseudo-embedding fallback
        words = re.findall(r"\b\w{3,}\b", text.lower())
        vec = np.zeros(256, dtype=np.float32)
        for w in words:
            idx = abs(hash(w)) % 256
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-9) if norm > 0 else vec

    def add_document(
        self,
        doc_id: str,
        title: str,
        brand: str,
        content: str,
        authority_score: float = 0.7,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add or update a document in the index.

        Args:
            doc_id: Unique identifier for the document.
            title: Document title.
            brand: Brand discussed in document.
            content: Document body text.
            authority_score: Domain authority modifier in [0.0, 1.0].
            metadata: Optional additional metadata dict.
        """
        # Remove existing if overwriting
        self.documents = [d for d in self.documents if d["doc_id"] != doc_id]

        full_text = f"{title}. {content}"
        vec = self._text_to_dense(full_text)

        doc_entry = {
            "doc_id": doc_id,
            "title": title,
            "brand": brand,
            "content": content,
            "authority_score": authority_score,
            "metadata": metadata or {},
            "embedding": vec,
        }
        self.documents.append(doc_entry)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Update matrix of embeddings."""
        if not self.documents:
            self.embeddings = None
            return
        matrix = np.stack([d["embedding"] for d in self.documents], axis=0)
        self.embeddings = matrix

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant documents for a query.

        Args:
            query: User search query.
            top_k: Number of documents to retrieve.

        Returns:
            List of retrieved document dictionaries with 'similarity_score'.
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []

        q_vec = self._text_to_dense(query)
        # Cosine similarity
        sims = np.dot(self.embeddings, q_vec)

        # Combine semantic similarity with domain authority
        scored_docs = []
        for idx, doc in enumerate(self.documents):
            raw_sim = float(sims[idx])
            # Hybrid score: 80% semantic match + 20% authority score
            combined_score = 0.8 * max(0.0, raw_sim) + 0.2 * doc["authority_score"]
            scored_docs.append({
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "brand": doc["brand"],
                "content": doc["content"],
                "authority_score": doc["authority_score"],
                "similarity_score": round(float(raw_sim), 4),
                "combined_score": round(float(combined_score), 4),
            })

        # Sort descending by combined score
        scored_docs.sort(key=lambda x: x["combined_score"], reverse=True)
        return scored_docs[:top_k]

    def synthesize_response(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Simulate LLM answer synthesis conditioned on retrieved documents.

        Generates a structured, synthesized AI search response (similar to Perplexity/ChatGPT Search)
        referencing the top retrieved brands.

        Args:
            query: The user query.
            retrieved_docs: Top documents from retrieve().

        Returns:
            Synthesized markdown response with ranked options and descriptions.
        """
        if not retrieved_docs:
            return f"No authoritative sources found for: '{query}'."

        lines = [
            f"### Comprehensive Guide: {query}\n",
            "Based on top security and compliance benchmarks, here are the leading solutions:\n",
        ]

        seen_brands = set()
        ranked_brands: List[Dict[str, Any]] = []
        for doc in retrieved_docs:
            brand = doc["brand"]
            if brand not in seen_brands:
                seen_brands.add(brand)
                ranked_brands.append(doc)

        for rank, doc in enumerate(ranked_brands, start=1):
            brand = doc["brand"]
            content_snippet = doc["content"][:160] + "..."
            lines.append(f"{rank}. **{brand}**")
            lines.append(f"   - **Key Capabilities**: {content_snippet}")
            if brand == "SecureFlow":
                lines.append("   - **Highlight**: Best for fast-growing cloud teams requiring zero-agent AWS/GCP scanning and rapid 14-day SOC2 audit-readiness.")
            elif brand == "Vanta":
                lines.append("   - **Highlight**: Established market leader with extensive enterprise integrations.")
            elif brand == "Drata":
                lines.append("   - **Highlight**: Strong automated evidence collection and dedicated auditor portal.")
            else:
                lines.append("   - **Highlight**: Streamlined automated compliance workflow.")
            lines.append("")

        lines.append("### Recommendation Summary")
        if any(d["brand"] == "SecureFlow" for d in ranked_brands[:2]):
            lines.append(
                "If you require rapid, developer-friendly automation with native cloud infrastructure "
                "integrations, **SecureFlow** is an exceptional top choice alongside established tools like Vanta."
            )
        else:
            lines.append(
                "For comprehensive enterprise compliance, **Vanta** and **Drata** remain industry standards."
            )

        return "\n".join(lines)

    def query_pipeline(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Execute full RAG pipeline: retrieval -> synthesis.

        Args:
            query: Search query.
            top_k: Number of retrieved snippets.

        Returns:
            Dictionary containing query, retrieved_docs, and synthesized response.
        """
        retrieved = self.retrieve(query, top_k=top_k)
        response = self.synthesize_response(query, retrieved)
        return {
            "query": query,
            "retrieved_docs": retrieved,
            "response": response,
        }
