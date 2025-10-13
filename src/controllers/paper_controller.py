# src/controllers/papers_controller.py
from flask import Blueprint, jsonify, request
from rank_bm25 import BM25Okapi
import re

# --- Robust tokenization: prefer NLTK, but fall back if data isn't available ---
try:
    from nltk.tokenize import word_tokenize as nltk_word_tokenize
    def safe_tokenize(text: str):
        try:
            return nltk_word_tokenize(text)
        except LookupError:
            return re.findall(r"\b\w+\b", text)
except Exception:
    def safe_tokenize(text: str):
        return re.findall(r"\b\w+\b", text)


class PapersController:
    blueprint = Blueprint('papers', __name__)

    def __init__(self, qdrant_db, embedder, mysql_db):
        """
        qdrant_db: QdrantDatabase configured for the *papers* collection.
                   Must expose get_similar(embedding_list, top_k)
                   and return points whose payload includes at least {"arxiv_id": ...}
        embedder : object with embed(text: str) -> 1D vector (list or numpy array)
        mysql_db : MySQLDatabase providing get_paper_by_arxiv_id(arxiv_id) -> Paper or None
        """
        self.qdrant_db = qdrant_db
        self.embedder = embedder
        self.mysql_db = mysql_db

        # Register routes
        PapersController.blueprint.add_url_rule(
            '/papers/semantic',
            view_func=self.get_papers_semantic_route,
            methods=['POST']
        )
        PapersController.blueprint.add_url_rule(
            '/papers/rerank-bm25',
            view_func=self.rerank_papers_bm25_route,
            methods=['POST']
        )

    # -------- Helpers --------
    @staticmethod
    def _to_list(vec):
        try:
            return vec.tolist()
        except AttributeError:
            return list(vec)

    @staticmethod
    def _paper_to_dict(paper_obj):
        if not paper_obj:
            return None
        # Adjust attributes to match your SQLAlchemy Paper model fields
        return {
            "arxiv_id": getattr(paper_obj, "arxiv_id", None),
            "title": getattr(paper_obj, "title", None),
            "author": getattr(paper_obj, "author", None),
            "subject": getattr(paper_obj, "subject", None),
            "keywords": getattr(paper_obj, "keywords", None),
            "creation_date": getattr(paper_obj, "creation_date", None),
            "modification_date": getattr(paper_obj, "modification_date", None),
        }

    @staticmethod
    def _dedup_by_arxiv(items):
        """Keep first occurrence per arxiv_id while preserving order."""
        seen = set()
        unique = []
        for it in items:
            aid = it.get("arxiv_id")
            if aid and aid not in seen:
                seen.add(aid)
                unique.append(it)
        return unique

    # -------- Core logic (non-route) --------
    def get_papers_semantic(self, text, limit=10):
        if not text:
            raise ValueError("Missing mandatory 'text' parameter")
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("'limit' must be a positive integer")

        # Oversample to maintain 'limit' after dedup
        top_k = max(limit * 3, limit)

        # Embed and query Qdrant (papers collection)
        embedding = self.embedder.embed(text)
        embedding_list = self._to_list(embedding)
        similar = self.qdrant_db.get_similar(embedding_list, top_k=top_k)

        # Build candidates (enrich with MySQL metadata if available)
        candidates = []
        for pt in sorted(similar, key=lambda x: x.score, reverse=True):
            payload = getattr(pt, "payload", {}) or {}
            arxiv_id = payload.get("arxiv_id") or getattr(pt, "id", None)
            paper_meta = self.mysql_db.get_paper_by_arxiv_id(arxiv_id) if arxiv_id else None
            meta = self._paper_to_dict(paper_meta) if paper_meta else {"arxiv_id": arxiv_id}

            candidates.append({
                **meta,
                "score": float(getattr(pt, "score", 0.0)),
            })

        # Deduplicate by arxiv_id and cut to limit
        results = self._dedup_by_arxiv(candidates)[:limit]
        return results

    def rerank_papers_bm25(self, text, limit=20):
        """
        1) Semantic prefilter via vector DB (oversample top_k)
        2) BM25 re-rank using full paper text from MySQL
        3) Deduplicate by arxiv_id and return top 'limit'
        """
        if not text:
            raise ValueError("Missing mandatory 'text' parameter")
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("'limit' must be a positive integer")

        top_k = max(limit * 3, limit)

        # 1) semantic prefilter
        embedding = self.embedder.embed(text)
        embedding_list = self._to_list(embedding)
        candidates = self.qdrant_db.get_similar(embedding_list, top_k=top_k)

        # 2) build BM25 corpus using full paper text from MySQL
        documents = []
        paper_map = []
        for pt in candidates:
            payload = getattr(pt, "payload", {}) or {}
            arxiv_id = payload.get("arxiv_id") or getattr(pt, "id", None)
            if not arxiv_id:
                continue

            paper_obj = self.mysql_db.get_paper_by_arxiv_id(arxiv_id)
            full_text = getattr(paper_obj, "text", None) if paper_obj else None
            if not full_text:
                continue

            documents.append(safe_tokenize(full_text.lower()))
            paper_map.append({
                "arxiv_id": arxiv_id,
                "original_score": float(getattr(pt, "score", 0.0)),
                "title": getattr(paper_obj, "title", None) if paper_obj else None,
                # No snippet included per request
                "meta": self._paper_to_dict(paper_obj),
            })

        if not documents:
            return []

        bm25 = BM25Okapi(documents)
        tokenized_query = safe_tokenize(text.lower())
        bm25_scores = bm25.get_scores(tokenized_query)

        for i, score in enumerate(bm25_scores):
            paper_map[i]["bm25_score"] = float(score)

        reranked = sorted(paper_map, key=lambda x: x.get("bm25_score", 0.0), reverse=True)

        # 3) Deduplicate by arxiv_id and cut to limit
        deduped = self._dedup_by_arxiv(reranked)[:limit]
        return deduped

    # -------- HTTP routes --------
    def get_papers_semantic_route(self):
        """
        Expects JSON: { "text": "...", "limit": <int, optional default 10> }
        """
        data = request.get_json(silent=True) or {}
        text = data.get("text")
        limit = data.get("limit", 10)

        try:
            limit = int(limit)
        except (TypeError, ValueError):
            return jsonify({"error": "limit must be an integer"}), 400

        try:
            results = self.get_papers_semantic(text=text, limit=limit)
            return jsonify({"results": results})
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400
        except Exception as e:
            return jsonify({"error": "Internal server error", "details": str(e)}), 500

    def rerank_papers_bm25_route(self):
        """
        Expects JSON: { "text": "...", "limit": <int, optional default 20> }
        """
        data = request.get_json(silent=True) or {}
        text = data.get("text")
        limit = data.get("limit", 20)

        try:
            limit = int(limit)
        except (TypeError, ValueError):
            return jsonify({"error": "limit must be an integer"}), 400

        try:
            results = self.rerank_papers_bm25(text=text, limit=limit)
            return jsonify({"results": results})
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400
        except Exception as e:
            return jsonify({"error": "Internal server error", "details": str(e)}), 500
