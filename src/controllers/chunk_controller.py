from flask import Blueprint, jsonify, request
from rank_bm25 import BM25Okapi
import re

# --- Robust tokenization: prefer NLTK, but fall back if data isn't available ---
try:
    from nltk.tokenize import word_tokenize as nltk_word_tokenize
    def safe_tokenize(text: str):
        try:
            # NLTK may require 'punkt' or 'punkt_tab' data; catch LookupError
            return nltk_word_tokenize(text)
        except LookupError:
            # Fallback: simple regex-based tokenization
            return re.findall(r"\b\w+\b", text)
except Exception:
    # If NLTK isn't installed at all
    def safe_tokenize(text: str):
        return re.findall(r"\b\w+\b", text)


class ChunksController:
    blueprint = Blueprint('chunks', __name__)

    def __init__(self, qdrant_db, embedder, mysql_db=None):
        """
        qdrant_db: must expose get_similar(embedding_list, top_k: int|None)
        embedder: must expose embed(text: str) -> 1D vector (list or numpy array)
        mysql_db: kept for parity with your other controllers (unused here)
        """
        self.qdrant_db = qdrant_db
        self.embedder = embedder
        self.mysql_db = mysql_db

        # Register routes
        ChunksController.blueprint.add_url_rule(
            '/chunks/semantic',
            view_func=self.get_chunks_semantic_route,
            methods=['POST']
        )
        ChunksController.blueprint.add_url_rule(
            '/chunks/rerank-bm25',
            view_func=self.rerank_chunks_bm25_route,
            methods=['POST']
        )

    # -------- Helpers --------
    @staticmethod
    def _to_list(vec):
        try:
            return vec.tolist()
        except AttributeError:
            return list(vec)

    # -------- Core logic (non-route) --------
    def get_chunks_semantic(self, text, limit=None):
        if not text:
            raise ValueError("Missing mandatory 'text' parameter")
        if limit is not None and (not isinstance(limit, int) or limit <= 0):
            raise ValueError("'limit' must be a positive integer")

        embedding = self.embedder.embed(text)
        embedding_list = self._to_list(embedding)

        similar_chunks = (
            self.qdrant_db.get_similar(embedding_list, top_k=limit)
            if limit else
            self.qdrant_db.get_similar(embedding_list)
        )

        sorted_chunks = sorted(similar_chunks, key=lambda x: x.score, reverse=True)

        results = []
        for chunk in sorted_chunks:
            payload = getattr(chunk, "payload", {}) or {}
            results.append({
                "id": getattr(chunk, "id", None),
                "score": float(getattr(chunk, "score", 0.0)),
                "text": payload.get("text"),
                "arxiv_id": payload.get("arxiv_id"),
            })
        return results

    def rerank_chunks_bm25(self, text, limit=20):
        if not text:
            raise ValueError("Missing mandatory 'text' parameter")
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("'limit' must be a positive integer")

        # 1) semantic prefilter from vector DB
        embedding = self.embedder.embed(text)
        embedding_list = self._to_list(embedding)
        similar_chunks = self.qdrant_db.get_similar(embedding_list, top_k=limit)

        # 2) build corpus for BM25
        documents = []
        chunk_map = []
        for chunk in similar_chunks:
            payload = getattr(chunk, "payload", {}) or {}
            chunk_text = payload.get("text", "")
            if chunk_text:
                documents.append(safe_tokenize(chunk_text.lower()))
                chunk_map.append({
                    "id": getattr(chunk, "id", None),
                    "original_score": float(getattr(chunk, "score", 0.0)),
                    "text": chunk_text,
                    "arxiv_id": payload.get("arxiv_id"),
                })

        if not documents:
            return []

        # 3) BM25 re-rank
        bm25 = BM25Okapi(documents)
        tokenized_query = safe_tokenize(text.lower())
        bm25_scores = bm25.get_scores(tokenized_query)

        for i, score in enumerate(bm25_scores):
            chunk_map[i]["bm25_score"] = float(score)

        reranked_chunks = sorted(
            chunk_map, key=lambda x: x.get("bm25_score", 0.0), reverse=True
        )
        return reranked_chunks

    # -------- HTTP routes --------
    def get_chunks_semantic_route(self):
        """
        Expects JSON: { "text": "...", "limit": <int, optional> }
        """
        data = request.get_json(silent=True) or {}
        text = data.get("text")
        limit = data.get("limit")

        try:
            limit = int(limit) if limit is not None else None
        except (TypeError, ValueError):
            return jsonify({"error": "limit must be an integer"}), 400

        try:
            results = self.get_chunks_semantic(text=text, limit=limit)
            return jsonify({"results": results})
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400
        except Exception as e:
            return jsonify({"error": "Internal server error", "details": str(e)}), 500

    def rerank_chunks_bm25_route(self):
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
            results = self.rerank_chunks_bm25(text=text, limit=limit)
            return jsonify({"results": results})
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400
        except Exception as e:
            return jsonify({"error": "Internal server error", "details": str(e)}), 500
