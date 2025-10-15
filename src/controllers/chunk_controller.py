from flask import Blueprint, jsonify, request
from rank_bm25 import BM25Okapi
import re
import time

from src.impl.logger import get_logger

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


class ChunksController:
    blueprint = Blueprint('chunks', __name__)

    def __init__(self, qdrant_db, embedder, mysql_db=None):
        self.logger = get_logger(__name__)
        self.logger.info("Initializing ChunksController")

        self.qdrant_db = qdrant_db
        self.embedder = embedder
        self.mysql_db = mysql_db

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
        self.logger.info("ChunksController routes registered")

    @staticmethod
    def _to_list(vec):
        try:
            return vec.tolist()
        except AttributeError:
            return list(vec)

    def get_chunks_semantic(self, text, limit=None):
        if not text:
            raise ValueError("Missing mandatory 'text' parameter")
        if limit is not None and (not isinstance(limit, int) or limit <= 0):
            raise ValueError("'limit' must be a positive integer")

        self.logger.debug(f"[semantic] Embedding text (len={len(text)}) limit={limit}")
        embedding = self.embedder.embed(text)
        embedding_list = self._to_list(embedding)

        start = time.time()
        similar_chunks = (
            self.qdrant_db.get_similar(embedding_list, top_k=limit)
            if limit else
            self.qdrant_db.get_similar(embedding_list)
        )
        elapsed = (time.time() - start) * 1000.0
        self.logger.info(f"[semantic] Retrieved {len(similar_chunks)} candidates from vector DB in {elapsed:.1f} ms")

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

        self.logger.debug(f"[semantic] Returning {len(results)} results")
        return results

    def rerank_chunks_bm25(self, text, limit=20):
        if not text:
            raise ValueError("Missing mandatory 'text' parameter")
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("'limit' must be a positive integer")

        self.logger.debug(f"[bm25] Start rerank (len={len(text)}) prefilter limit={limit}")

        embedding = self.embedder.embed(text)
        embedding_list = self._to_list(embedding)

        t0 = time.time()
        similar_chunks = self.qdrant_db.get_similar(embedding_list, top_k=limit)
        t1 = time.time()
        self.logger.info(f"[bm25] Prefilter fetched {len(similar_chunks)} chunks in {(t1 - t0)*1000:.1f} ms")

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
            self.logger.warning("[bm25] No documents available after prefilter; returning empty list")
            return []

        bm25 = BM25Okapi(documents)
        tokenized_query = safe_tokenize(text.lower())
        bm25_scores = bm25.get_scores(tokenized_query)

        for i, score in enumerate(bm25_scores):
            chunk_map[i]["bm25_score"] = float(score)

        reranked_chunks = sorted(
            chunk_map, key=lambda x: x.get("bm25_score", 0.0), reverse=True
        )
        self.logger.debug(f"[bm25] Reranked {len(reranked_chunks)} chunks")
        return reranked_chunks

    def get_chunks_semantic_route(self):
        data = request.get_json(silent=True) or {}
        text = data.get("text")
        limit = data.get("limit")

        try:
            limit = int(limit) if limit is not None else None
        except (TypeError, ValueError):
            self.logger.warning("[route:semantic] Invalid 'limit' type; must be integer")
            return jsonify({"error": "limit must be an integer"}), 400

        try:
            self.logger.info(f"[route:semantic] Request received len(text)={len(text) if text else 0} limit={limit}")
            results = self.get_chunks_semantic(text=text, limit=limit)
            self.logger.info(f"[route:semantic] Returning {len(results)} results")
            return jsonify({"results": results})
        except ValueError as ve:
            self.logger.warning(f"[route:semantic] 4xx: {ve}")
            return jsonify({"error": str(ve)}), 400
        except Exception as e:
            self.logger.error("[route:semantic] 5xx unexpected error", exc_info=True)
            return jsonify({"error": "Internal server error", "details": str(e)}), 500


    def rerank_chunks_bm25_route(self):
        data = request.get_json(silent=True) or {}
        text = data.get("text")
        limit = data.get("limit", 20)

        try:
            limit = int(limit)
        except (TypeError, ValueError):
            self.logger.warning("[route:bm25] Invalid 'limit' type; must be integer")
            return jsonify({"error": "limit must be an integer"}), 400

        try:
            self.logger.info(f"[route:bm25] Request received len(text)={len(text) if text else 0} limit={limit}")
            results = self.rerank_chunks_bm25(text=text, limit=limit)
            self.logger.info(f"[route:bm25] Returning {len(results)} results")
            return jsonify({"results": results})
        except ValueError as ve:
            self.logger.warning(f"[route:bm25] 4xx: {ve}")
            return jsonify({"error": str(ve)}), 400
        except Exception as e:
            self.logger.error("[route:bm25] 5xx unexpected error", exc_info=True)
            return jsonify({"error": "Internal server error", "details": str(e)}), 500
