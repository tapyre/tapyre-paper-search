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


class PapersController:
    blueprint = Blueprint('papers', __name__)

    def __init__(self, qdrant_db, embedder, mysql_db):
        self.logger = get_logger(__name__)
        self.logger.info("Initializing PapersController")

        self.qdrant_db = qdrant_db
        self.embedder = embedder
        self.mysql_db = mysql_db

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
        self.logger.info("PapersController routes registered")

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
        seen = set()
        unique = []
        for it in items:
            aid = it.get("arxiv_id")
            if aid and aid not in seen:
                seen.add(aid)
                unique.append(it)
        return unique

    def get_papers_semantic(self, text, limit=10):
        if not text:
            raise ValueError("Missing mandatory 'text' parameter")
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("'limit' must be a positive integer")

        self.logger.debug(f"[semantic] Query received len(text)={len(text)} limit={limit}")

        top_k = max(limit * 3, limit)
        t0 = time.time()

        embedding = self.embedder.embed(text)
        embedding_list = self._to_list(embedding)
        similar = self.qdrant_db.get_similar(embedding_list, top_k=top_k)
        t1 = time.time()
        self.logger.info(f"[semantic] Retrieved {len(similar)} candidates from vector DB in {(t1 - t0)*1000:.1f} ms")

        candidates = []
        for pt in sorted(similar, key=lambda x: x.score, reverse=True):
            payload = getattr(pt, "payload", {}) or {}
            arxiv_id = payload.get("arxiv_id") or getattr(pt, "id", None)
            paper_meta = self.mysql_db.get_paper_by_arxiv_id(arxiv_id) if arxiv_id else None
            meta = self._paper_to_dict(paper_meta) if paper_meta else {"arxiv_id": arxiv_id}
            candidates.append({**meta, "score": float(getattr(pt, "score", 0.0))})

        results = self._dedup_by_arxiv(candidates)[:limit]
        self.logger.debug(f"[semantic] Returning {len(results)} results after dedup")
        return results

    def rerank_papers_bm25(self, text, limit=20):
        if not text:
            raise ValueError("Missing mandatory 'text' parameter")
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("'limit' must be a positive integer")

        self.logger.debug(f"[bm25] Rerank request len(text)={len(text)} limit={limit}")

        top_k = max(limit * 3, limit)
        t0 = time.time()

        embedding = self.embedder.embed(text)
        embedding_list = self._to_list(embedding)
        candidates = self.qdrant_db.get_similar(embedding_list, top_k=top_k)
        t1 = time.time()
        self.logger.info(f"[bm25] Prefilter fetched {len(candidates)} candidates in {(t1 - t0)*1000:.1f} ms")

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
                "title": getattr(paper_obj, "title", None),
                "meta": self._paper_to_dict(paper_obj),
            })

        if not documents:
            self.logger.warning("[bm25] No valid documents found for rerank; returning empty list")
            return []

        bm25 = BM25Okapi(documents)
        tokenized_query = safe_tokenize(text.lower())
        bm25_scores = bm25.get_scores(tokenized_query)

        for i, score in enumerate(bm25_scores):
            paper_map[i]["bm25_score"] = float(score)

        reranked = sorted(paper_map, key=lambda x: x.get("bm25_score", 0.0), reverse=True)
        deduped = self._dedup_by_arxiv(reranked)[:limit]
        self.logger.debug(f"[bm25] Returning {len(deduped)} reranked results")
        return deduped

    def get_papers_semantic_route(self):
        data = request.get_json(silent=True) or {}
        text = data.get("text")
        limit = data.get("limit", 10)

        try:
            limit = int(limit)
        except (TypeError, ValueError):
            self.logger.warning("[route:semantic] Invalid 'limit' type; must be integer")
            return jsonify({"error": "limit must be an integer"}), 400

        try:
            self.logger.info(f"[route:semantic] Request len(text)={len(text) if text else 0} limit={limit}")
            results = self.get_papers_semantic(text=text, limit=limit)
            self.logger.info(f"[route:semantic] Returning {len(results)} results")
            return jsonify({"results": results})
        except ValueError as ve:
            self.logger.warning(f"[route:semantic] 4xx: {ve}")
            return jsonify({"error": str(ve)}), 400
        except Exception as e:
            self.logger.error("[route:semantic] 5xx unexpected error", exc_info=True)
            return jsonify({"error": "Internal server error", "details": str(e)}), 500

    def rerank_papers_bm25_route(self):
        data = request.get_json(silent=True) or {}
        text = data.get("text")
        limit = data.get("limit", 20)

        try:
            limit = int(limit)
        except (TypeError, ValueError):
            self.logger.warning("[route:bm25] Invalid 'limit' type; must be integer")
            return jsonify({"error": "limit must be an integer"}), 400

        try:
            self.logger.info(f"[route:bm25] Request len(text)={len(text) if text else 0} limit={limit}")
            results = self.rerank_papers_bm25(text=text, limit=limit)
            self.logger.info(f"[route:bm25] Returning {len(results)} results")
            return jsonify({"results": results})
        except ValueError as ve:
            self.logger.warning(f"[route:bm25] 4xx: {ve}")
            return jsonify({"error": str(ve)}), 400
        except Exception as e:
            self.logger.error("[route:bm25] 5xx unexpected error", exc_info=True)
            return jsonify({"error": "Internal server error", "details": str(e)}), 500
