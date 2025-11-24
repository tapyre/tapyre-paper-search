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

    def _get_matching_chunks_semantic(self, chunk_pts, query_embedding, top_k=5):
        if not chunk_pts:
            return []
        scored_chunks = []
        for pt in chunk_pts:
            payload = getattr(pt, "payload", {}) or {}
            scored_chunks.append({
                "chunk_id": payload.get("id", getattr(pt, "id", None)),
                "text": payload.get("text", None),
                "score": float(getattr(pt, "score", 0.0))
            })
        return sorted(scored_chunks, key=lambda x: x["score"], reverse=True)[:top_k]

    def _get_matching_chunks_bm25(self, chunk_objs, query_text, top_k=5):
        if not chunk_objs:
            return []
        documents = [safe_tokenize(getattr(chunk, "text", "").lower()) for chunk in chunk_objs]
        bm25 = BM25Okapi(documents)
        tokenized_query = safe_tokenize(query_text.lower())
        scores = bm25.get_scores(tokenized_query)
        scored_chunks = []
        for i, chunk in enumerate(chunk_objs):
            scored_chunks.append({
                "chunk_id": getattr(chunk, "id", None),
                "text": getattr(chunk, "text", None),
                "score": float(scores[i])
            })
        return sorted(scored_chunks, key=lambda x: x["score"], reverse=True)[:top_k]

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

        paper_chunks = {}
        for pt in similar:
            payload = getattr(pt, "payload", {}) or {}
            arxiv_id = payload.get("arxiv_id") or getattr(pt, "id", None)
            if not arxiv_id:
                continue
            paper_chunks.setdefault(arxiv_id, []).append(pt)

        candidates = []
        for arxiv_id, chunk_pts in paper_chunks.items():
            paper_meta = self.mysql_db.get_paper_by_arxiv_id(arxiv_id) if arxiv_id else None
            meta = self._paper_to_dict(paper_meta) if paper_meta else {"arxiv_id": arxiv_id}
            matching_chunks = self._get_matching_chunks_semantic(chunk_pts, embedding_list, top_k=5)
            # Use the highest chunk score as the paper score
            paper_score = max((c["score"] for c in matching_chunks), default=0.0)
            candidates.append({**meta, "score": paper_score, "matching_chunks": matching_chunks})

        results = self._dedup_by_arxiv(sorted(candidates, key=lambda x: x["score"], reverse=True))[:limit]
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

        paper_chunks = {}
        for pt in candidates:
            payload = getattr(pt, "payload", {}) or {}
            arxiv_id = payload.get("arxiv_id") or getattr(pt, "id", None)
            if not arxiv_id:
                continue
            paper_chunks.setdefault(arxiv_id, []).append(pt)

        paper_map = []
        for arxiv_id, chunk_pts in paper_chunks.items():
            paper_obj = self.mysql_db.get_paper_by_arxiv_id(arxiv_id) if arxiv_id else None
            meta = self._paper_to_dict(paper_obj)
            chunk_objs = []
            for pt in chunk_pts:
                payload = getattr(pt, "payload", {}) or {}
                class Chunk:
                    pass
                chunk = Chunk()
                setattr(chunk, "id", payload.get("id", getattr(pt, "id", None)))
                setattr(chunk, "text", payload.get("text", None))
                chunk_objs.append(chunk)
            matching_chunks = self._get_matching_chunks_bm25(chunk_objs, text, top_k=5)
            paper_score = max((c["score"] for c in matching_chunks), default=0.0)
            paper_map.append({
                "arxiv_id": arxiv_id,
                "bm25_score": paper_score,
                "title": getattr(paper_obj, "title", None),
                "meta": meta,
                "matching_chunks": matching_chunks
            })

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
