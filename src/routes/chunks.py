from flask import Blueprint, request, jsonify
from src.impl.specter_2_embedder import Specter2Embedder 
from src.impl.qdrant_database import QdrantDatabase
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

chunk_blueprint = Blueprint('chunks', __name__)

@chunk_blueprint.route('/semantic', methods=['GET'])
def get_chunks():
    text = request.args.get('text')
    if not text:
        return jsonify({"error": "Missing mandatory 'text' parameter"}), 400

    embedder = Specter2Embedder()
    qdrant_db = QdrantDatabase()

    limit = request.args.get('limit', default=None, type=int)
    embedding = embedder.embed(text)

    if limit:
        similar_chunks = qdrant_db.get_similar(embedding.tolist(), top_k=limit)
    else:
        similar_chunks = qdrant_db.get_similar(embedding.tolist())

    sorted_chunks = sorted(similar_chunks, key=lambda x: x.score, reverse=True)

    results = []
    for chunk in sorted_chunks:
        results.append({
            "id": chunk.id,
            "score": chunk.score,
            "text": chunk.payload.get("text"),
            "arxiv_id": chunk.payload.get("arxiv_id")
        })

    return jsonify({"chunks": results})


@chunk_blueprint.route('/rerank', methods=['GET'])
def rerank_with_bm25():
    text = request.args.get('text')
    if not text:
        return jsonify({"error": "Missing mandatory 'text' parameter"}), 400

    embedder = Specter2Embedder()
    qdrant_db = QdrantDatabase()

    limit = request.args.get('limit', default=20, type=int) 
    embedding = embedder.embed(text)
    similar_chunks = qdrant_db.get_similar(embedding.tolist(), top_k=limit)

    documents = []
    chunk_map = []

    for chunk in similar_chunks:
        chunk_text = chunk.payload.get("text", "")
        if chunk_text:
            documents.append(word_tokenize(chunk_text.lower()))
            chunk_map.append({
                "id": chunk.id,
                "original_score": chunk.score,
                "text": chunk_text,
                "arxiv_id": chunk.payload.get("arxiv_id")
            })

    bm25 = BM25Okapi(documents)
    query_tokens = word_tokenize(text.lower())

    scores = bm25.get_scores(query_tokens)

    for i, score in enumerate(scores):
        chunk_map[i]["bm25_score"] = score

    reranked_chunks = sorted(chunk_map, key=lambda x: x["bm25_score"], reverse=True)

    return jsonify({
        "chunks": reranked_chunks
    })


@chunk_blueprint.route('/healthcheck', methods=['GET'])
def healthcheck():
    return jsonify({"status": "ok"})
