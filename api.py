# api.py
from flask import Flask, request, jsonify
from src.impl.specter_2_embedder import Specter2Embedder 
from src.impl.mysql_database import MySQLDatabase
from src.impl.qdrant_database import QdrantDatabase

app = Flask(__name__)
print("running api.py")

@app.route('/', methods=['GET'])
def helloworld():
    return "Hello, World!"

@app.route('/chunks', methods=['GET'])
def get_chunks():
    text = request.args.get('text')
    if not text:
        return jsonify({"error": "Missing mandatory 'text' parameter"}), 400

    embedder = Specter2Embedder()
    mysql_db = MySQLDatabase()
    qdrant_db = QdrantDatabase()

    limit = request.args.get('limit', default=None, type=int)

    embedding = embedder.embed(text)
    if limit:
        similar_chunks = qdrant_db.get_similar(embedding.tolist(), top_k=limit)
    else:
        similar_chunks = qdrant_db.get_similar(embedding.tolist())

    results = []
    for chunk_id, score in similar_chunks:
        chunk_data = mysql_db.get_chunk_by_uuid(chunk_id)
        if not chunk_data:
            continue
        results.append({
            "uuid": chunk_id,
            "text": chunk_data.text,
            "arxiv_id": chunk_data.arxiv_id,
            "score": score
        })

    return jsonify({"chunks": results})
