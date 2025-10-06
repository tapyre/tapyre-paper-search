class ChunksController():
    def __init__(self, qdrant_db, embedder, mysql_db):
        self.qdrant_db = qdrant_db
        self.embedder = embedder
        self.mysql_db = mysql_db
        # self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter2")

    def get_chunks_semantic(self, text, limit=None):
        if not text:
            raise ValueError("Missing mandatory 'text' parameter")

        embedding = self.embedder.embed(text)

        if limit:
            similar_chunks = self.qdrant_db.get_similar(embedding.tolist(), top_k=limit)
        else:
            similar_chunks = self.qdrant_db.get_similar(embedding.tolist())

        sorted_chunks = sorted(similar_chunks, key=lambda x: x.score, reverse=True)

        results = []
        for chunk in sorted_chunks:
            results.append({
                "id": chunk.id,
                "score": chunk.score,
                "text": chunk.payload.get("text"),
                "arxiv_id": chunk.payload.get("arxiv_id")
            })

        return results

    def rerank_chunks_bm25(self, text, limit=20):
        if not text:
            raise ValueError("Missing mandatory 'text' parameter")

        embedding = self.embedder.embed(text)
        similar_chunks = self.qdrant_db.get_similar(embedding.tolist(), top_k=limit)

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
        tokenized_query = word_tokenize(text.lower())
        bm25_scores = bm25.get_scores(tokenized_query)

        for i, score in enumerate(bm25_scores):
            chunk_map[i]["bm25_score"] = score

        reranked_chunks = sorted(chunk_map, key=lambda x: x["bm25_score"], reverse=True)

        return reranked_chunkss