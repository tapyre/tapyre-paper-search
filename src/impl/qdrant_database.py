import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.models import Distance, VectorParams
from src.core.database import Database

class QdrantDatabase(Database):
    def __init__(self):
        self.collection_name = os.getenv("QDRANT_COLLECTION", "chunks")
        # self.host = os.getenv("QDRANT_HOST", "qdrant_db")
        self.host = os.getenv("QDRANT_HOST", "localhost")
        self.port = int(os.getenv("QDRANT_PORT", 6333))
        self.vector_size = int(os.getenv("VECTOR_SIZE", 768)) 

        self.client = None
        super().__init__()

    def connect(self):
        self.client = QdrantClient(host=self.host, port=self.port)

        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )

    def disconnect(self):
        self.client = None

    def add_chunk(self, arxiv_id: str, embedding, text: str):
        if not arxiv_id or embedding is None or text is None:
            raise ValueError("Both arxiv_id, embedding, and text are required.")

        if hasattr(embedding, "tolist"):
            embedding_list = embedding.tolist()
        else:
            embedding_list = embedding

        if isinstance(embedding_list[0], (list, tuple)):
            embedding_list = embedding_list[0]

        pk = str(uuid.uuid4())
        point = PointStruct(
            id=pk,
            vector=embedding_list,
            payload={
                "arxiv_id": arxiv_id,
                "text": text
            }
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        return pk

    def get_similar(self, embedding: list[float], top_k: int = 5):
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()

        if isinstance(embedding, list) and isinstance(embedding[0], list):
            embedding = embedding[0]

        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=top_k
        )
        return search_result

    def get_statistics(self):
        stats = self.client.get_collection(self.collection_name)
        return {
            "points_count": stats.points_count,
            "segments_count": getattr(stats, "segments_count", None),
        }