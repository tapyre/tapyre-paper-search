import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.models import Distance, VectorParams
from src.core.database import Database
from src.impl.logger import get_logger


class QdrantDatabase(Database):
    def __init__(self):
        self.logger = get_logger(__name__)
        self.collection_name = os.getenv("QDRANT_COLLECTION", "chunks")
        self.host = os.getenv("QDRANT_HOST", "localhost")
        self.port = int(os.getenv("QDRANT_PORT", 6333))
        self.vector_size = int(os.getenv("VECTOR_SIZE", 768))
        self.client = None

        self.logger.info(
            "[QdrantDatabase] Initialized with host=%s, port=%d, collection=%s, vector_size=%d",
            self.host, self.port, self.collection_name, self.vector_size
        )

        try:
            super().__init__()
            self.logger.debug("[QdrantDatabase] Super init successful")
        except Exception as e:
            self.logger.exception("[QdrantDatabase] Error during super init: %s", e)
            raise

    def connect(self):
        self.logger.info("[QdrantDatabase] Connecting to Qdrant at %s:%d...", self.host, self.port)
        self.client = QdrantClient(host=self.host, port=self.port)

        try:
            self.client.get_collection(self.collection_name)
            self.logger.info("[QdrantDatabase] Collection '%s' already exists.", self.collection_name)
        except Exception:
            self.logger.warning("[QdrantDatabase] Collection '%s' not found. Creating a new one...", self.collection_name)
            try:
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                self.logger.info("[QdrantDatabase] Collection '%s' created successfully.", self.collection_name)
            except Exception as e:
                self.logger.exception("[QdrantDatabase] Failed to create collection '%s': %s", self.collection_name, e)
                raise

    def disconnect(self):
        if self.client:
            self.client = None
            self.logger.info("[QdrantDatabase] Disconnected from Qdrant.")
        else:
            self.logger.warning("[QdrantDatabase] Disconnect called, but client was already None.")

    def add_chunk(self, arxiv_id: str, embedding, text: str):
        if not arxiv_id or embedding is None or text is None:
            self.logger.error("[QdrantDatabase] Missing required fields (arxiv_id=%s, embedding=%s, text=%s).",
                              arxiv_id, type(embedding), "present" if text else "None")
            raise ValueError("Both arxiv_id, embedding, and text are required.")

        if hasattr(embedding, "tolist"):
            embedding_list = embedding.tolist()
        else:
            embedding_list = embedding

        if isinstance(embedding_list[0], (list, tuple)):
            embedding_list = embedding_list[0]

        pk = str(uuid.uuid4())
        self.logger.debug("[QdrantDatabase] Adding chunk with ID %s for paper %s", pk, arxiv_id)

        point = PointStruct(
            id=pk,
            vector=embedding_list,
            payload={
                "arxiv_id": arxiv_id,
                "text": text
            }
        )

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            self.logger.info("[QdrantDatabase] Added chunk %s for paper %s", pk, arxiv_id)
            return pk
        except Exception as e:
            self.logger.exception("[QdrantDatabase] Failed to upsert chunk for paper %s: %s", arxiv_id, e)
            raise

    def get_similar(self, embedding: list[float], top_k: int = 5):
        self.logger.debug("[QdrantDatabase] Searching for top %d similar embeddings.", top_k)

        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()

        if isinstance(embedding, list) and isinstance(embedding[0], list):
            embedding = embedding[0]

        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                limit=top_k
            )
            self.logger.info("[QdrantDatabase] Found %d similar vectors.", len(search_result))
            return search_result
        except Exception as e:
            self.logger.exception("[QdrantDatabase] Error during similarity search: %s", e)
            raise

    def get_statistics(self):
        self.logger.info("[QdrantDatabase] Fetching collection statistics for '%s'", self.collection_name)
        try:
            stats = self.client.get_collection(self.collection_name)
            data = {
                "points_count": stats.points_count,
                "segments_count": getattr(stats, "segments_count", None),
            }
            self.logger.info("[QdrantDatabase] Statistics: %s", data)
            return data
        except Exception as e:
            self.logger.exception("[QdrantDatabase] Failed to fetch collection stats: %s", e)
            raise
