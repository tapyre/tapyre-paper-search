import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.models import Distance, VectorParams
from src.core.database import Database
from src.impl.logger import get_logger


class QdrantDatabase(Database):
    """
    Vector database implementation backed by Qdrant.

    Responsibilities:
    - Manage connectivity to a Qdrant server (host/port)
    - Ensure the target collection exists (create if missing)
    - Upsert chunk vectors + payloads (arxiv_id + text)
    - Run similarity search queries
    - Provide simple collection statistics for monitoring

    Design goals:
    - Production robustness: clear logging and retry behavior for transient network errors
    - Configurability: environment variables control host/port/collection/vector size
    - Minimal payload schema: store enough context to retrieve original text chunks
    """

    def __init__(self):
        # Logger for structured runtime diagnostics
        self.logger = get_logger(__name__)

        # Configuration via environment variables (with safe defaults for Docker)
        self.collection_name = os.getenv("QDRANT_COLLECTION", "chunks")
        self.host = os.getenv("QDRANT_HOST", "qdrant_db")
        self.port = int(os.getenv("QDRANT_PORT", 6333))

        # Vector dimension must match the embedder output size (e.g., Specter2 -> 768)
        self.vector_size = int(os.getenv("VECTOR_SIZE", 768))

        # Client will be created in connect()
        self.client = None

        self.logger.info(
            "[QdrantDatabase] Initialized with host=%s, port=%d, collection=%s, vector_size=%d",
            self.host, self.port, self.collection_name, self.vector_size
        )

        # Parent-class initialization hook for consistency with other DB implementations
        try:
            super().__init__()
            self.logger.debug("[QdrantDatabase] Super init successful")
        except Exception as e:
            self.logger.exception("[QdrantDatabase] Error during super init: %s", e)
            raise

    def connect(self):
        """
        Establish connection to Qdrant and ensure the configured collection exists.

        Behavior:
        - Creates a QdrantClient instance
        - Checks whether the collection exists
        - If not found, creates (recreates) it with configured vector params

        Note:
        - recreate_collection() will drop existing collection state if it exists.
          In this code it's only called in the "collection not found" branch,
          but depending on Qdrant behavior and exceptions, ensure you do not
          accidentally recreate in production.
        """
        self.logger.info("[QdrantDatabase] Connecting to Qdrant at %s:%d...", self.host, self.port)
        self.client = QdrantClient(host=self.host, port=self.port)

        try:
            # Fast existence check
            self.client.get_collection(self.collection_name)
            self.logger.info("[QdrantDatabase] Collection '%s' already exists.", self.collection_name)

        except Exception:
            # Collection missing or server returned an error.
            # We interpret this as "needs creation" and attempt to create it.
            self.logger.warning(
                "[QdrantDatabase] Collection '%s' not found. Creating a new one...",
                self.collection_name
            )
            try:
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                self.logger.info(
                    "[QdrantDatabase] Collection '%s' created successfully.",
                    self.collection_name
                )
            except Exception as e:
                # Creation failure is fatal (cannot proceed without collection)
                self.logger.exception(
                    "[QdrantDatabase] Failed to create collection '%s': %s",
                    self.collection_name,
                    e
                )
                raise

    def disconnect(self):
        """
        Disconnect from Qdrant.

        Note:
        - QdrantClient is generally stateless over HTTP, so "disconnect" here
          simply clears the client reference to prevent accidental reuse.
        """
        if self.client:
            self.client = None
            self.logger.info("[QdrantDatabase] Disconnected from Qdrant.")
        else:
            self.logger.warning("[QdrantDatabase] Disconnect called, but client was already None.")

    def add_chunk(
        self,
        arxiv_id: str,
        embedding,
        text: str,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ):
        """
        Insert a single chunk embedding into Qdrant.

        Parameters:
        - arxiv_id: paper identifier (used to group chunks by document)
        - embedding: vector representation (list/np array/tensor supported)
        - text: original chunk text stored as payload
        - max_retries: number of retry attempts for transient errors
        - base_delay: base delay for exponential backoff in seconds

        Returns:
        - The generated point ID (UUID string) if insert succeeds

        Error handling policy:
        - Transient network/protocol errors: retry with exponential backoff
        - All other errors: fail immediately (no retry)

        Important:
        - This method assumes self.client is initialized via connect()
        - It uses wait=True to ensure the write is committed before returning
          (safer but slower for bulk ingestion)
        """
        # Validate required fields early to avoid writing malformed points
        if not arxiv_id or embedding is None or text is None:
            self.logger.error(
                "[QdrantDatabase] Missing required fields (arxiv_id=%s, embedding=%s, text=%s).",
                arxiv_id,
                type(embedding),
                "present" if text else "None",
            )
            raise ValueError("Both arxiv_id, embedding, and text are required.")

        # Normalize embedding input to a plain Python list
        # This supports numpy arrays / torch tensors (via .tolist()).
        if hasattr(embedding, "tolist"):
            embedding_list = embedding.tolist()
        else:
            embedding_list = embedding

        # Some embedders return shape (1, dim) instead of (dim,)
        # Flatten one level if needed.
        if isinstance(embedding_list[0], (list, tuple)):
            embedding_list = embedding_list[0]

        # Generate a unique primary key for the vector point
        pk = str(uuid.uuid4())
        self.logger.debug(
            "[QdrantDatabase] Adding chunk with ID %s for paper %s", pk, arxiv_id
        )

        # Build Qdrant point with vector + payload metadata
        point = PointStruct(
            id=pk,
            vector=embedding_list,
            payload={
                "arxiv_id": arxiv_id,
                "text": text,
            },
        )

        # Retry loop for transient failures
        attempt = 0
        while attempt < max_retries:
            attempt += 1
            try:
                # Upsert point into collection (insert or update by point ID)
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[point],
                    wait=True,  # ensure write is persisted before returning
                )

                self.logger.info(
                    "[QdrantDatabase] Added chunk %s for paper %s (attempt %d/%d)",
                    pk,
                    arxiv_id,
                    attempt,
                    max_retries,
                )
                return pk

            except (
                ResponseHandlingException,
                httpx.RemoteProtocolError,
                httpcore.RemoteProtocolError,
            ) as e:
                # Transient errors: retry with exponential backoff (capped)
                if attempt >= max_retries:
                    self.logger.exception(
                        "[QdrantDatabase] Giving up on chunk %s for paper %s after %d attempts: %s",
                        pk,
                        arxiv_id,
                        attempt,
                        e,
                    )
                    raise

                delay = min(base_delay * (2 ** (attempt - 1)), 30.0)
                self.logger.warning(
                    "[QdrantDatabase] Transient error when upserting chunk %s "
                    "for paper %s (attempt %d/%d): %s – retrying in %.1fs",
                    pk,
                    arxiv_id,
                    attempt,
                    max_retries,
                    e,
                    delay,
                )
                time.sleep(delay)

            except Exception as e:
                # Non-transient errors: fail fast and surface the exception
                self.logger.exception(
                    "[QdrantDatabase] Failed to upsert chunk %s for paper %s (no retry): %s",
                    pk,
                    arxiv_id,
                    e,
                )
                raise

    def get_similar(self, embedding: list[float], top_k: int = 5):
        """
        Perform a similarity search in Qdrant.

        Parameters:
        - embedding: query vector (list/np array/tensor supported)
        - top_k: number of nearest neighbors to return

        Returns:
        - List of search hits (Qdrant result objects with score + payload)

        Notes:
        - Uses cosine distance if the collection was created with Distance.COSINE
        - Payload is returned by default (depending on client settings)
        """
        self.logger.debug("[QdrantDatabase] Searching for top %d similar embeddings.", top_k)

        # Normalize embedding input to a plain Python list
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()

        # Flatten if query vector is nested
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
        """
        Fetch basic collection statistics from Qdrant.

        Returns:
        - dict containing:
          - points_count: number of vectors stored
          - segments_count: optional internal storage segments count
        """
        self.logger.info(
            "[QdrantDatabase] Fetching collection statistics for '%s'",
            self.collection_name
        )
        try:
            stats = self.client.get_collection(self.collection_name)

            # Some fields may vary by Qdrant version, so we use getattr for optional values
            data = {
                "points_count": stats.points_count,
                "segments_count": getattr(stats, "segments_count", None),
            }

            self.logger.info("[QdrantDatabase] Statistics: %s", data)
            return data

        except Exception as e:
            self.logger.exception("[QdrantDatabase] Failed to fetch collection stats: %s", e)
            raise
