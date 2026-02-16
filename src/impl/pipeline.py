from tqdm.auto import tqdm
from src.impl.simple_data_provider import SimpleDataProvider
from src.impl.fitz_pdf_converter import FitzPdfConverter
from src.impl.specter_2_embedder import Specter2Embedder
from src.impl.mysql_database import MySQLDatabase
from src.impl.qdrant_database import QdrantDatabase
from src.impl.logger import get_logger
import os


class Pipeline:
    """
    Orchestrates the end-to-end ingestion flow for arXiv papers.

    High-level workflow:
    1) Pull next PDF from a DataProvider
    2) Extract full text + metadata from PDF
    3) Clean and chunk text for embedding
    4) Insert the Paper record into a relational DB (MySQL)
    5) Compute embeddings for each chunk (Specter2 / embedding model)
    6) Store embeddings + chunk text in a vector DB (Qdrant)

    Key design goals:
    - Resumable ingestion: DataProvider is responsible for progress tracking
    - Safe operation: individual failures should not crash the pipeline
    - Observability: structured logging + progress bars for monitoring
    - Scalability: chunk-level progress feedback and batching opportunities
    """

    def __init__(self, relational_db, vector_db, data_provider, pdf_converter, embedder):
        # Store injected dependencies (dependency injection makes testing and swapping components easy)
        self.mysql_db = relational_db
        self.qdrant_db = vector_db
        self.data_provider = data_provider
        self.pdf_converter = pdf_converter
        self.embedder = embedder

        # Logger for traceable runtime diagnostics
        self.logger = get_logger(__name__)
        self.logger.debug(
            "Pipeline initialized with %s, %s, %s, %s, %s",
            type(relational_db).__name__,
            type(vector_db).__name__,
            type(data_provider).__name__,
            type(pdf_converter).__name__,
            type(embedder).__name__,
        )

        # Toggle to disable progress bars (useful for CI logs or non-interactive environments)
        self._tqdm_disable = False

        # Custom bar formats for consistent and readable terminal UI
        self._paper_bar_format = "{l_bar}{bar}| {n_fmt} papers [{elapsed}<{remaining}, {rate_fmt}]"
        self._chunk_bar_format = (
            "  {l_bar}{bar}| {n_fmt}/{total_fmt} chunks "
            "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )

    def process(self):
        """
        Run the ingestion process until the DataProvider reports completion.

        Failure policy:
        - If a specific paper fails at any stage, it is skipped and processing continues
        - If the provider returns None, the pipeline terminates gracefully

        Operational notes:
        - The relational DB insert happens before embeddings are written to Qdrant.
          This can be used to track which papers were "seen" even if embedding fails.
        """
        self.logger.info("Pipeline processing started.")

        # Outer progress bar tracking number of processed papers
        with tqdm(
            desc="Papers",
            unit="paper",
            dynamic_ncols=True,
            mininterval=0.5,
            bar_format=self._paper_bar_format,
            disable=self._tqdm_disable,
        ) as paper_bar:

            # Continue until provider signals completion
            while self.data_provider.hasNext():
                # Fetch next paper (arxiv_id, pdf_data)
                result = self.data_provider.next()

                # Provider returning None is treated as completion or fatal provider error
                if result is None:
                    self.logger.warning(
                        "Data provider returned None — no more papers or an error occurred."
                    )
                    break

                arxiv_id, pdf_data = result

                # Validate provider output to avoid downstream exceptions
                if arxiv_id is None or pdf_data is None:
                    self.logger.warning(
                        "Received invalid paper result (arxiv_id=%s, pdf_data=%s). Skipping...",
                        arxiv_id,
                        "None" if pdf_data is None else "bytes",
                    )
                    continue

                # Skip papers already present in the relational DB
                # This protects against duplicates and enables resumable ingestion.
                if self.mysql_db.paper_exists(arxiv_id):
                    self.logger.info("Paper %s already exists – skipping.", arxiv_id)
                    continue

                self.logger.info("Processing paper: %s", arxiv_id)

                # Step 1: Extract full text from PDF bytes
                try:
                    text = self.pdf_converter.pdf_to_string(pdf_data)
                except Exception as e:
                    # Text extraction failure is non-fatal: skip paper and continue
                    self.logger.error(
                        "Failed to extract text for %s: %s", arxiv_id, e, exc_info=True
                    )
                    continue

                # Reject empty extraction results early (saves DB and embedding work)
                if not text or not text.strip():
                    self.logger.warning("No text extracted for %s – skipping.", arxiv_id)
                    continue

                # Step 2: Extract metadata (best effort)
                # Metadata failure should not block ingestion, so we default to {}.
                try:
                    metadata = self.pdf_converter.pdf_metadata(pdf_data)
                except Exception as e:
                    self.logger.error(
                        "Failed to extract metadata for %s: %s", arxiv_id, e, exc_info=True
                    )
                    metadata = {}

                # Step 3: Clean and chunk text for embedding
                cleaned_text = self.pdf_converter.clean_string(text)
                chunks = self.pdf_converter.chunk_string(cleaned_text)

                # If chunking yields nothing, we cannot embed (skip paper)
                if not chunks:
                    self.logger.warning(
                        "No chunks generated for %s – skipping embedding.", arxiv_id
                    )
                    paper_bar.update(1)
                    continue

                # Step 4: Insert paper into relational DB
                # Store the original extracted text plus metadata fields.
                self.mysql_db.add_paper(
                    arxiv_id=arxiv_id,
                    text=text,
                    title=metadata.get("title"),
                    author=metadata.get("author"),
                    subject=metadata.get("subject"),
                    keywords=metadata.get("keywords"),
                    creator=metadata.get("creator"),
                    producer=metadata.get("producer"),
                    creation_date=metadata.get("creationDate"),
                    modification_date=metadata.get("modDate"),
                    trapped=metadata.get("trapped"),
                )
                self.logger.info(
                    "Created DB record for paper %s (%d chunks)", arxiv_id, len(chunks)
                )

                # Step 5: Generate embeddings for each chunk
                # embed() is expected to return a list/array aligned with chunks length.
                try:
                    embeddings = self.embedder.embed(chunks)
                except Exception as e:
                    # Embedding failure is non-fatal: DB record exists, but vector DB will miss this paper
                    self.logger.error(
                        "Failed to embed chunks for %s: %s", arxiv_id, e, exc_info=True
                    )
                    paper_bar.update(1)
                    continue

                # Step 6: Store chunk embeddings in vector DB (Qdrant)
                # Inner progress bar tracks chunk upload progress for the current paper.
                with tqdm(
                    total=len(chunks),
                    desc=f"Embedding {arxiv_id}",
                    unit="chunk",
                    dynamic_ncols=True,
                    mininterval=0.1,
                    leave=False,
                    bar_format=self._chunk_bar_format,
                    disable=self._tqdm_disable,
                ) as chunk_bar:

                    # Iterate chunk + embedding pairs and persist each to Qdrant
                    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings), start=1):
                        # Convert numpy/tensor embedding to plain Python list for JSON serialization
                        self.qdrant_db.add_chunk(arxiv_id, embedding.tolist(), chunk)

                        chunk_bar.update(1)

                        # Periodic progress logs help in non-interactive logs and debugging
                        if idx % 25 == 0 or idx == len(chunks):
                            chunk_bar.set_postfix_str(f" last={idx}")
                            self.logger.debug(
                                "Embedded %d/%d chunks for %s.", idx, len(chunks), arxiv_id
                            )

                # Update paper progress bar status with last processed paper ID
                paper_bar.set_postfix_str(f"last={arxiv_id} chunks={len(chunks)}")
                paper_bar.update(1)

        self.logger.info("Pipeline processing finished.")
