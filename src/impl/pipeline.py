from src.impl.simple_data_provider import SimpleDataProvider
from src.impl.fitz_pdf_converter import FitzPdfConverter
from src.impl.specter_2_embedder import Specter2Embedder
from src.impl.mysql_database import MySQLDatabase
from src.impl.qdrant_database import QdrantDatabase
from src.impl.logger import get_logger
import os


class Pipeline:
    def __init__(self, relational_db, vector_db, data_provider, pdf_converter, embedder):
        self.mysql_db = relational_db
        self.qdrant_db = vector_db
        self.data_provider = data_provider
        self.pdf_converter = pdf_converter
        self.embedder = embedder
        self.logger = get_logger(__name__)
        self.logger.debug("Pipeline initialized with %s, %s, %s, %s, %s",
                          type(relational_db).__name__,
                          type(vector_db).__name__,
                          type(data_provider).__name__,
                          type(pdf_converter).__name__,
                          type(embedder).__name__)

    def process(self):
        self.logger.info("Pipeline processing started.")
        while self.data_provider.hasNext():
            result = self.data_provider.next()
            if result is None:
                self.logger.warning("Data provider returned None — no more papers or an error occurred.")
                break

            arxiv_id, pdf_data = result
            if arxiv_id is None or pdf_data is None:
                self.logger.warning("Received invalid paper result (arxiv_id=%s, pdf_data=%s). Skipping...",
                                    arxiv_id, "None" if pdf_data is None else "bytes")
                continue

            if self.mysql_db.paper_exists(arxiv_id):
                self.logger.info("Paper %s already exists – skipping entire processing.", arxiv_id)
                continue

            self.logger.info("Processing paper: %s", arxiv_id)

            text = self.pdf_converter.pdf_to_string(pdf_data)
            metadata = self.pdf_converter.pdf_metadata(pdf_data)
            cleaned_text = self.pdf_converter.clean_string(text)
            chunks = self.pdf_converter.chunk_string(cleaned_text)

            self.logger.debug("Metadata for %s: %s", arxiv_id, metadata)

            self.mysql_db.add_paper(
                arxiv_id=arxiv_id,
                text=text,
                title=metadata.get('title'),
                author=metadata.get('author'),
                subject=metadata.get('subject'),
                keywords=metadata.get('keywords'),
                creator=metadata.get('creator'),
                producer=metadata.get('producer'),
                creation_date=metadata.get('creationDate'),
                modification_date=metadata.get('modDate'),
                trapped=metadata.get('trapped')
            )
            self.logger.info("Created DB record for paper %s. Chunks to embed: %d", arxiv_id, len(chunks))

            for idx, chunk in enumerate(chunks, start=1):
                embedding = self.embedder.embed(chunk)
                self.qdrant_db.add_chunk(arxiv_id, embedding.tolist(), chunk)
                if idx % 25 == 0 or idx == len(chunks):
                    self.logger.debug("Embedded and stored %d/%d chunks for %s.", idx, len(chunks), arxiv_id)

        self.logger.info("Pipeline processing finished.")