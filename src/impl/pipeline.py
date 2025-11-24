from tqdm.auto import tqdm
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
        self.logger.debug(
            "Pipeline initialized with %s, %s, %s, %s, %s",
            type(relational_db).__name__,
            type(vector_db).__name__,
            type(data_provider).__name__,
            type(pdf_converter).__name__,
            type(embedder).__name__,
        )

        self._tqdm_disable = False

        self._paper_bar_format = "{l_bar}{bar}| {n_fmt} papers [{elapsed}<{remaining}, {rate_fmt}]"
        self._chunk_bar_format = (
            "  {l_bar}{bar}| {n_fmt}/{total_fmt} chunks "
            "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )

    def process(self):
        self.logger.info("Pipeline processing started.")

        with tqdm(
            desc="Papers",
            unit="paper",
            dynamic_ncols=True,
            mininterval=0.5,
            bar_format=self._paper_bar_format,
            disable=self._tqdm_disable,
        ) as paper_bar:
            while self.data_provider.hasNext():
                result = self.data_provider.next()
                if result is None:
                    self.logger.warning("Data provider returned None — no more papers or an error occurred.")
                    break

                arxiv_id, pdf_data = result
                if arxiv_id is None or pdf_data is None:
                    self.logger.warning(
                        "Received invalid paper result (arxiv_id=%s, pdf_data=%s). Skipping...",
                        arxiv_id,
                        "None" if pdf_data is None else "bytes",
                    )
                    continue

                if self.mysql_db.paper_exists(arxiv_id):
                    self.logger.info("Paper %s already exists – skipping.", arxiv_id)
                    continue

                self.logger.info("Processing paper: %s", arxiv_id)

                try:
                    text = self.pdf_converter.pdf_to_string(pdf_data)
                except Exception as e:
                    self.logger.error("Failed to extract text for %s: %s", arxiv_id, e, exc_info=True)
                    continue

                if not text or not text.strip():
                    self.logger.warning("No text extracted for %s – skipping.", arxiv_id)
                    continue

                try:
                    metadata = self.pdf_converter.pdf_metadata(pdf_data)
                except Exception as e:
                    self.logger.error("Failed to extract metadata for %s: %s", arxiv_id, e, exc_info=True)
                    metadata = {}

                cleaned_text = self.pdf_converter.clean_string(text)
                chunks = self.pdf_converter.chunk_string(cleaned_text)

                if not chunks:
                    self.logger.warning("No chunks generated for %s – skipping embedding.", arxiv_id)
                    paper_bar.update(1)
                    continue

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
                self.logger.info("Created DB record for paper %s (%d chunks)", arxiv_id, len(chunks))

                try:
                    embeddings = self.embedder.embed(chunks)  
                except Exception as e:
                    self.logger.error("Failed to embed chunks for %s: %s", arxiv_id, e, exc_info=True)
                    paper_bar.update(1)
                    continue

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
                    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings), start=1):
                        self.qdrant_db.add_chunk(arxiv_id, embedding.tolist(), chunk)
                        chunk_bar.update(1)
                        if idx % 25 == 0 or idx == len(chunks):
                            chunk_bar.set_postfix_str(f" last={idx}")
                            self.logger.debug("Embedded %d/%d chunks for %s.", idx, len(chunks), arxiv_id)

                paper_bar.set_postfix_str(f"last={arxiv_id} chunks={len(chunks)}")
                paper_bar.update(1)

        self.logger.info("Pipeline processing finished.")
