from src.impl.simple_data_provider import SimpleDataProvider
from src.impl.fitz_pdf_converter import FitzPdfConverter
from src.impl.specter_2_embedder import Specter2Embedder 
from src.impl.mysql_database import MySQLDatabase
from src.impl.qdrant_database import QdrantDatabase
import os

class Pipeline:
    def __init__(self, relational_db, vector_db, data_provider, pdf_converter, embedder):
        self.mysql_db = relational_db
        self.qdrant_db = vector_db
        self.data_provider = data_provider
        self.pdf_converter = pdf_converter
        self.embedder = embedder

    def process(self):    
        while self.data_provider.hasNext():
            result = self.data_provider.next()
            if result is None:
                print("Data provider returned None — no more papers or an error occurred", flush=True)
                break

            arxiv_id, pdf_data = result
            if arxiv_id is None or pdf_data is None:
                print("Received invalid paper result, skipping...", flush=True)
                continue

            text = self.pdf_converter.pdf_to_string(pdf_data)
            metadata = self.pdf_converter.pdf_metadata(pdf_data)
            cleaned_text = self.pdf_converter.clean_string(text)
            chunks = self.pdf_converter.chunk_string(cleaned_text)
            print("Metadata: ", metadata)
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
            print("Chunks: ", len(chunks))
            print(f"Processing paper: {arxiv_id}")
            for chunk in chunks:
                embedding = self.embedder.embed(chunk)
                self.qdrant_db.add_chunk(arxiv_id, embedding.tolist(), chunk)

    def call(self, pdf_data, arxiv_id):
            text = self.pdf_converter.pdf_to_string(pdf_data)
            cleaned_text = self.pdf_converter.clean_string(text)
            chunks = self.pdf_converter.chunk_string(cleaned_text)
            print("Chunks: ", len(chunks))
            for chunk in chunks:
                embedding = self.embedder.embed(chunk)
                chunk_uuid = self.mysql_db.add_chunk(text=chunk, arxiv_id=arxiv_id) 
                self.qdrant_db.add_chunk(uuid=chunk_uuid, embedding=embedding.tolist())
                print(f"Added chunk with UUID: {chunk_uuid}")