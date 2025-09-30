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
            arxiv_id, pdf_data = self.data_provider.next()
            text = self.pdf_converter.pdf_to_string(pdf_data)
            cleaned_text = self.pdf_converter.clean_string(text)
            chunks = self.pdf_converter.chunk_string(cleaned_text)
            print("Chunks: ", len(chunks))
            for chunk in chunks:
                embedding = self.embedder.embed(chunk)
                chunk_uuid = self.mysql_db.add_chunk(text=chunk, arxiv_id=arxiv_id) 
                self.qdrant_db.add_chunk(uuid=chunk_uuid, embedding=embedding.tolist())
                print(f"Added chunk with UUID: {chunk_uuid}")

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