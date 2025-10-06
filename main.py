from src.impl.arxiv_data_provider import ArxivDataProvider
from src.impl.fitz_pdf_converter import FitzPdfConverter
from src.impl.specter_2_embedder import Specter2Embedder 
from src.impl.mysql_database import MySQLDatabase
from src.impl.qdrant_database import QdrantDatabase
from src.impl.pipeline import Pipeline
# from api import app
import os


def main():
    print("Starting main.py...")
    print("Init My")
    mysql_db = MySQLDatabase() 
    qdrant_db = QdrantDatabase()
    data_provider = ArxivDataProvider(first_id="arXiv:0701.00001", last_id="arXiv:2512.99999", rate_limit_seconds=3.0)
    pdf_converter = FitzPdfConverter()
    embedder = Specter2Embedder()

    # app.run(debug=True, host='0.0.0.0', port=8000, threaded=True)

    print("Activated pipeline")
    Pipeline(mysql_db, qdrant_db, data_provider, pdf_converter, embedder).process()


    # while data_provider.hasNext():
    #     arxiv_id, pdf_data = data_provider.next()
    #     text = pdf_converter.pdf_to_string(pdf_data)
    #     cleaned_text = pdf_converter.clean_string(text)
    #     chunks = pdf_converter.chunk_string(cleaned_text)
    #     print("Chunks: ", len(chunks))
    #     for chunk in chunks:
    #         embedding = embedder.embed(chunk)
    #         chunk_uuid = mysql_db.add_chunk(text=chunk, arxiv_id=arxiv_id) 
    #         qdrant_db.add_chunk(uuid=chunk_uuid, embedding=embedding.tolist())
    #         print(f"Added chunk with UUID: {chunk_uuid}")

if __name__ == "__main__":
    main()
