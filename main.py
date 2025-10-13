from src.impl.arxiv_data_provider import ArxivDataProvider
from src.impl.fitz_pdf_converter import FitzPdfConverter
from src.impl.specter_2_embedder import Specter2Embedder 
from src.impl.mysql_database import MySQLDatabase
from src.impl.qdrant_database import QdrantDatabase
from src.impl.pipeline import Pipeline
from api import StatisticsAPI
import os


def main():
    print("              _____                                              ")
    print("             |_   _|_ _ _ __  _   _ _ __ ___                     ")
    print("               | |/ _` | '_ \\| | | | '__/ _ \\                    ")
    print("               | | (_| | |_) | |_| | | |  __/                    ")
    print("               |_|\\__,_| .__/ \\__, |_|  \\___|                    ")
    print("  ____                 |_|    |___/__                      _     ")
    print(" |  _ \\ __ _ _ __   ___ _ __    / ___|  ___  __ _ _ __ ___| |__  ")
    print(" | |_) / _` | '_ \\ / _ \\ '__|___\\___ \\ / _ \\/ _` | '__/ __| '_ \\ ")
    print(" |  __/ (_| | |_) |  __/ | |_____|__) |  __/ (_| | | | (__| | | |")
    print(" |_|   \\__,_| .__/ \\___|_|      |____/ \\___|\\__,_|_|  \\___|_| |_|")
    print("            |_|                                                  ")
    print("Starting MySQL")
    mysql_db = MySQLDatabase() 
    print("Starting Qdrant")
    qdrant_db = QdrantDatabase()
    print("Starting Embedder")
    embedder = Specter2Embedder()

    print("Starting API")
    api = StatisticsAPI(mysql_db, qdrant_db, embedder, port=8000)
    api.run()

    # print("Starting Arxiv Data Provider")
    # data_provider = ArxivDataProvider(first_id="arXiv:2405.00001", last_id="arXiv:2512.99999", rate_limit_seconds=3.0)
    # print("Starting PDF Converter")
    # pdf_converter = FitzPdfConverter(

    # # app.run(debug=True, host='0.0.0.0', port=8000, threaded=True)

    # print("Activated pipeline")
    # Pipeline(mysql_db, qdrant_db, data_provider, pdf_converter, embedder).process()


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
