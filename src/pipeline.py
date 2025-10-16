from src.impl.arxiv_data_provider import ArxivDataProvider
from src.impl.fitz_pdf_converter import FitzPdfConverter
from src.impl.specter_2_embedder import Specter2Embedder 
from src.impl.mysql_database import MySQLDatabase
from src.impl.qdrant_database import QdrantDatabase
from src.impl.pipeline import Pipeline
from src.impl.logger import get_logger
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

    logger = get_logger(__name__)

    logger.info("Starting MySQL...")
    mysql_db = MySQLDatabase() 

    logger.info("Starting Qdrant...")
    qdrant_db = QdrantDatabase()

    logger.info("Starting Arxiv Data Provider...")
    data_provider = ArxivDataProvider(
        first_id="arXiv:2405.00010",
        last_id="arXiv:2512.99999",
        rate_limit_seconds=3.0
    )

    logger.info("Starting PDF Converter...")
    pdf_converter = FitzPdfConverter()

    logger.info("Starting Embedder...")
    embedder = Specter2Embedder()

    logger.info("Activating pipeline...")
    Pipeline(mysql_db, qdrant_db, data_provider, pdf_converter, embedder).process()


if __name__ == "__main__":
    main()
