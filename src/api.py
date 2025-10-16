# src/api.py
from src.controllers.statistics_controller import StatisticsController
from src.controllers.chunk_controller import ChunksController
from src.controllers.paper_controller import PapersController
from src.impl.logger import get_logger
from flask import Flask, jsonify
from src.impl.specter_2_embedder import Specter2Embedder
from src.impl.mysql_database import MySQLDatabase
from src.impl.qdrant_database import QdrantDatabase
import os

def create_app():
    logger = get_logger(__name__)
    logger.info("Initializing StatisticsAPI...")

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))

    mysql_db = MySQLDatabase()
    qdrant_db = QdrantDatabase()
    embedder = Specter2Embedder()

    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def health():
        try:
            logger.debug("[health] Checking service status...")
            mysql_ok = mysql_db.ping() if hasattr(mysql_db, "ping") else True
            qdrant_ok = qdrant_db.ping() if hasattr(qdrant_db, "ping") else True

            status = {
                "status": "ok" if mysql_ok and qdrant_ok else "degraded",
                "mysql": "up" if mysql_ok else "down",
                "qdrant": "up" if qdrant_ok else "down",
            }
            logger.info("[health] Healthcheck OK")
            return jsonify(status), 200 if mysql_ok and qdrant_ok else 503
        except Exception as e:
            logger.error("[health] Healthcheck failed", exc_info=True)
            return jsonify({"status": "error", "details": str(e)}), 500

    logger.info("register StatisticsController...")
    statistics_controller = StatisticsController(mysql_db, qdrant_db)  # noqa: F841
    app.register_blueprint(StatisticsController.blueprint)

    logger.info("register ChunksController...")
    chunks_controller = ChunksController(qdrant_db, embedder)  # noqa: F841
    app.register_blueprint(ChunksController.blueprint)

    logger.info("register PapersController...")
    papers_controller = PapersController(qdrant_db, embedder, mysql_db)  # noqa: F841
    app.register_blueprint(PapersController.blueprint)

    logger.info("All controllers registered successfully.")
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host=os.getenv("API_HOST", "0.0.0.0"), port=int(os.getenv("API_PORT", "8000")))
