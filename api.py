from src.controllers.statistics_controller import StatisticsController
from src.controllers.chunk_controller import ChunksController
from src.controllers.paper_controller import PapersController
from src.impl.logger import get_logger
from flask import Flask, Blueprint, jsonify
from src.impl.specter_2_embedder import Specter2Embedder 
from src.impl.mysql_database import MySQLDatabase
from src.impl.qdrant_database import QdrantDatabase

class StatisticsAPI:
    def __init__(self, mysql_db, qdrant_db, embedder, host='0.0.0.0', port=5000):
        self.logger = get_logger(__name__)
        self.logger.info("Initializing StatisticsAPI...")

        self.mysql_db = mysql_db
        self.qdrant_db = qdrant_db
        self.host = host
        self.port = port

        self.app = Flask(__name__)

        @self.app.route("/", methods=["GET"])
        def health():
            try:
                self.logger.debug("[health] Checking service status...")
                mysql_ok = self.mysql_db.ping() if hasattr(self.mysql_db, "ping") else True
                qdrant_ok = self.qdrant_db.ping() if hasattr(self.qdrant_db, "ping") else True

                status = {
                    "status": "ok" if mysql_ok and qdrant_ok else "degraded",
                    "mysql": "up" if mysql_ok else "down",
                    "qdrant": "up" if qdrant_ok else "down",
                }

                self.logger.info("[health] Healthcheck OK")
                return jsonify(status), 200 if mysql_ok and qdrant_ok else 503

            except Exception as e:
                self.logger.error("[health] Healthcheck failed", exc_info=True)
                return jsonify({
                    "status": "error",
                    "details": str(e)
                }), 500

        self.logger.info("register StatisticsController...")
        self.statistics_controller = StatisticsController(mysql_db, qdrant_db)
        self.app.register_blueprint(StatisticsController.blueprint)

        self.logger.info("register ChunksController...")
        self.chunks_controller = ChunksController(qdrant_db, embedder)
        self.app.register_blueprint(ChunksController.blueprint)

        self.logger.info("register PapersController...")
        self.papers_controller = PapersController(qdrant_db, embedder, mysql_db)
        self.app.register_blueprint(PapersController.blueprint)

        self.logger.info("All controllers registered successfully.")

    def run(self):
        print("  _____                          ")
        print(" |_   _|_ _ _ __  _   _ _ __ ___ ")
        print("   | |/ _` | '_ \\| | | | '__/ _ \\")
        print("   | | (_| | |_) | |_| | | |  __/")
        print("   |_|\\__,_| .__/ \\__, |_|  \\___|")
        print("           |_|  __|___/_         ")
        print("          / \\  |  _ \\_ _|        ")
        print("         / _ \\ | |_) | |         ")
        print("        / ___ \\|  __/| |         ")
        print("       /_/   \\_\\_|  |___|        ")
        print("                                 ")

        self.logger.info(f"Starting Flask API on {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port)

def main():
    mysql_db = MySQLDatabase() 
    qdrant_db = QdrantDatabase()
    embedder = Specter2Embedder()
    
    api = StatisticsAPI(mysql_db, qdrant_db, embedder, port=8000)
    api.run()

if __name__ == "__main__":
    main()