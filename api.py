from src.controllers.statistics_controller import StatisticsController
from src.controllers.chunk_controller import ChunksController
from src.controllers.paper_controller import PapersController
from flask import Flask, Blueprint, jsonify

class StatisticsAPI:
    def __init__(self, mysql_db, qdrant_db, embedder, host='0.0.0.0', port=5000):
        self.mysql_db = mysql_db
        self.qdrant_db = qdrant_db
        self.host = host
        self.port = port

        self.app = Flask(__name__)

        self.statistics_controller = StatisticsController(mysql_db, qdrant_db)
        self.app.register_blueprint(StatisticsController.blueprint)

        self.chunks_controller = ChunksController(qdrant_db, embedder)
        self.app.register_blueprint(ChunksController.blueprint)

        self.papers_controller = PapersController(qdrant_db, embedder, mysql_db)
        self.app.register_blueprint(PapersController.blueprint)

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
        self.app.run(host=self.host, port=self.port)