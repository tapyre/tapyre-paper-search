from flask import Blueprint, jsonify
from src.impl.logger import get_logger


class StatisticsController:
    blueprint = Blueprint('statistics', __name__)

    def __init__(self, mysql_db, qdrant_db):
        self.logger = get_logger(__name__)
        self.logger.info("Initializing StatisticsController")

        self.mysql_db = mysql_db
        self.qdrant_db = qdrant_db

        StatisticsController.blueprint.add_url_rule(
            '/statistics',
            view_func=self.get_statistics_route,
            methods=['GET']
        )
        self.logger.info("StatisticsController route '/statistics' registered")

    def get_statistics(self):
        try:
            self.logger.debug("[statistics] Fetching MySQL stats...")
            mysql_stats = self.mysql_db.get_statistics()

            self.logger.debug("[statistics] Fetching Qdrant stats...")
            qdrant_stats = self.qdrant_db.get_statistics()

            statistics = {
                "mysql": mysql_stats,
                "qdrant": qdrant_stats
            }
            self.logger.info("[statistics] Statistics fetched successfully")
            return statistics

        except Exception as e:
            self.logger.error("[statistics] Error while fetching statistics", exc_info=True)
            raise

    def get_statistics_route(self):
        try:
            self.logger.info("[route:statistics] Request received")
            stats = self.get_statistics()
            self.logger.info("[route:statistics] Returning statistics response")
            return jsonify(stats)
        except Exception as e:
            self.logger.error("[route:statistics] Internal server error", exc_info=True)
            return jsonify({"error": "Internal server error", "details": str(e)}), 500
