from flask import Blueprint, jsonify

class StatisticsController:
    blueprint = Blueprint('statistics', __name__)

    def __init__(self, mysql_db, qdrant_db):
        self.mysql_db = mysql_db
        self.qdrant_db = qdrant_db

        StatisticsController.blueprint.add_url_rule(
            '/statistics',
            view_func=self.get_statistics_route,
            methods=['GET']
        )

    def get_statistics(self):
        mysql_stats = self.mysql_db.get_statistics()
        qdrant_stats = self.qdrant_db.get_statistics()
        statistics = {
            "mysql": mysql_stats,
            "qdrant": qdrant_stats
        }
        return statistics

    def get_statistics_route(self):
        stats = self.get_statistics()
        return jsonify(stats)
