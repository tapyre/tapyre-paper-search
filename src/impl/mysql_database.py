import os
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import SQLAlchemyError

from src.models.paper import Paper
from src.models.base import Base
from src.core.database import Database
from src.impl.logger import get_logger


class MySQLDatabase(Database):
    def __init__(self):
        self.logger = get_logger(__name__)
        self.logger.info("[MySQLDatabase] Initialization started")

        # user = os.getenv("MYSQL_USER")
        # password = os.getenv("MYSQL_PASSWORD")
        # host = os.getenv("MYSQL_HOST", "mysql_db")
        # database = os.getenv("MYSQL_DATABASE")

        # self.logger.debug("[MySQLDatabase] Env vars - USER: %s, HOST: %s, DB: %s", user, host, database)

        # if not all([user, password, database]):
        #     raise ValueError("[MySQLDatabase] Missing env vars: MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE")

        # self.db_url = f"mysql+pymysql://{user}:{password}@{host}/{database}"
        self.db_url = "mysql+pymysql://root:root@127.0.0.1:3306/test"
        self.logger.info("[MySQLDatabase] Connection URL: %s", self.db_url)

        self.engine = None
        self.Session = None

        try:
            super().__init__()
            self.logger.info("[MySQLDatabase] super().__init__ successful")
        except Exception as e:
            self.logger.exception("[MySQLDatabase] Error calling super(): %s", e)
            raise

        try:
            self.logger.info("[MySQLDatabase] Connection established successfully")
        except Exception as e:
            self.logger.exception("[MySQLDatabase] Error while establishing connection: %s", e)
            raise

    def connect(self):
        self.logger.info("[MySQLDatabase] Creating engine and session...")
        self.engine = create_engine(self.db_url, echo=True)
        self.Session = scoped_session(sessionmaker(bind=self.engine))
        self.logger.info("[MySQLDatabase] Engine and Session created. Creating tables...")
        Base.metadata.create_all(self.engine)
        self.logger.info("[MySQLDatabase] Tables created")

    def disconnect(self):
        self.logger.info("[MySQLDatabase] Disconnecting from database...")
        if self.Session:
            self.Session.remove()
        if self.engine:
            self.engine.dispose()
        self.logger.info("[MySQLDatabase] Disconnected")

    def get_session(self):
        return self.Session()

    def add_paper(
        self,
        arxiv_id: str,
        text: str,
        title: str = None,
        author: str = None,
        subject: str = None,
        keywords: str = None,
        creator: str = None,
        producer: str = None,
        creation_date: datetime = None,
        modification_date: datetime = None,
        trapped: str = None
    ):
        session = self.get_session()
        try:
            self.logger.info("[MySQLDatabase] Inserting paper: %s", arxiv_id)
            paper = Paper(
                arxiv_id=arxiv_id,
                text=text,
                title=title,
                author=author,
                subject=subject,
                keywords=keywords,
                creator=creator,
                producer=producer,
                creation_date=creation_date,
                modification_date=modification_date,
                trapped=trapped
            )
            session.add(paper)
            session.commit()
            self.logger.info("[MySQLDatabase] Paper saved: %s", arxiv_id)
            return arxiv_id
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.exception("[MySQLDatabase] Error saving paper %s: %s", arxiv_id, e)
            raise
        finally:
            session.close()

    def get_paper_by_arxiv_id(self, arxiv_id: str):
        session = self.get_session()
        try:
            self.logger.info("[MySQLDatabase] Fetching paper by ID: %s", arxiv_id)
            return session.query(Paper).filter_by(arxiv_id=arxiv_id).first()
        finally:
            session.close()

    def get_all_papers(self):
        session = self.get_session()
        try:
            self.logger.info("[MySQLDatabase] Fetching all papers")
            return session.query(Paper).all()
        finally:
            session.close()

    def delete_paper(self, arxiv_id: str):
        session = self.get_session()
        try:
            self.logger.info("[MySQLDatabase] Deleting paper with ID: %s", arxiv_id)
            paper = session.query(Paper).filter_by(arxiv_id=arxiv_id).first()
            if paper:
                session.delete(paper)
                session.commit()
                self.logger.info("[MySQLDatabase] Paper deleted: %s", arxiv_id)
                return True
            self.logger.info("[MySQLDatabase] Paper not found: %s", arxiv_id)
            return False
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.exception("[MySQLDatabase] Error deleting paper %s: %s", arxiv_id, e)
            return False
        finally:
            session.close()

    def get_statistics(self):
        session = self.get_session()
        try:
            self.logger.info("[MySQLDatabase] Gathering database statistics")
            total_papers = session.query(Paper).count()
            latest_paper = session.query(Paper).order_by(Paper.creation_date.desc()).first()
            earliest_paper = session.query(Paper).order_by(Paper.creation_date.asc()).first()

            # Use DB name from the engine URL (or fall back to env var)
            db_name = (self.engine.url.database if self.engine else None) or os.getenv("MYSQL_DATABASE")

            db_size = None
            if self.engine and db_name:
                with self.engine.connect() as conn:
                    result = conn.execute(
                        text("""
                            SELECT ROUND(SUM(data_length + index_length) / 1024 / 1024, 2) AS size_mb
                            FROM information_schema.tables
                            WHERE table_schema = :db
                        """),
                        {"db": db_name}
                    )
                    db_size = result.scalar_one_or_none()

            stats = {
                "total_papers": total_papers,
                "latest_paper_date": latest_paper.creation_date if latest_paper else None,
                "earliest_paper_date": earliest_paper.creation_date if earliest_paper else None,
                "database_size_mb": db_size
            }
            self.logger.info("[MySQLDatabase] Statistics: %s", stats)
            return stats
        finally:
            session.close()
