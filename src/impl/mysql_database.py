import os
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import SQLAlchemyError
from src.models.paper import Paper
from src.models.base import Base
from src.core.database import Database
from sqlalchemy import text
class MySQLDatabase(Database):
    def __init__(self):
        print("[MySQLDatabase] Initialisierung gestartet")

        # user = os.getenv("MYSQL_USER")
        # password = os.getenv("MYSQL_PASSWORD")
        # host = os.getenv("MYSQL_HOST", "mysql_db")
        # database = os.getenv("MYSQL_DATABASE")

        # print(f"[MySQLDatabase] Umgebungsvariablen - USER: {user}, HOST: {host}, DB: {database}")

        # if not all([user, password, database]):
        #     raise ValueError("[MySQLDatabase] Fehlende Umgebungsvariablen: MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE")

        # self.db_url = f"mysql+pymysql://{user}:{password}@{host}/{database}"
        self.db_url = f"mysql+pymysql://root:root@127.0.0.1:3306/test"
        print(f"[MySQLDatabase] Verbindungs-URL: {self.db_url}")

        self.engine = None
        self.Session = None

        try:
            super().__init__()
            print("[MySQLDatabase] Super init erfolgreich")
        except Exception as e:
            print(f"[MySQLDatabase] Fehler beim Aufruf von super(): {e}")
            raise

        try:
            print("[MySQLDatabase] Verbindung erfolgreich hergestellt")
        except Exception as e:
            print(f"[MySQLDatabase] Fehler beim Herstellen der Verbindung: {e}")
            raise

    def connect(self):
        print("[MySQLDatabase] Verbindung zur Datenbank wird aufgebaut...")
        self.engine = create_engine(self.db_url, echo=True)
        self.Session = scoped_session(sessionmaker(bind=self.engine))
        print("[MySQLDatabase] Engine und Session erstellt. Erstelle Tabellen...")
        Base.metadata.create_all(self.engine)
        print("[MySQLDatabase] Tabellen erstellt")

    def disconnect(self):
        print("[MySQLDatabase] Trenne Verbindung zur Datenbank...")
        if self.Session:
            self.Session.remove()
        if self.engine:
            self.engine.dispose()
        print("[MySQLDatabase] Verbindung getrennt")

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
            print(f"[MySQLDatabase] Füge Paper ein: {arxiv_id}")
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
            print(f"[MySQLDatabase] Paper gespeichert: {arxiv_id}")
            return arxiv_id
        except SQLAlchemyError as e:
            session.rollback()
            print(f"[MySQLDatabase] Fehler beim Speichern des Papers: {e}")
            raise
        finally:
            session.close()

    def get_paper_by_arxiv_id(self, arxiv_id: str):
        session = self.get_session()
        try:
            print(f"[MySQLDatabase] Suche Paper mit ID: {arxiv_id}")
            return session.query(Paper).filter_by(arxiv_id=arxiv_id).first()
        finally:
            session.close()

    def get_all_papers(self):
        session = self.get_session()
        try:
            print("[MySQLDatabase] Hole alle Papers")
            return session.query(Paper).all()
        finally:
            session.close()

    def delete_paper(self, arxiv_id: str):
        session = self.get_session()
        try:
            print(f"[MySQLDatabase] Lösche Paper mit ID: {arxiv_id}")
            paper = session.query(Paper).filter_by(arxiv_id=arxiv_id).first()
            if paper:
                session.delete(paper)
                session.commit()
                print("[MySQLDatabase] Paper gelöscht")
                return True
            print("[MySQLDatabase] Paper nicht gefunden")
            return False
        except SQLAlchemyError as e:
            session.rollback()
            print(f"[MySQLDatabase] Fehler beim Löschen des Papers: {e}")
            return False
        finally:
            session.close()
    def get_statistics(self):
        session = self.get_session()
        try:
            print("[MySQLDatabase] Sammle Statistiken über die Datenbank")
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
            print(f"[MySQLDatabase] Statistiken: {stats}")
            return stats
        finally:
            session.close()