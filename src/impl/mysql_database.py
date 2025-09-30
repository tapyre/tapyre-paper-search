import os
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import SQLAlchemyError
from src.models.paper import Paper
from src.models.base import Base
from src.core.database import Database

class MySQLDatabase(Database):
    def __init__(self):
        user = os.getenv("MYSQL_USER")
        password = os.getenv("MYSQL_PASSWORD")
        host = os.getenv("MYSQL_HOST", "mysql_db")
        database = os.getenv("MYSQL_DATABASE")

        if not all([user, password, database]):
            raise ValueError("Missing one or more required environment variables: MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE")

        self.db_url = f"mysql+pymysql://{user}:{password}@{host}/{database}"
        # self.db_url = "mysql+pymysql://root:root@127.0.0.1:3306/test"

        self.engine = None
        self.Session = None
        super().__init__()

    def connect(self):
        self.engine = create_engine(self.db_url, echo=True)
        self.Session = scoped_session(sessionmaker(bind=self.engine))
        Base.metadata.create_all(self.engine)

    def disconnect(self):
        if self.Session:
            self.Session.remove()
        if self.engine:
            self.engine.dispose()

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
            return arxiv_id
        except SQLAlchemyError as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_paper_by_arxiv_id(self, arxiv_id: str):
        session = self.get_session()
        try:
            return session.query(Paper).filter_by(arxiv_id=arxiv_id).first()
        finally:
            session.close()

    def get_all_papers(self):
        session = self.get_session()
        try:
            return session.query(Paper).all()
        finally:
            session.close()

    def delete_paper(self, arxiv_id: str):
        session = self.get_session()
        try:
            paper = session.query(Paper).filter_by(arxiv_id=arxiv_id).first()
            if paper:
                session.delete(paper)
                session.commit()
                return True
            return False
        except SQLAlchemyError:
            session.rollback()
            return False
        finally:
            session.close()