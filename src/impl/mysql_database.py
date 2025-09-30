import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import SQLAlchemyError
from src.models.chunk import Chunk
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
        # self.db_url = f"mysql+pymysql://root:rootpassword@127.0.0.1:3306/test

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

    def add_chunk(self, text: str, arxiv_id: str = None):
        session = self.get_session()
        try:
            chunk = Chunk(text=text, arxiv_id=arxiv_id)
            _uuid = chunk.get_uuid()
            session.add(chunk)
            session.commit()
            return _uuid
        except SQLAlchemyError as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_chunk_by_uuid(self, uuid: str):
        session = self.get_session()
        try:
            return session.query(Chunk).filter_by(uuid=uuid).first()
        finally:
            session.close()

    def get_all_chunks(self):
        session = self.get_session()
        try:
            return session.query(Chunk).all()
        finally:
            session.close()

    def delete_chunk(self, uuid: str):
        session = self.get_session()
        try:
            chunk = session.query(Chunk).filter_by(uuid=uuid).first()
            if chunk:
                session.delete(chunk)
                session.commit()
                return True
            return False
        except SQLAlchemyError:
            session.rollback()
            return False
        finally:
            session.close()
