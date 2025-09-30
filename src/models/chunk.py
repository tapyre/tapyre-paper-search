from sqlalchemy import Column, String, Text
from sqlalchemy.ext.declarative import declarative_base
import uuid
from src.models.base import Base

class Chunk(Base):
    __tablename__ = 'chunks'

    uuid = Column(String(36), primary_key=True)
    text = Column(Text, nullable=False)
    arxiv_id = Column(String(255), nullable=False)

    def __init__(self, text: str, arxiv_id: str = None):
        self.uuid = str(uuid.uuid4())
        self.text = text
        self.arxiv_id = arxiv_id

    def get_uuid(self):
        return self.uuid