from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from sqlalchemy.ext.declarative import declarative_base
import uuid
from src.models.base import Base
from datetime import datetime

class Paper(Base):
    __tablename__ = 'papers'

    ## Primary fields
    arxiv_id = Column(String(255), primary_key=True)
    text = Column(MEDIUMTEXT, nullable=False)

    ## Metadata fields  
    title = Column(String(255), nullable=True)
    author = Column(String(255), nullable=True)
    subject = Column(String(255), nullable=True)
    keywords = Column(Text, nullable=True)
    creator = Column(String(255), nullable=True)
    producer = Column(String(255), nullable=True)
    
    creation_date = Column(DateTime, nullable=True)
    modification_date = Column(DateTime, nullable=True)
    processed_at = Column(DateTime, nullable=False)

    trapped = Column(String(50), nullable=True)
    
    def __init__(
        self,
        arxiv_id,
        text,
        title=None,
        author=None,
        subject=None,
        keywords=None,
        creator=None,
        producer=None,
        creation_date=None,
        modification_date=None,
        trapped=None
    ):
        self.arxiv_id = arxiv_id
        self.text = text
        self.title = title
        self.author = author
        self.subject = subject
        self.keywords = keywords
        self.creator = creator
        self.producer = producer
        self.creation_date = None if creation_date == '' else creation_date
        self.modification_date = None if modification_date == '' else modification_date
        self.processed_at = datetime.utcnow()
        self.trapped = trapped