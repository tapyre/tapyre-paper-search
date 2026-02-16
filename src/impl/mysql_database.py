import os
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, DataError

from src.models.paper import Paper
from src.models.base import Base
from src.core.database import Database
from src.impl.logger import get_logger


class MySQLDatabase(Database):
    """
    SQLAlchemy-backed MySQL database implementation for storing and retrieving Paper entities.

    Responsibilities:
    - Build the SQLAlchemy connection URL from environment variables
    - Create and manage the SQLAlchemy engine + session factory
    - Provide CRUD operations for Paper records
    - Provide basic database statistics for monitoring / dashboards

    Design goals:
    - Production stability: avoid crashes on common DB exceptions
    - Clean lifecycle: explicit connect()/disconnect()
    - Observability: detailed logging for debugging and operations
    - Predictable schema: create tables via Base.metadata on startup
    """

    def __init__(self):
        # Initialize a module-scoped logger for structured diagnostics
        self.logger = get_logger(__name__)
        self.logger.info("[MySQLDatabase] Initialization started")

        # Read required connection parameters from environment variables
        user = os.getenv("MYSQL_USER")
        password = os.getenv("MYSQL_PASSWORD")
        host = os.getenv("MYSQL_HOST", "mysql_db")
        database = os.getenv("MYSQL_DATABASE")

        # Log only safe connection details (avoid logging secrets such as the password)
        self.logger.debug("[MySQLDatabase] Env vars - USER: %s, HOST: %s, DB: %s", user, host, database)

        # Validate required config early to fail fast and avoid undefined runtime states
        if not all([user, password, database]):
            raise ValueError("[MySQLDatabase] Missing env vars: MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE")

        # Build SQLAlchemy URL (PyMySQL driver)
        self.db_url = f"mysql+pymysql://{user}:{password}@{host}/{database}"
        self.logger.info("[MySQLDatabase] Connection URL: %s", self.db_url)

        # Engine and session factory are created in connect()
        self.engine = None
        self.Session = None

        # Parent-class initialization hook
        # This is useful if the base Database implements shared logic or enforces interface behavior.
        try:
            super().__init__()
            self.logger.info("[MySQLDatabase] super().__init__ successful")
        except Exception as e:
            # If base init fails, this is typically fatal (misconfigured class lifecycle)
            self.logger.exception("[MySQLDatabase] Error calling super(): %s", e)
            raise

        # Note:
        # The current code logs "Connection established successfully" here,
        # but the actual connection is only created in connect().
        # We keep the logging, but operationally the DB is not connected yet.
        try:
            self.logger.info("[MySQLDatabase] Connection established successfully")
        except Exception as e:
            self.logger.exception("[MySQLDatabase] Error while establishing connection: %s", e)
            raise

    def connect(self):
        """
        Establish database connectivity by creating:
        - SQLAlchemy Engine (connection pool)
        - Scoped session factory (thread-safe session retrieval)
        - Database schema (tables) if not yet present

        Notes:
        - pool_pre_ping=True helps prevent stale connection errors
          by checking connections before using them.
        """
        self.logger.info("[MySQLDatabase] Creating engine and session...")

        # Create engine with connection pooling
        self.engine = create_engine(self.db_url, echo=False, pool_pre_ping=True)

        # Create a scoped session factory
        # scoped_session is helpful for multi-threaded environments.
        self.Session = scoped_session(sessionmaker(bind=self.engine))

        self.logger.info("[MySQLDatabase] Engine and Session created. Creating tables...")

        # Create all tables defined via SQLAlchemy declarative Base
        Base.metadata.create_all(self.engine)

        self.logger.info("[MySQLDatabase] Tables created")

    def disconnect(self):
        """
        Cleanly release DB resources.

        - Remove any scoped session references
        - Dispose the engine and its connection pool
        """
        self.logger.info("[MySQLDatabase] Disconnecting from database...")
        if self.Session:
            self.Session.remove()
        if self.engine:
            self.engine.dispose()
        self.logger.info("[MySQLDatabase] Disconnected")

    def get_session(self):
        """
        Obtain a new SQLAlchemy session from the scoped session factory.

        This method keeps session creation centralized and allows the rest of the code
        to treat session acquisition uniformly.
        """
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
        trapped: str = None,
        skip_if_exists: bool = True,
    ):
        """
        Insert a new Paper record into the database.

        Parameters:
        - arxiv_id: primary key identifier for the paper
        - text: extracted paper content (typically large)
        - metadata fields: optional descriptive fields from PDF/arXiv
        - skip_if_exists: avoid duplicate inserts (pre-check via paper_exists)

        Returns:
        - arxiv_id if insert succeeds (or paper already exists and skipping is enabled)
        - None if insert fails due to integrity, data, or SQLAlchemy errors

        Error handling policy:
        - IntegrityError: treat as "already exists / constraint violation" → skip
        - DataError: treat as "invalid column length/encoding" → skip
        - SQLAlchemyError: unexpected DB error → log with stack trace and skip
        """
        session = self.get_session()
        try:
            # Optional pre-check to avoid duplicate inserts
            # Note: this is not fully race-condition safe; unique constraints still matter.
            if skip_if_exists and self.paper_exists(arxiv_id):
                self.logger.info("[MySQLDatabase] Paper %s already exists, skipping insert.", arxiv_id)
                return arxiv_id

            self.logger.info("[MySQLDatabase] Inserting paper: %s", arxiv_id)

            # Construct ORM entity
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
                trapped=trapped,
            )

            # Persist record
            session.add(paper)
            session.commit()

            self.logger.info("[MySQLDatabase] Paper saved: %s", arxiv_id)
            return arxiv_id

        except IntegrityError as e:
            # Typical causes: duplicate primary key, unique constraint violations
            session.rollback()
            self.logger.warning("[MySQLDatabase] IntegrityError on %s, skipping. Detail: %s", arxiv_id, e)
            return None

        except DataError as e:
            # Typical causes: field too long for column, invalid encoding, malformed data
            session.rollback()
            author_len = len(author) if author is not None else 0
            self.logger.warning(
                "[MySQLDatabase] DataError on %s (author_len=%d), skipping. Detail: %s",
                arxiv_id, author_len, e
            )
            return None

        except SQLAlchemyError as e:
            # Catch-all for unexpected DB-layer exceptions
            session.rollback()
            self.logger.exception("[MySQLDatabase] Unexpected SQLAlchemyError on %s: %s", arxiv_id, e)
            return None

        finally:
            # Ensure sessions are closed to avoid connection pool exhaustion
            session.close()

    def get_paper_by_arxiv_id(self, arxiv_id: str):
        """
        Retrieve a single Paper by its primary key (arxiv_id).

        Returns:
        - Paper instance if found, else None
        """
        session = self.get_session()
        try:
            self.logger.info("[MySQLDatabase] Fetching paper by ID: %s", arxiv_id)
            return session.query(Paper).filter_by(arxiv_id=arxiv_id).first()
        finally:
            session.close()

    def get_all_papers(self):
        """
        Retrieve all Paper records.

        Warning:
        - This can be very expensive for large databases.
        - Prefer pagination or streaming for production-scale datasets.
        """
        session = self.get_session()
        try:
            self.logger.info("[MySQLDatabase] Fetching all papers")
            return session.query(Paper).all()
        finally:
            session.close()

    def delete_paper(self, arxiv_id: str):
        """
        Delete a Paper record by ID.

        Returns:
        - True if a record was found and deleted
        - False if not found or deletion failed
        """
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
            # Roll back transaction to keep session usable for further operations
            session.rollback()
            self.logger.exception("[MySQLDatabase] Error deleting paper %s: %s", arxiv_id, e)
            return False

        finally:
            session.close()

    def get_statistics(self):
        """
        Gather basic database statistics.

        Included metrics:
        - total_papers: number of Paper records
        - latest_paper_date: newest creation_date present in the table
        - earliest_paper_date: oldest creation_date present in the table
        - database_size_mb: approximate DB size via information_schema (MB)

        Returns:
        - dict containing computed statistics
        """
        session = self.get_session()
        try:
            self.logger.info("[MySQLDatabase] Gathering database statistics")

            # High-level record count
            total_papers = session.query(Paper).count()

            # Latest and earliest record by creation_date
            latest_paper = session.query(Paper).order_by(Paper.creation_date.desc()).first()
            earliest_paper = session.query(Paper).order_by(Paper.creation_date.asc()).first()

            # Resolve database name (prefer engine URL if available)
            db_name = (self.engine.url.database if self.engine else None) or os.getenv("MYSQL_DATABASE")

            # Compute DB size from MySQL metadata tables
            db_size = None
            if self.engine and db_name:
                with self.engine.connect() as conn:
                    result = conn.execute(
                        text("""
                            SELECT ROUND(SUM(data_length + index_length) / 1024 / 1024, 2) AS size_mb
                            FROM information_schema.tables
                            WHERE table_schema = :db
                        """),
                        {"db": db_name},
                    )
                    db_size = result.scalar_one_or_none()

            stats = {
                "total_papers": total_papers,
                "latest_paper_date": latest_paper.creation_date if latest_paper else None,
                "earliest_paper_date": earliest_paper.creation_date if earliest_paper else None,
                "database_size_mb": db_size,
            }

            self.logger.info("[MySQLDatabase] Statistics: %s", stats)
            return stats

        finally:
            session.close()

    def paper_exists(self, arxiv_id: str) -> bool:
        """
        Check whether a Paper exists by primary key.

        Implementation detail:
        - Uses session.get() which is optimized for PK lookup.

        Returns:
        - True if paper exists, else False
        """
        session = self.get_session()
        try:
            return session.get(Paper, arxiv_id) is not None
        finally:
            session.close()
