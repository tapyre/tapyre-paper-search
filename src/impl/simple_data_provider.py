from src.core.data_provider import DataProvider
from src.impl.logger import get_logger
import os


class SimpleDataProvider(DataProvider):
    def __init__(self, path="data"):
        self.logger = get_logger(__name__)
        self.path = path
        self.data = []

        self.logger.info("[SimpleDataProvider] Initializing with path: %s", self.path)

        if not os.path.exists(self.path):
            self.logger.warning("[SimpleDataProvider] Path does not exist: %s", self.path)
            return

        try:
            files = os.listdir(self.path)
            pdf_files = [f for f in files if f.lower().endswith(".pdf")]
            self.logger.info("[SimpleDataProvider] Found %d PDF files in %s", len(pdf_files), self.path)

            for filename in pdf_files:
                full_path = os.path.join(self.path, filename)
                try:
                    with open(full_path, "rb") as f:
                        pdf_data = f.read()
                    arxiv_id = filename.lower()[:-4]
                    self.data.append((arxiv_id, pdf_data))
                    self.logger.debug("[SimpleDataProvider] Loaded PDF: %s (size: %d bytes)", filename, len(pdf_data))
                except Exception as e:
                    self.logger.exception("[SimpleDataProvider] Failed to load file %s: %s", filename, e)

            self.logger.info("[SimpleDataProvider] Loaded %d valid PDF files.", len(self.data))
        except Exception as e:
            self.logger.exception("[SimpleDataProvider] Error reading directory %s: %s", self.path, e)

    def next(self):
        if self.data:
            next_item = self.data.pop(0)
            self.logger.debug("[SimpleDataProvider] Returning next item: %s (remaining: %d)", next_item[0], len(self.data))
            return next_item
        self.logger.info("[SimpleDataProvider] No more data available.")
        return None

    def hasNext(self) -> bool:
        has_next = len(self.data) > 0
        self.logger.debug("[SimpleDataProvider] hasNext() -> %s", has_next)
        return has_next
