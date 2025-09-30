from abc import ABC, abstractmethod

class PdfConverter(ABC):
    @abstractmethod
    def pdf_to_string(self, pdf) -> str:
        """Extracts text from a PDF file and returns it as a string."""
        pass

    @abstractmethod
    def clean_string(self, text: str) -> str:
        """Cleans the extracted text and returns the cleaned string."""
        pass

    @abstractmethod
    def chunk_string(self, text: str) -> list[str]:
        """Chunks the cleaned text into smaller segments."""
        pass