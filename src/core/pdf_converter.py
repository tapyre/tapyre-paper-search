from abc import ABC, abstractmethod

class PdfConverter(ABC):
    @abstractmethod
    def pdf_to_string(self, pdf) -> str:
        pass

    @abstractmethod
    def clean_string(self, text: str) -> str:
        pass

    @abstractmethod
    def chunk_string(self, text: str) -> list[str]:
        pass
        
    @abstractmethod
    def pdf_metadata(self, pdf) -> dict:
        pass