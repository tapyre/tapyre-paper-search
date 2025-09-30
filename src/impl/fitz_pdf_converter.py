from src.core.pdf_converter import PdfConverter
import fitz 

class FitzPdfConverter(PdfConverter):
    def pdf_to_string(self, pdf) -> str:
        if isinstance(pdf, str):
            doc = fitz.open(pdf)
        elif isinstance(pdf, bytes):
            doc = fitz.open(stream=pdf, filetype="pdf")
        else:
            raise ValueError("Input must be a file path or PDF binary content")

        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        return full_text

    def clean_string(self, text: str) -> str:
        text = text.replace('\n', ' ')
        text = ' '.join(text.split())
        return text

    def chunk_string(self, text: str, chunk_size=500, overlap=50) -> list[str]:
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks
