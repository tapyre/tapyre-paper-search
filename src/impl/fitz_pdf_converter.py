from src.core.pdf_converter import PdfConverter
import fitz 
from datetime import datetime, timedelta, timezone
import re

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

    def pdf_metadata(self, pdf) -> dict:
        if isinstance(pdf, str):
            doc = fitz.open(pdf)
        elif isinstance(pdf, bytes):
            doc = fitz.open(stream=pdf, filetype="pdf")
        else:
            raise ValueError("Input must be a file path or PDF binary content")

        metadata = doc.metadata 
        doc.close()

        expected_keys = [
            'title', 'author', 'subject', 'keywords',
            'creator', 'producer', 'creationDate',
            'modDate', 'trapped'
        ]

        complete_metadata = {key: metadata.get(key) for key in expected_keys}
        
        # Convert date strings to datetime objects
        if complete_metadata.get('creationDate'):
            complete_metadata['creationDate'] = self.parse_pdf_date(complete_metadata['creationDate'])
        if complete_metadata.get('modDate'):
            complete_metadata['modDate'] = self.parse_pdf_date(complete_metadata['modDate'])
        
        return complete_metadata

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


    def parse_pdf_date(self, pdf_date_str):
        """
        Convert PDF-style date string like 'D:20250926011445+00\'00\'' to Python datetime
        """
        if not pdf_date_str or not pdf_date_str.startswith('D:'):
            return None

        pdf_date_str = pdf_date_str[2:]

        match = re.match(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})([+-])(\d{2})'?(\d{2})?", pdf_date_str)
        if not match:
            return None

        year, month, day, hour, minute, second, tz_sign, tz_hour, tz_minute = match.groups()
        dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))

        offset = timedelta(hours=int(tz_hour or 0), minutes=int(tz_minute or 0))
        if tz_sign == '-':
            offset = -offset

        return dt.replace(tzinfo=timezone(offset))