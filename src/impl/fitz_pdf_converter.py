from src.core.pdf_converter import PdfConverter
from src.impl.logger import get_logger
import fitz
from datetime import datetime, timedelta, timezone
import re
from typing import Optional
import unicodedata



class FitzPdfConverter(PdfConverter):
    def __init__(self):
        self.logger = get_logger(__name__)
        self.logger.debug("FitzPdfConverter initialized.")

    def pdf_to_string(self, pdf) -> str:
        if isinstance(pdf, str):
            self.logger.info("Opening PDF from file path.")
        elif isinstance(pdf, bytes):
            self.logger.info("Opening PDF from bytes stream.")
            self.logger.debug("PDF bytes length: %s", len(pdf))
        else:
            self.logger.error("Invalid input type for pdf_to_string: %s", type(pdf))
            raise ValueError("Input must be a file path or PDF binary content")

        doc = None
        try:
            if isinstance(pdf, str):
                doc = fitz.open(pdf)
            else:
                doc = fitz.open(stream=pdf, filetype="pdf")

            self.logger.debug("PDF opened successfully. Page count: %d", len(doc))
            full_text = ""
            for i, page in enumerate(doc, start=1):
                text = page.get_text()
                full_text += text
                if i % 10 == 0 or i == len(doc):
                    self.logger.debug("Extracted text up to page %d/%d (current chunk length=%d).",
                                      i, len(doc), len(text))

            self.logger.info("Finished extracting text from PDF. Total length: %d characters.",
                             len(full_text))
            return full_text
        finally:
            if doc is not None:
                doc.close()
                self.logger.debug("PDF document closed after text extraction.")

    def pdf_metadata(self, pdf) -> dict:
        if isinstance(pdf, str):
            self.logger.info("Opening PDF from file path for metadata.")
        elif isinstance(pdf, bytes):
            self.logger.info("Opening PDF from bytes stream for metadata.")
            self.logger.debug("PDF bytes length: %s", len(pdf))
        else:
            self.logger.error("Invalid input type for pdf_metadata: %s", type(pdf))
            raise ValueError("Input must be a file path or PDF binary content")

        doc = None
        try:
            if isinstance(pdf, str):
                doc = fitz.open(pdf)
            else:
                doc = fitz.open(stream=pdf, filetype="pdf")

            metadata = doc.metadata or {}
            self.logger.debug("Raw metadata keys: %s", list(metadata.keys()))

            expected_keys = [
                'title', 'author', 'subject', 'keywords',
                'creator', 'producer', 'creationDate',
                'modDate', 'trapped'
            ]

            complete_metadata = {key: metadata.get(key) for key in expected_keys}

            if complete_metadata.get('creationDate'):
                original = complete_metadata['creationDate']
                parsed = self.parse_pdf_date(original)
                complete_metadata['creationDate'] = parsed
                self.logger.debug("Parsed creationDate: %s -> %s", original, parsed)

            if complete_metadata.get('modDate'):
                original = complete_metadata['modDate']
                parsed = self.parse_pdf_date(original)
                complete_metadata['modDate'] = parsed
                self.logger.debug("Parsed modDate: %s -> %s", original, parsed)

            self.logger.info("Metadata extracted successfully.")
            return complete_metadata
        finally:
            if doc is not None:
                doc.close()
                self.logger.debug("PDF document closed after metadata extraction.")

    def clean_string(self, text: str) -> str:
        before_len = len(text)
        text = unicodedata.normalize("NFKC", text)

        text = text.replace('\r\n', '\n').replace('\r', '\n')

        WHITESPACE_CHARS = [
            "\u00A0", 
            "\u2007", 
            "\u202F", 
            "\u2009", 
            "\u2002", "\u2003", "\u2004", "\u2005", "\u2006", 
            "\u2008", "\u200A",
            "\u3000", 
            "\u180E", 
            "\u200B", "\u200C", "\u200D", "\u2060", 
        ]
        for ch in WHITESPACE_CHARS:
            text = text.replace(ch, ' ')

        text = text.replace('–', '-')   
        text = text.replace('—', '-')   
        text = text.replace('−', '-')   
        text = text.replace('­', '')    

        text = text.replace('\n', ' ')


        text = ' '.join(text.split())
        after_len = len(text)
        self.logger.debug("Cleaned string: length %d -> %d.", before_len, after_len)
        return text

    def chunk_string(self, text: str, chunk_size=500, overlap=50) -> list[str]:
        self.logger.info("Chunking text with chunk_size=%d and overlap=%d.", chunk_size, overlap)
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunks.append(text[start:end])
            start += chunk_size - overlap

        self.logger.info("Produced %d chunks from text of length %d.", len(chunks), text_len)
        return chunks

    def parse_pdf_date(self, pdf_date_str) -> Optional[datetime]:
        if not pdf_date_str or not str(pdf_date_str).startswith('D:'):
            self.logger.warning("PDF date string missing or not starting with 'D:': %s", pdf_date_str)
            return None

        original = pdf_date_str
        pdf_date_str = pdf_date_str[2:]

        match = re.match(
            r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})([+-])(\d{2})'?(\d{2})?",
            pdf_date_str
        )
        if not match:
            self.logger.warning("Failed to parse PDF date string: %s", original)
            return None

        year, month, day, hour, minute, second, tz_sign, tz_hour, tz_minute = match.groups()
        dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))

        offset = timedelta(hours=int(tz_hour or 0), minutes=int(tz_minute or 0))
        if tz_sign == '-':
            offset = -offset

        parsed = dt.replace(tzinfo=timezone(offset))
        self.logger.debug("Parsed PDF date '%s' into datetime '%s'.", original, parsed)
        return parsed
