from src.core.pdf_converter import PdfConverter
from src.impl.logger import get_logger
import fitz
from datetime import datetime, timedelta, timezone
import re
from typing import Optional
import unicodedata


class FitzPdfConverter(PdfConverter):
    """
    PDF converter implementation based on PyMuPDF (fitz).

    Responsibilities:
    - Open PDFs from either a file path (str) or raw PDF bytes (bytes)
    - Extract full text content in a robust, page-by-page manner
    - Extract and normalize PDF metadata (including date parsing)
    - Provide post-processing helpers for text normalization and chunking

    Primary design goals:
    - Pipeline stability: partial extraction failures must not crash processing
    - Reproducibility: deterministic cleaning/chunking behavior
    - Observability: detailed logging for debugging in long-running jobs
    """

    def __init__(self):
        # Logger for structured diagnostics (debug/info/warn/error)
        self.logger = get_logger(__name__)
        self.logger.debug("FitzPdfConverter initialized.")

    def pdf_to_string(self, pdf) -> str:
        """
        Extract all text from a PDF into a single string.

        Supported input types:
        - str   : interpreted as a file path
        - bytes : interpreted as raw PDF binary data

        Extraction strategy:
        - Open the document with PyMuPDF
        - Iterate pages sequentially
        - Append page text to a growing buffer
        - Continue gracefully if individual pages fail to decode

        Returns:
        - The concatenated text of all successfully processed pages
        """
        # Validate input type early to keep callsites simple and errors explicit
        if isinstance(pdf, str):
            self.logger.info("Opening PDF from file path.")
        elif isinstance(pdf, bytes):
            self.logger.info("Opening PDF from bytes stream.")
            self.logger.debug("PDF bytes length: %s", len(pdf))
        else:
            self.logger.error("Invalid input type for pdf_to_string: %s", type(pdf))
            raise ValueError("Input must be a file path or PDF binary content")

        doc = None

        # Accumulator for extracted text; note: for extremely large PDFs,
        # using a list + ''.join(...) can be more efficient than repeated concatenation.
        full_text = ""

        # Counter for pages that failed during extraction (non-fatal)
        error_pages = 0

        try:
            # Open the PDF using the appropriate method depending on input type
            if isinstance(pdf, str):
                doc = fitz.open(pdf)
            else:
                doc = fitz.open(stream=pdf, filetype="pdf")

            page_count = len(doc)
            self.logger.debug("PDF opened successfully. Page count: %d", page_count)

            # Extract text page-by-page to isolate failures and provide progress logging
            for i, page in enumerate(doc, start=1):
                try:
                    # PyMuPDF text extraction (plain text mode)
                    text = page.get_text()
                    full_text += text

                    # Log progress occasionally to avoid extremely verbose logs
                    if i % 10 == 0 or i == page_count:
                        self.logger.debug(
                            "Extracted text up to page %d/%d (current page length=%d).",
                            i, page_count, len(text)
                        )
                except Exception as e:
                    # Page extraction failures are logged and skipped to keep the pipeline running
                    error_pages += 1
                    self.logger.error(
                        "Failed to extract text from page %d/%d: %s – skipping this page.",
                        i, page_count, e
                    )
                    continue

            # Summarize extraction quality if page-level issues occurred
            if error_pages > 0:
                self.logger.warning(
                    "Text extraction finished with %d page errors (total pages=%d).",
                    error_pages, page_count
                )

            self.logger.info(
                "Finished extracting text from PDF. Total length: %d characters.",
                len(full_text)
            )
            return full_text

        finally:
            # Always close document handles to prevent resource leaks in batch jobs
            if doc is not None:
                doc.close()
                self.logger.debug("PDF document closed after text extraction.")

    def pdf_metadata(self, pdf) -> dict:
        """
        Extract a curated set of metadata fields from a PDF.

        Supported input types:
        - str   : interpreted as a file path
        - bytes : interpreted as raw PDF binary data

        Metadata handling strategy:
        - Read raw metadata from PyMuPDF (doc.metadata)
        - Normalize to a known key set for predictable downstream usage
        - Parse creation/modification date strings into datetime objects where possible

        Returns:
        - A dict containing only the expected metadata keys (missing keys -> None)
        - {} if the PDF cannot be opened
        """
        # Input validation mirrors pdf_to_string() for consistent behavior
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
            # Opening the PDF can fail for corrupted files or unsupported formats
            try:
                if isinstance(pdf, str):
                    doc = fitz.open(pdf)
                else:
                    doc = fitz.open(stream=pdf, filetype="pdf")
            except Exception as e:
                # Metadata extraction failures should not crash the overall pipeline
                self.logger.error(
                    "Failed to open PDF for metadata extraction: %s", e, exc_info=True
                )
                return {}

            # PyMuPDF returns a dict-like metadata structure (may be empty)
            metadata = doc.metadata or {}
            self.logger.debug("Raw metadata keys: %s", list(metadata.keys()))

            # Standardized key list ensures stable schema for consumers (DB, indexers, etc.)
            expected_keys = [
                'title', 'author', 'subject', 'keywords',
                'creator', 'producer', 'creationDate',
                'modDate', 'trapped'
            ]

            # Build a stable output dict (missing keys become None)
            complete_metadata = {key: metadata.get(key) for key in expected_keys}

            # Parse date strings into datetime objects when present
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
            # Ensure resources are always released
            if doc is not None:
                doc.close()
                self.logger.debug("PDF document closed after metadata extraction.")

    def clean_string(self, text: str) -> str:
        """
        Normalize and clean extracted PDF text for downstream NLP processing.

        Operations:
        - Unicode normalization (NFKC) to standardize composed characters
        - Normalize line endings (Windows/Mac -> Unix)
        - Replace common non-breaking and special whitespace characters with ' '
        - Normalize dash variants to '-' and remove soft hyphen artifacts
        - Replace newlines with spaces
        - Collapse repeated whitespace into single spaces

        Returns:
        - Cleaned, single-line text with stable spacing and fewer hidden characters
        """
        before_len = len(text)

        # Normalize unicode representations (e.g., compatibility forms)
        text = unicodedata.normalize("NFKC", text)

        # Unify line endings across platforms
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # List of known whitespace/formatting characters frequently seen in PDFs
        # These can break tokenization or cause subtle search mismatches.
        WHITESPACE_CHARS = [
            "\u00A0",  # NO-BREAK SPACE
            "\u2007",  # FIGURE SPACE
            "\u202F",  # NARROW NO-BREAK SPACE
            "\u2009",  # THIN SPACE
            "\u2002", "\u2003", "\u2004", "\u2005", "\u2006",  # EN/EM and other spaces
            "\u2008", "\u200A",
            "\u3000",  # IDEOGRAPHIC SPACE
            "\u180E",  # MONGOLIAN VOWEL SEPARATOR (historically used as whitespace)
            "\u200B", "\u200C", "\u200D", "\u2060",  # zero-width / word joiner chars
        ]

        # Replace special whitespace with normal spaces
        for ch in WHITESPACE_CHARS:
            text = text.replace(ch, ' ')

        # Normalize dash variants to a simple ASCII hyphen
        text = text.replace('–', '-')  # en dash
        text = text.replace('—', '-')  # em dash
        text = text.replace('−', '-')  # minus sign
        text = text.replace('­', '')   # soft hyphen (often invisible, breaks words)

        # Convert multi-line text into single-line representation
        text = text.replace('\n', ' ')

        # Collapse all repeated whitespace into single spaces
        text = ' '.join(text.split())

        after_len = len(text)
        self.logger.debug("Cleaned string: length %d -> %d.", before_len, after_len)
        return text

    def chunk_string(self, text: str, chunk_size=500, overlap=50) -> list[str]:
        """
        Split a long text into overlapping chunks.

        Typical use-cases:
        - Feeding chunks into embedding models (vector DB ingestion)
        - Limiting token length for LLM contexts
        - Enabling sliding-window retrieval

        Parameters:
        - chunk_size: maximum chunk length (characters)
        - overlap   : number of characters reused between consecutive chunks

        Returns:
        - List of chunk strings preserving original order
        """
        self.logger.info("Chunking text with chunk_size=%d and overlap=%d.", chunk_size, overlap)

        chunks = []
        start = 0
        text_len = len(text)

        # Sliding window chunking over character offsets
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunks.append(text[start:end])

            # Advance start while keeping overlap to maintain continuity
            start += chunk_size - overlap

        self.logger.info("Produced %d chunks from text of length %d.", len(chunks), text_len)
        return chunks

    def parse_pdf_date(self, pdf_date_str) -> Optional[datetime]:
        """
        Parse PDF date strings into timezone-aware datetime objects.

        Common PDF date format:
        - "D:YYYYMMDDHHmmSS+HH'mm" or "D:YYYYMMDDHHmmSS-HH'mm"

        Behavior:
        - Returns None if input is missing or not parseable
        - Returns a timezone-aware datetime if parsing succeeds

        Note:
        - Some PDFs provide incomplete date strings; this implementation
          expects full YYYYMMDDHHmmSS plus timezone.
        """
        # PDF date strings typically start with "D:"
        if not pdf_date_str or not str(pdf_date_str).startswith('D:'):
            self.logger.warning("PDF date string missing or not starting with 'D:': %s", pdf_date_str)
            return None

        original = pdf_date_str

        # Remove PDF "D:" prefix
        pdf_date_str = pdf_date_str[2:]

        # Parse full timestamp + timezone offset
        match = re.match(
            r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})([+-])(\d{2})'?(\d{2})?",
            pdf_date_str
        )
        if not match:
            self.logger.warning("Failed to parse PDF date string: %s", original)
            return None

        year, month, day, hour, minute, second, tz_sign, tz_hour, tz_minute = match.groups()

        # Construct naive datetime from numeric components
        dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))

        # Build timezone offset; missing tz_minute is treated as 0
        offset = timedelta(hours=int(tz_hour or 0), minutes=int(tz_minute or 0))
        if tz_sign == '-':
            offset = -offset

        # Attach timezone information (timezone-aware datetime)
        parsed = dt.replace(tzinfo=timezone(offset))
        self.logger.debug("Parsed PDF date '%s' into datetime '%s'.", original, parsed)
        return parsed
