from src.core.data_provider import DataProvider
from src.impl.logger import get_logger
import re
import requests
import time


class ArxivDataProvider(DataProvider):
    def __init__(self, first_id="", last_id="", rate_limit_seconds=6.0, max_retries: int = 3):
        self.logger = get_logger(__name__)
        self.logger.info("Initializing ArxivDataProvider")

        self.first_id = first_id
        self.last_id = last_id
        self.current_id = first_id
        self.last_pull = 0.0
        self.finished = False
        self.rate_limit_seconds = rate_limit_seconds
        self.max_retries = max_retries

    def hasNext(self):
        self.logger.debug(f"[hasNext] Current ID: {self.current_id}, finished={self.finished}")
        return not self.finished

    def next(self):
        """Fetch the next PDF from arXiv sequentially."""
        if self.finished:
            self.logger.info("[next] No more IDs to process — finished.")
            return None

        while True:
            next_id = self._get_next_id(self.current_id)
            self.logger.debug(f"[next] Next ID candidate: {next_id}")

            if next_id == self.last_id or next_id is None:
                self.logger.info("[next] Reached last ID or invalid next ID — marking finished.")
                self.finished = True
                return None

            self.current_id = next_id
            pdf_data = self._fetch_pdf(next_id)

            if pdf_data and len(pdf_data) > 0:
                self.logger.info(f"[next] Successfully fetched PDF for {next_id} ({len(pdf_data)} bytes)")
                break

            self.logger.warning(f"[next] No valid PDF found for {next_id}, continuing...")
        return next_id, pdf_data

    def _get_next_id(self, current_id):
        """Increment arXiv ID in YYMM.NNNNN format."""
        match = re.match(r'arXiv:(\d{2})(\d{2})\.(\d{4,5})(?:v\d+)?', current_id)
        if not match:
            self.logger.warning(f"[get_next_id] Invalid current ID format: {current_id}")
            return None

        yy, mm, num = map(int, match.groups())
        num_digits = 4 if yy < 15 or (yy == 14 and mm <= 12) else 5
        max_num = 9999 if num_digits == 4 else 99999

        num += 1
        if num > max_num:
            num = 1
            mm += 1
            if mm > 12:
                mm = 1
                yy += 1
            if yy > 99 or (yy == 7 and mm < 4):
                return None
            num_digits = 4 if yy < 15 or (yy == 14 and mm <= 12) else 5

        next_id = f'arXiv:{yy:02d}{mm:02d}.{num:0{num_digits}d}'
        return next_id

    def _fetch_pdf(self, paper_id):
        """Download PDF from arXiv with retries and rate limiting."""
        attempts = 0
        while attempts < self.max_retries:
            now = time.time()
            wait_time = self.last_pull + self.rate_limit_seconds - now
            if wait_time > 0:
                self.logger.debug(f"[fetch_pdf] Rate limiting: sleeping {wait_time:.2f}s")
                time.sleep(wait_time)

            self.last_pull = time.time()

            match = re.match(r'arXiv:(\d{4,6}\.\d{4,5})', paper_id)
            if not match:
                self.logger.error(f"[fetch_pdf] Invalid arXiv ID format: {paper_id}")
                return b""

            arxiv_id = match.group(1)
            url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            self.logger.info(f"[fetch_pdf] Fetching PDF from {url} (attempt {attempts+1}/{self.max_retries})")

            try:
                response = requests.get(url, timeout=10)
                status = response.status_code

                if status == 200:
                    self.logger.debug(f"[fetch_pdf] 200 OK for {arxiv_id} ({len(response.content)} bytes)")
                    return response.content

                elif status == 404:
                    self.logger.warning(f"[fetch_pdf] 404 Not Found for {arxiv_id}")
                    return b""

                else:
                    attempts += 1
                    backoff = min(600, 100 * attempts)
                    self.logger.critical(
                        f"[fetch_pdf] CRITICAL: Unexpected HTTP {status} for {arxiv_id}, "
                        f"retrying in {backoff}s (attempt {attempts}/{self.max_retries})"
                    )
                    time.sleep(backoff)

            except requests.exceptions.RequestException as e:
                attempts += 1
                backoff = min(60, 5 * attempts)
                self.logger.error(
                    f"[fetch_pdf] Request exception for {arxiv_id}: {e}, "
                    f"retrying in {backoff}s (attempt {attempts}/{self.max_retries})",
                    exc_info=True
                )
                time.sleep(backoff)

        self.logger.critical(f"[fetch_pdf] Max retries reached for {paper_id}, giving up.")
        return b""
