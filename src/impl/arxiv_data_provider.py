import os
import re
import time
import json
import requests
from pathlib import Path

from src.core.data_provider import DataProvider
from src.impl.logger import get_logger


class ArxivDataProvider(DataProvider):
    """
    Concrete implementation of a DataProvider that sequentially fetches PDFs
    from the arXiv repository using predictable ID iteration.

    Design goals:
    - Deterministic sequential traversal of ID ranges
    - Crash-safe execution via persistent state
    - Robust long-running operation with retries and rate limiting
    - Production observability via structured logging

    This provider is intended for large-scale ingestion pipelines where
    reliability and resumability are critical.
    """

    def __init__(
        self,
        first_id: str = "",
        last_id: str = "",
        rate_limit_seconds: float = 3.0,
        max_retries: int = 3,
    ):
        # Initialize logger for structured runtime diagnostics
        self.logger = get_logger(__name__)
        self.logger.info("Initializing ArxivDataProvider")

        # Boundaries of the ID range to process
        self.first_id = first_id
        self.last_id = last_id

        # Current progress pointer
        self.current_id = first_id

        # Timestamp of last HTTP request (used for rate limiting)
        self.last_pull = 0.0

        # Indicates whether iteration has completed
        self.finished = False

        # Runtime parameters controlling stability and politeness
        self.rate_limit_seconds = rate_limit_seconds
        self.max_retries = max_retries

        # Determine location of persistent state file
        # Allows override via environment variable for container deployments
        state_path = os.getenv("ARXIV_STATE_FILE", "/app/state/arxiv_state.json")
        self.state_file = Path(state_path)

        # Ensure state directory exists before reading/writing
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # Attempt to restore previous processing state
        self._load_state()

    def _load_state(self) -> None:
        """
        Restore persisted provider state from disk.

        This allows interrupted runs (crashes, restarts, deployments)
        to continue exactly where they left off.
        """
        if not self.state_file.exists():
            self.logger.info("[state] No state file found, starting from first_id.")
            return

        try:
            # Load serialized state
            with self.state_file.open("r", encoding="utf-8") as f:
                state = json.load(f)

            # Extract required fields only
            saved_current = state.get("current_id")
            saved_finished = state.get("finished", False)

            # Restore progress if valid
            if saved_current:
                self.logger.info(
                    f"[state] Restoring state from {self.state_file}: "
                    f"current_id={saved_current}, finished={saved_finished}"
                )
                self.current_id = saved_current
                self.finished = saved_finished

        except Exception as e:
            # Loading errors must never crash the ingestion pipeline
            self.logger.error(f"[state] Failed to load state file: {e}", exc_info=True)

    def _save_state(self) -> None:
        """
        Persist current processing state atomically.

        Atomic write pattern:
        1) write temporary file
        2) replace original file

        This prevents corruption if process crashes mid-write.
        """
        tmp_file = self.state_file.with_suffix(".tmp")

        # Minimal set of fields required to fully resume execution
        data = {
            "current_id": self.current_id,
            "first_id": self.first_id,
            "last_id": self.last_id,
            "finished": self.finished,
            "last_pull": self.last_pull,
        }

        try:
            # Write temporary file first
            with tmp_file.open("w", encoding="utf-8") as f:
                json.dump(data, f)

            # Atomic replacement of previous state file
            os.replace(tmp_file, self.state_file)

            self.logger.debug(
                f"[state] Saved state: current_id={self.current_id}, finished={self.finished}"
            )

        except Exception as e:
            self.logger.error(f"[state] Failed to save state: {e}", exc_info=True)

            # Cleanup temporary file if replacement failed
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except OSError:
                    pass

    def hasNext(self) -> bool:
        """
        Return True if additional documents remain to be processed.

        This method is used by orchestration logic to determine whether
        iteration should continue.
        """
        self.logger.debug(f"[hasNext] Current ID: {self.current_id}, finished={self.finished}")
        return not self.finished

    def next(self):
        """
        Fetch the next available PDF from arXiv.

        Execution flow:
        1) Compute next candidate ID
        2) Attempt download
        3) Persist progress if successful
        4) Skip missing documents
        5) Stop when range exhausted
        """
        if self.finished:
            self.logger.info("[next] No more IDs to process — finished.")
            return None

        while True:
            # Determine next sequential ID
            next_id = self._get_next_id(self.current_id)
            self.logger.debug(f"[next] Next ID candidate: {next_id}")

            # Stop conditions
            if next_id == self.last_id or next_id is None:
                self.logger.info("[next] Reached last ID or invalid next ID — marking finished.")
                self.finished = True
                self._save_state()
                return None

            self.current_id = next_id

            # Attempt download
            pdf_data = self._fetch_pdf(next_id)

            # Valid PDF received
            if pdf_data and len(pdf_data) > 0:
                self.logger.info(
                    f"[next] Successfully fetched PDF for {next_id} ({len(pdf_data)} bytes)"
                )
                self._save_state()
                break

            # Skip missing or invalid documents
            self.logger.warning(f"[next] No valid PDF found for {next_id}, continuing...")

        return next_id, pdf_data

    def _get_next_id(self, current_id: str) -> str | None:
        """
        Compute next sequential arXiv identifier.

        Format handled:
        arXiv:YYMM.NNNNN

        Logic:
        - Increment numeric part
        - Handle rollover after 10000
        - Handle month/year overflow
        - Reject invalid historical ranges
        """
        match = re.match(r'arXiv:(\d{2})(\d{2})\.(\d{4,5})(?:v\d+)?', current_id)
        if not match:
            self.logger.warning(f"[get_next_id] Invalid current ID format: {current_id}")
            return None

        yy, mm, num = map(int, match.groups())

        ROLLOVER_LIMIT = 10000
        num += 1

        # Handle numeric rollover
        if num >= ROLLOVER_LIMIT:
            num = 0
            mm += 1

            # Month rollover
            if mm > 12:
                mm = 1
                yy += 1

            # Safety bounds to avoid invalid IDs
            if yy > 99 or (yy == 7 and mm < 4):
                return None

        next_id = f'arXiv:{yy:02d}{mm:02d}.{num:05d}'
        return next_id

    def _fetch_pdf(self, paper_id: str) -> bytes:
        """
        Download PDF from arXiv with retry and throttling logic.

        Reliability features:
        - Rate limiting between requests
        - Retry strategy for transient failures
        - Immediate abort for 404 responses
        - Exponential-style backoff
        """
        attempts = 0

        while attempts < self.max_retries:
            # Respect rate limit
            now = time.time()
            wait_time = self.last_pull + self.rate_limit_seconds - now

            if wait_time > 0:
                self.logger.debug(f"[fetch_pdf] Rate limiting: sleeping {wait_time:.2f}s")
                time.sleep(wait_time)

            self.last_pull = time.time()

            # Extract numeric ID required for PDF URL
            match = re.match(r'arXiv:(\d{4,6}\.\d{4,5})', paper_id)
            if not match:
                self.logger.error(f"[fetch_pdf] Invalid arXiv ID format: {paper_id}")
                return b""

            arxiv_id = match.group(1)
            url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            self.logger.info(
                f"[fetch_pdf] Fetching PDF from {url} (attempt {attempts+1}/{self.max_retries})"
            )

            try:
                response = requests.get(url, timeout=10)
                status = response.status_code

                # Successful response
                if status == 200:
                    self.logger.debug(
                        f"[fetch_pdf] 200 OK for {arxiv_id} ({len(response.content)} bytes)"
                    )
                    return response.content

                # File does not exist
                elif status == 404:
                    self.logger.warning(f"[fetch_pdf] 404 Not Found for {arxiv_id}")
                    return b""

                # Unexpected HTTP response
                else:
                    attempts += 1
                    backoff = 5 * 60 * 60

                    self.logger.critical(
                        f"[fetch_pdf] CRITICAL: Unexpected HTTP {status} for {arxiv_id}, "
                        f"retrying in {backoff}s (attempt {attempts}/{self.max_retries})"
                    )

                    time.sleep(backoff)

            except requests.exceptions.RequestException as e:
                # Network-level failure → retry with increasing delay
                attempts += 1
                backoff = min(600, 100 * attempts)

                self.logger.error(
                    f"[fetch_pdf] Request exception for {arxiv_id}: {e}, "
                    f"retrying in {backoff}s (attempt {attempts}/{self.max_retries})",
                    exc_info=True
                )

                time.sleep(backoff)

        # Retry limit exceeded
        self.logger.critical(f"[fetch_pdf] Max retries reached for {paper_id}, giving up.")
        return b""
