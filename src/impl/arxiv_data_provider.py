from src.core.data_provider import DataProvider
import re
import requests
import time


class ArxivDataProvider(DataProvider):
    def __init__(self, first_id="", last_id="", rate_limit_seconds=3.0, max_retries: int = 3):
        self.first_id = first_id
        self.last_id = last_id
        self.current_id = first_id
        self.last_pull = 0.0
        self.finished = False
        self.rate_limit_seconds = rate_limit_seconds
        self.max_retries = max_retries

    def hasNext(self):
        print("Current ID:", self.current_id, "Checking hasNext", flush=True)
        return not self.finished

    def next(self):
        print("Current ID:", self.current_id, "Fetching next", flush=True)
        if self.finished:
            # Always return a consistent tuple to avoid unpacking errors in callers
            return None
        while True:
            next_id = self._get_next_id(self.current_id)
            print("Next ID:", next_id, flush=True)
            if next_id == self.last_id or next_id is None:
                self.finished = True
                return None

            self.current_id = next_id
            pdf_data = self._fetch_pdf(next_id)
            if pdf_data and len(pdf_data) > 0:
                break
            print("No valid PDF found, continuing to next ID...", flush=True)
        return next_id, pdf_data

    def _get_next_id(self, current_id):
        # Parse current_id: arXiv:YYMM.number or arXiv:YYMM.numbervV

        match = re.match(r'arXiv:(\d{2})(\d{2})\.(\d{4,5})(?:v\d+)?', current_id)
        if not match:
            return None

        yy, mm, num = match.group(1), match.group(2), match.group(3)
        yy, mm, num = int(yy), int(mm), int(num)

        # Determine padding
        if yy < 15 or (yy == 14 and mm <= 12):
            num_digits = 4
            max_num = 9999
        else:
            num_digits = 5
            max_num = 99999

        # Increment number
        num += 1
        if num > max_num:
            # Move to next month
            num = 1
            mm += 1
            if mm > 12:
                mm = 1
                yy += 1
            # Check for valid year/month range
            if yy > 99 or (yy == 7 and mm < 4):  # 07=2007, 0704 is first valid
                return None

            # Update padding for new month/year
            if yy < 15 or (yy == 14 and mm <= 12):
                num_digits = 4
            else:
                num_digits = 5

        # Format next_id
        next_id = f'arXiv:{yy:02d}{mm:02d}.{num:0{num_digits}d}'
        return next_id

    def _fetch_pdf(self, paper_id):
        attempts = 0
        while attempts < self.max_retries:
            now = time.time()
            wait_time = self.last_pull + self.rate_limit_seconds - now
            if wait_time > 0:
                print(f"Rate limiting: sleeping for {wait_time:.2f} seconds", flush=True)
                time.sleep(wait_time)

            self.last_pull = time.time()

            match = re.match(r'arXiv:(\d{4,6}\.\d{4,5})', paper_id)
            if not match:
                return b""

            arxiv_id = match.group(1)
            url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    return response.content
                elif response.status_code == 404:
                    print(f"PDF not found (404) for ID {arxiv_id}", flush=True)
                    return b""
                else:
                    attempts += 1
                    backoff = min(60, 5 * attempts)
                    print(f"Unexpected status code {response.status_code}, retrying in {backoff}s (attempt {attempts}/{self.max_retries})...", flush=True)
                    time.sleep(backoff)
            except requests.exceptions.RequestException as e:
                attempts += 1
                backoff = min(60, 5 * attempts)
                print(f"Request exception during PDF fetch: {e}, retrying in {backoff}s (attempt {attempts}/{self.max_retries})...", flush=True)
                time.sleep(backoff)

        print(f"Max retries reached for {paper_id}, giving up and returning empty result.", flush=True)
        return b""

if __name__ == "__main__":
    first_id = "arXiv:2301.99998"
    last_id = "arXiv:2302.00010"
    provider = ArxivDataProvider(first_id=first_id, last_id=last_id, rate_limit_seconds=0.5)

    while provider.hasNext():
        result = provider.next()
        if result is None:
            break
        paper_id, pdf_data = result
        print(f"Fetched {paper_id}: {len(pdf_data)} bytes")