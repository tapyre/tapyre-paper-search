from src.core.data_provider import DataProvider
import os

class SimpleDataProvider(DataProvider):
    def __init__(self, path="data"):
        self.path = path
        self.data = []
        for filename in os.listdir(self.path):
            if filename.lower().endswith('.pdf'):
                full_path = os.path.join(self.path, filename)
                with open(full_path, "rb") as f:
                    pdf_data = f.read()
                self.data.append((filename.lower()[:-4], pdf_data))
                
    def next(self):
        if self.data:
            return self.data.pop(0)
        return None

    def hasNext(self)-> bool:
        return len(self.data) > 0