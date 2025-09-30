from src.core.embedder import Embedder
from typing import List, Union
from transformers import AutoTokenizer, AutoModel
import torch

class Specter2Embedder(Embedder):
    def __init__(self, model_name: str = "allenai/specter2_base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval() 

    def embed(self, text: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(text, str):
            text = [text]

        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        embeddings = model_output.last_hidden_state[:, 0, :]
        return embeddings
