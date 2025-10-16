from src.core.embedder import Embedder
from src.impl.logger import get_logger
from typing import List, Union
from transformers import AutoTokenizer, AutoModel
import torch


class Specter2Embedder(Embedder):
    def __init__(self, model_name: str = "allenai/specter2_base"):
        self.logger = get_logger(__name__)
        self.logger.info("[Specter2Embedder] Initializing model: %s", model_name)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(0)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            device_name = "Apple MPS (Metal)"
        else:
            self.device = torch.device("cpu")
            device_name = "CPU"

        self.logger.info("[Specter2Embedder] Using device: %s", device_name)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            self.logger.info("[Specter2Embedder] Model and tokenizer loaded successfully.")
        except Exception as e:
            self.logger.exception("[Specter2Embedder] Failed to load model '%s': %s", model_name, e)
            raise

    def embed(self, text: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(text, str):
            self.logger.debug("[Specter2Embedder] Received single string for embedding.")
            text = [text]
        else:
            self.logger.debug("[Specter2Embedder] Received batch of %d texts for embedding.", len(text))

        try:
            encoded_input = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            self.logger.debug("[Specter2Embedder] Tokenized input successfully (tokens per text: %d).",
                              encoded_input['input_ids'].shape[1])

            with torch.no_grad():
                model_output = self.model(**encoded_input)

            embeddings = model_output.last_hidden_state[:, 0, :]
            self.logger.debug("[Specter2Embedder] Generated embeddings with shape: %s", tuple(embeddings.shape))

            return embeddings
        except Exception as e:
            self.logger.exception("[Specter2Embedder] Error during embedding: %s", e)
            raise
