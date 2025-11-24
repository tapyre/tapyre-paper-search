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
            cap = torch.cuda.get_device_capability(0)
            self.logger.info("[CUDA] device=%s cap=sm_%d%d torch_cuda=%s",
                             device_name, cap[0], cap[1], torch.version.cuda)
            torch.backends.cudnn.benchmark = True
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            device_name = "Apple MPS (Metal)"
        else:
            self.device = torch.device("cpu")
            device_name = "CPU"

        self.logger.info("[Specter2Embedder] Using device: %s", device_name)

        try:
            # safetensors = sicher & umgeht torch.load-CVE
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(
                model_name,
                use_safetensors=True,
                trust_remote_code=False,
            )
            self.model.to(self.device)
            self.model.eval()
            self.logger.info("[Specter2Embedder] Model and tokenizer loaded successfully.")
        except Exception as e:
            self.logger.exception("[Specter2Embedder] Failed to load model '%s': %s", model_name, e)
            raise

    @torch.inference_mode()
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
                truncation=False,
                return_tensors="pt",
            )
            encoded_input = {k: v.to(self.device, non_blocking=True) for k, v in encoded_input.items()}

            self.logger.debug(
                "[Specter2Embedder] Tokenized input successfully (seq_len=%d).",
                int(encoded_input["input_ids"].shape[1])
            )

            if self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    model_output = self.model(**encoded_input)
            else:
                model_output = self.model(**encoded_input)


            embeddings = model_output.last_hidden_state[:, 0, :]  # CLS
            self.logger.debug("[Specter2Embedder] Generated embeddings shape: %s", tuple(embeddings.shape))

            return embeddings.detach().to("cpu")

        except Exception as e:
            self.logger.exception("[Specter2Embedder] Error during embedding: %s", e)
            raise
