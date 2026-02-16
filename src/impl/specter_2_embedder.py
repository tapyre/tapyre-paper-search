from src.core.embedder import Embedder
from src.impl.logger import get_logger
from typing import List, Union
from transformers import AutoTokenizer, AutoModel
import torch


class Specter2Embedder(Embedder):
    """
    Embedder implementation using the SPECTER2 model from AllenAI.

    Responsibilities:
    - Load tokenizer + model from HuggingFace
    - Select the best available compute device (CUDA > MPS > CPU)
    - Convert input text (single or batch) into dense vector embeddings
    - Return embeddings on CPU for downstream serialization / storage (e.g., Qdrant)

    Embedding strategy:
    - Tokenize input text(s)
    - Run forward pass through transformer model in inference mode
    - Use the CLS token representation (last_hidden_state[:, 0, :]) as embedding

    Design goals:
    - Stable inference: use eval() + inference_mode()
    - Performance: use CUDA autocast (fp16) when available
    - Observability: log device selection and tensor shapes
    """

    def __init__(self, model_name: str = "allenai/specter2_base"):
        # Logger for traceable runtime diagnostics
        self.logger = get_logger(__name__)
        self.logger.info("[Specter2Embedder] Initializing model: %s", model_name)

        # Determine the best execution device
        # Priority:
        # 1) CUDA (NVIDIA GPU) for maximum throughput
        # 2) Apple MPS (Metal) for Mac acceleration
        # 3) CPU fallback for compatibility
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

            # Log detailed GPU properties for debugging and performance analysis
            device_name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            self.logger.info(
                "[CUDA] device=%s cap=sm_%d%d torch_cuda=%s",
                device_name,
                cap[0],
                cap[1],
                torch.version.cuda
            )

            # Enable cudnn auto-tuner for faster convolutions / kernels when shapes are stable
            torch.backends.cudnn.benchmark = True

        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            device_name = "Apple MPS (Metal)"

        else:
            self.device = torch.device("cpu")
            device_name = "CPU"

        self.logger.info("[Specter2Embedder] Using device: %s", device_name)

        # Load tokenizer and transformer model weights
        # We explicitly disable remote code execution for safety and reproducibility.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            self.model = AutoModel.from_pretrained(
                model_name,
                use_safetensors=True,      # prefer safer tensor format if available
                trust_remote_code=False,   # do not execute custom model code
            )

            # Move model weights to selected device and switch to inference mode
            self.model.to(self.device)
            self.model.eval()

            self.logger.info("[Specter2Embedder] Model and tokenizer loaded successfully.")

        except Exception as e:
            # Model loading failures are fatal because embeddings cannot be produced
            self.logger.exception(
                "[Specter2Embedder] Failed to load model '%s': %s",
                model_name,
                e
            )
            raise

    @torch.inference_mode()
    def embed(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Generate embeddings for a string or a list of strings.

        Parameters:
        - text: either a single string or a list of strings

        Returns:
        - torch.Tensor of shape (batch_size, hidden_dim) on CPU

        Implementation details:
        - Uses tokenizer(...) to build model inputs
        - Moves all tensors to the configured device
        - Runs model forward pass (with autocast on CUDA for speed)
        - Uses CLS token embedding as the final representation
        """
        # Normalize input to always be a list, simplifying downstream processing
        if isinstance(text, str):
            self.logger.debug("[Specter2Embedder] Received single string for embedding.")
            text = [text]
        else:
            self.logger.debug(
                "[Specter2Embedder] Received batch of %d texts for embedding.",
                len(text)
            )

        try:
            # Tokenize input strings into model-ready tensors
            # padding=True pads to the longest sequence in the batch
            # truncation=False means very long inputs may exceed model limits
            # (depending on tokenizer/model config this may raise an error or be slow)
            encoded_input = self.tokenizer(
                text,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )

            # Move all tokenizer outputs to GPU/MPS/CPU
            # non_blocking=True can speed up CPU->GPU transfers for pinned memory
            encoded_input = {k: v.to(self.device, non_blocking=True) for k, v in encoded_input.items()}

            self.logger.debug(
                "[Specter2Embedder] Tokenized input successfully (seq_len=%d).",
                int(encoded_input["input_ids"].shape[1])
            )

            # Forward pass:
            # - On CUDA, use autocast(fp16) to increase throughput and reduce memory usage
            # - On CPU/MPS, run standard precision
            if self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    model_output = self.model(**encoded_input)
            else:
                model_output = self.model(**encoded_input)

            # Use CLS token embedding from the last hidden state
            # Shape: (batch_size, hidden_dim)
            embeddings = model_output.last_hidden_state[:, 0, :]  # CLS

            self.logger.debug(
                "[Specter2Embedder] Generated embeddings shape: %s",
                tuple(embeddings.shape)
            )

            # Detach from graph (safety) and move to CPU for storage/serialization
            return embeddings.detach().to("cpu")

        except Exception as e:
            # Catch-all for tokenizer/model runtime errors
            self.logger.exception("[Specter2Embedder] Error during embedding: %s", e)
            raise
