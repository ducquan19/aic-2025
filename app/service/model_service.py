import torch
import numpy as np


class ModelService:
    def __init__(self, model, preprocess, tokenizer, device: str = "cuda"):
        self.model = model
        self.model = model.to(device)
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def embedding(self, query_text: str) -> np.ndarray:
        """
        Return (1, ndim 1024) torch.Tensor
        """
        with torch.no_grad():
            text_tokens = self.tokenizer([query_text]).to(self.device)
            feats = self.model.encode_text(text_tokens)  # [1, D]
            # L2-normalize (phần 3 bên dưới)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            arr = feats.cpu().numpy().astype(np.float32)
            # DEBUG: in ra kích thước một lần
            # print("TEXT EMBEDDING SHAPE:", arr.shape)  # kỳ vọng (1, 512) với ViT-B-32
            return arr
