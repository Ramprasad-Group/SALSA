#!/usr/bin/env python3
"""
Local OCR wrapper using microsoft/trocr-base-printed (no Modal).
trying microsoft/trocr-large-printed

Dependencies:
  pip install transformers pillow torch

API:
  OCRModel().inference(Path("/abs/path/to/image.jpg"))
    -> ("/abs/path/to/image.jpg", "recognized text")
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image, ImageOps
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def _auto_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class OCRModel:
    def __init__(self, model_id: str = "microsoft/trocr-large-printed", device: str | None = None):
        """
        Parameters
        ----------
        model_id : str
            Hugging Face model id.
        device : str | None
            "cuda:0" or "cpu". Defaults to auto-detect.
        """
        self.device = torch.device(device) if device else _auto_device()
        self.processor = TrOCRProcessor.from_pretrained(model_id)
        model = VisionEncoderDecoderModel.from_pretrained(model_id)
        try:
            self.model = model.to(self.device)
        except RuntimeError as e:
            # e.g. CUDA OOM alongside ChartDete/LineFormer sharing the same GPU process.
            print(f"[OCR] WARNING: failed to move model to {self.device} ({e}); falling back to CPU")
            self.device = torch.device("cpu")
            self.model = model.to(self.device)
        self.model.eval()

    def inference(self, path: Path) -> Tuple[str, str]:
        """
        Run OCR on a single image.

        Returns
        -------
        (path_str, text)
        """
        # Open image (respect EXIF orientation)
        img = Image.open(path)
        img = ImageOps.exif_transpose(img).convert("RGB")

        with torch.no_grad():
            pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.to(self.device)
            generated_ids = self.model.generate(pixel_values)
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Return path as string to match downstream JSON usage
        return str(Path(path).resolve()), text