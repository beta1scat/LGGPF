"""
OWLv2 vision-language object detection.

Uses the OWLv2 (Open-World Localization v2) model from HuggingFace
Transformers to detect objects given a text description.
"""

import numpy as np
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection


class VisionLanguageOwlVit:
    """OWLv2-based open-vocabulary object detector.

    Args:
        path: Local path or HuggingFace model ID for the OWLv2 model.
    """

    def __init__(self, path: str):
        self.processor = Owlv2Processor.from_pretrained(path)
        self.model = Owlv2ForObjectDetection.from_pretrained(path)

    def get_boxes_by_text(self, image, text: str, threshold: float = 0.1):
        """Detect objects matching a text description.

        Args:
            image: Input image as a PIL Image or numpy array (H, W, C).
            text: Text description of the object to detect.
            threshold: Confidence threshold for detections.

        Returns:
            Tuple of (boxes, scores) where boxes is a tensor of shape (N, 4)
            in xyxy format and scores is a tensor of shape (N,).
        """
        inputs = self.processor(text=[[text]], images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        if isinstance(image, np.ndarray):
            target_sizes = torch.Tensor([image.shape[:2]])
        else:
            target_sizes = torch.Tensor([image.size[::-1]])

        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=threshold
        )
        boxes = results[0]["boxes"]
        scores = results[0]["scores"]
        return boxes, scores
