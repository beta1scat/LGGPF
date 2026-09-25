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

    def __init__(self, path: str, device: str | None = None):
        self.processor = Owlv2Processor.from_pretrained(path)
        self.model = Owlv2ForObjectDetection.from_pretrained(path)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()

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
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)

        if isinstance(image, np.ndarray):
            target_sizes = torch.Tensor([image.shape[:2]]).to(self.device)
        else:
            target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)

        if hasattr(self.processor, "post_process_object_detection"):
            results = self.processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=threshold
            )
        elif hasattr(self.processor, "post_process_grounded_object_detection"):
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=threshold
            )
        elif hasattr(getattr(self.processor, "image_processor", None), "post_process_object_detection"):
            results = self.processor.image_processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=threshold
            )
        else:
            raise AttributeError("Owlv2Processor has no post_process method for object detection.")

        boxes = results[0]["boxes"].to("cpu")
        scores = results[0]["scores"].to("cpu")
        return boxes, scores
