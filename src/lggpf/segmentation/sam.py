"""
Segment Anything Model (SAM) wrapper for instance segmentation.

Uses Meta's SAM (vit_h variant) to produce binary masks from bounding box prompts.
"""

import numpy as np
from segment_anything import SamPredictor, sam_model_registry


class SegmentAnythingModel:
    """SAM-based instance segmentation with bounding box prompts.

    Args:
        path: Path to the SAM checkpoint file (e.g. ``sam_vit_h_4b8939.pth``).
        model_type: SAM model variant. Default ``"vit_h"``.
        device: Torch device string. Default ``"cuda"``.
    """

    def __init__(self, path: str, model_type: str = "vit_h", device: str = "cuda"):
        self.sam = sam_model_registry[model_type](checkpoint=path)
        self.sam = self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)

    def segment(self, image: np.ndarray, box) -> np.ndarray:
        """Segment an object within a bounding box.

        Args:
            image: RGB image as (H, W, 3) numpy array.
            box: Bounding box as [x1, y1, x2, y2] (numpy array or list).

        Returns:
            Binary mask as (H, W) boolean numpy array.
        """
        box = np.asarray(box, dtype=np.float32)
        self.predictor.set_image(image)
        masks, _, _ = self.predictor.predict(
            box=box[None, :],  # Add batch dimension
            multimask_output=False,
        )
        return masks[0]
