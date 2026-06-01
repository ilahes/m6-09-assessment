from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image


class CatDetector:
    CLASS_NAMES = ("cat",)

    def __init__(
        self,
        model_path: str = "/app/models/best.onnx",
        imgsz: int = 640,
        conf_threshold: float = 0.25,
    ):
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self.session.get_inputs()[0].name

    # ------------------------------------------------------------------
    def _letterbox(self, img: Image.Image):
        """Resize + pad to (imgsz, imgsz) preserving aspect ratio."""
        w, h = img.size
        scale = min(self.imgsz / w, self.imgsz / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (self.imgsz, self.imgsz), (114, 114, 114))
        pad_x = (self.imgsz - nw) // 2
        pad_y = (self.imgsz - nh) // 2
        canvas.paste(resized, (pad_x, pad_y))
        return np.array(canvas, dtype=np.uint8), scale, (pad_x, pad_y)

    # ------------------------------------------------------------------
    def predict(self, image_path: str) -> list[dict]:
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size

        arr, scale, (pad_x, pad_y) = self._letterbox(img)
        x = (arr.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

        raw = self.session.run(None, {self._input_name: x})[0]

        # Handle (1, 300, 6) or (300, 6)
        if raw.ndim == 3:
            raw = raw[0]

        results = []
        for row in raw:
            x1, y1, x2, y2, score, cls = row
            if float(score) < self.conf_threshold:
                continue
            # Undo letterbox
            rx1 = (float(x1) - pad_x) / scale
            ry1 = (float(y1) - pad_y) / scale
            rx2 = (float(x2) - pad_x) / scale
            ry2 = (float(y2) - pad_y) / scale
            # Clip to image bounds
            rx1 = max(0.0, min(orig_w, rx1))
            ry1 = max(0.0, min(orig_h, ry1))
            rx2 = max(0.0, min(orig_w, rx2))
            ry2 = max(0.0, min(orig_h, ry2))
            # Skip degenerate boxes
            if rx2 <= rx1 or ry2 <= ry1:
                continue
            results.append({
                "xmin": rx1, "ymin": ry1, "xmax": rx2, "ymax": ry2,
                "confidence": float(score),
                "class": self.CLASS_NAMES[int(cls)] if int(cls) < len(self.CLASS_NAMES) else "cat",
            })
        return results
