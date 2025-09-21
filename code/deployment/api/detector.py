from typing import List, Dict, Any, Tuple

import numpy as np
import cv2
import onnxruntime as ort

from cfg import DetectorConfig, ort_providers, ort_opts


def cxcywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def xyxy2cxcywh(boxes: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = boxes.T
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return np.stack([cx, cy, w, h], axis=1)


def nms(
    boxes: np.ndarray,
    logits: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.int32)

    boxes = boxes.astype(np.float32)
    scores = (1.0 / (1.0 + np.exp(-logits))).max(axis=1)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    w = np.maximum(0.0, x2 - x1)
    h = np.maximum(0.0, y2 - y1)
    areas = w * h

    order = np.argsort(scores)[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        iw = np.maximum(0.0, xx2 - xx1)
        ih = np.maximum(0.0, yy2 - yy1)
        inter = iw * ih

        union = areas[i] + areas[rest] - inter
        iou = inter / np.maximum(union, 1e-7)

        inds = np.where(iou <= iou_threshold)[0]
        order = rest[inds]

    return np.array(keep, dtype=np.int32), scores


class Detector:
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.det_session = ort.InferenceSession(
            self.config.spike_detector_onnx,
            providers=ort_providers,
            sess_options=ort_opts,
        )

    def _nms_filter(
        self, boxes: np.ndarray, logits: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        keep_ids, scores = nms(boxes, logits, self.config.nms_iou_threshold)

        return boxes[keep_ids], logits[keep_ids], scores[keep_ids]

    def _confidence_filter(
        self, boxes: np.ndarray, logits: np.ndarray, scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        keep = scores > self.config.confidence_threshold

        return boxes[keep], logits[keep], scores[keep]

    def __call__(self, image: np.ndarray) -> List[Dict[str, Any]]:
        h, w, _ = image.shape

        image = cv2.resize(
            image, self.config.resize_shape, interpolation=cv2.INTER_LINEAR
        )
        image = image.transpose(2, 0, 1)[None].astype(np.float32) / 255

        boxes_cxcywh, logits = self.det_session.run(None, {"input": image})
        boxes_cxcywh = boxes_cxcywh.squeeze(0)
        logits = logits.squeeze(0)
        boxes_xyxy = cxcywh2xyxy(boxes_cxcywh)

        boxes_xyxy, logits, scores = self._nms_filter(boxes_xyxy, logits)
        boxes_xyxy, logits, scores = self._confidence_filter(boxes_xyxy, logits, scores)
        labels = np.argmax(logits, axis=1)

        boxes_cxcywh = xyxy2cxcywh(boxes_xyxy)
        boxes_cxcywh = boxes_cxcywh * np.array([w, h, w, h])
        boxes_cxcywh = boxes_cxcywh.astype(np.int32)
        result = [
            {
                "box": (
                    int(boxes_cxcywh[i][0]),
                    int(boxes_cxcywh[i][1]),
                    int(boxes_cxcywh[i][2]),
                    int(boxes_cxcywh[i][3]),
                ),
                "class": int(labels[i]) - 1,
            }
            for i in range(len(boxes_cxcywh))
        ]

        return result
