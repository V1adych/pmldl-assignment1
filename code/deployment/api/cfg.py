import os
import multiprocessing
import onnxruntime as ort
from dataclasses import dataclass
from typing import Tuple


num_gunicorn_workers = int(os.environ.get("GUNICORN_WORKERS", "1"))
ort_opts = ort.SessionOptions()
ort_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_opts.enable_cpu_mem_arena = True
ort_opts.enable_mem_pattern = True
cores = multiprocessing.cpu_count()
ort_opts.intra_op_num_threads = max(1, cores // num_gunicorn_workers)
ort_opts.inter_op_num_threads = 1

ort_providers = ["CPUExecutionProvider"]


@dataclass(frozen=True)
class DetectorConfig:
    spike_detector_onnx: str = "onnx/spike_detector.onnx"
    resize_shape: Tuple[int, int] = (560, 560)
    nms_iou_threshold: float = 0.15
    confidence_threshold: float = 0.3
