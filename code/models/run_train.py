from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import json
from pathlib import Path

import tyro

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image

import pytorch_lightning as pl

import rfdetr

from utils import DETRCriterion, HungarianMatcher


CLASS_NAME_TO_ID: Dict[str, int] = {
    "normal": 0,
    "renewed": 0,
    "broken": 1,
    "floating": 1,
    "absent": 2,
}


def get_detr(num_classes: int, group_detr: int, resolution: int):
    cfg = rfdetr.config.RFDETRBaseConfig()
    cfg.group_detr = group_detr
    cfg.num_select = 300
    cfg.resolution = resolution
    model = rfdetr.detr.RFDETR(**cfg.__dict__)
    model.model.reinitialize_detection_head(num_classes)
    lw_model = model.model.model
    return lw_model


class OBDataset(Dataset):
    def __init__(
        self,
        roots: List[str],
        size: int = 560,
        do_augmentations: bool = True,
        hflip_prob: float = 0.5,
        normalize: bool = True,
    ):
        super().__init__()
        self.roots = [Path(r) for r in roots]
        self.size = size
        self.do_augmentations = do_augmentations
        self.hflip_prob = hflip_prob
        self.normalize = normalize

        means = rfdetr.detr.RFDETR.means
        stds = rfdetr.detr.RFDETR.stds

        base_transforms = [
            transforms.Resize(
                (self.size, self.size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
        ]
        if self.normalize:
            base_transforms.append(transforms.Normalize(mean=means, std=stds))
        self.img_transform = transforms.Compose(base_transforms)

        self.samples: List[Tuple[str, str]] = []
        for root in self.roots:
            img_dir = root / "images"
            annot_dir = root / "annot"
            if not img_dir.exists() or not annot_dir.exists():
                continue
            for img_path in img_dir.iterdir():
                if not img_path.is_file():
                    continue
                stem = img_path.stem
                ann_path = annot_dir / f"{stem}.json"
                if ann_path.exists():
                    self.samples.append((str(img_path), str(ann_path)))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_targets(self, ann_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        with open(ann_path, "r") as f:
            boxes_list = json.load(f)

        cxcywh, labels = [], []
        for b in boxes_list:
            label_name = b.get("label")
            if label_name not in CLASS_NAME_TO_ID:
                continue
            label_id = CLASS_NAME_TO_ID[label_name]
            cxcywh.append([float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])])
            labels.append(label_id)

        if len(cxcywh) == 0:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(
                (0,), dtype=torch.long
            )

        return torch.tensor(cxcywh, dtype=torch.float32), torch.tensor(
            labels, dtype=torch.long
        )

    def __getitem__(self, idx: int):
        img_path, ann_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        boxes, labels = self._load_targets(ann_path)

        if self.do_augmentations and torch.rand(1).item() < self.hflip_prob:
            img = TF.hflip(img)
            if boxes.numel() > 0:
                boxes = boxes.clone()
                boxes[:, 0] = 1.0 - boxes[:, 0]

        img = self.img_transform(img)

        target = {"boxes": boxes, "labels": labels}
        return img, target


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return images, targets


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(min=0)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter + 1e-7
    return inter / union


@dataclass
class Args:
    train_roots: List[str]
    val_roots: List[str]
    resume_training_checkpoint: Optional[str] = None
    num_classes: int = 3
    ckpt_dir: str = "det_checkpoints"
    size: int = 560
    num_epochs: int = 100
    batch_size: int = 8
    num_workers: int = 4
    seed: int = 42
    eps: float = 1e-4
    lr: float = 1e-4
    class_cost: float = 1.0
    bbox_cost: float = 5.0
    giou_cost: float = 2.0
    eos_coef: float = 1.0
    group_detr: int = 3
    do_augmentations: bool = False
    hflip_prob: float = 0.5
    normalize: bool = False
    prefetch_factor: int = 2


class DETRLightning(pl.LightningModule):
    def __init__(self, args: "Args"):
        super().__init__()
        self.save_hyperparameters(vars(args))
        self.args = args
        self.model = get_detr(
            num_classes=args.num_classes + 1,
            group_detr=args.group_detr,
            resolution=args.size,
        )
        self.matcher = HungarianMatcher(
            class_cost=args.class_cost,
            bbox_cost=args.bbox_cost,
            giou_cost=args.giou_cost,
            num_classes=args.num_classes,
        )
        self.criterion = DETRCriterion(
            num_classes=args.num_classes, matcher=self.matcher, eos_coef=args.eos_coef
        )

    def forward(self, images: torch.Tensor):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        losses = self.criterion(outputs, targets)
        loss = sum(losses.values())
        self.log(
            "train_labels_loss",
            losses["loss_ce"],
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train_boxes_loss",
            losses["loss_bbox"],
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train_giou_loss",
            losses["loss_giou"],
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train_cardinality_loss",
            losses["cardinality_error"],
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        losses = self.criterion(outputs, targets)
        loss = sum(losses.values())
        self.log(
            "val_labels_loss",
            losses["loss_ce"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_boxes_loss",
            losses["loss_bbox"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_giou_loss",
            losses["loss_giou"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_cardinality_loss",
            losses["cardinality_error"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.args.lr, weight_decay=1e-4
        )
        return optimizer


def main():
    args = tyro.cli(Args)
    pl.seed_everything(args.seed)

    train_ds = OBDataset(
        roots=args.train_roots,
        size=args.size,
        do_augmentations=args.do_augmentations,
        hflip_prob=args.hflip_prob,
        normalize=args.normalize,
    )
    val_ds = OBDataset(
        roots=args.val_roots,
        size=args.size,
        do_augmentations=False,
        normalize=args.normalize,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=args.prefetch_factor,
    )

    lit_model = DETRLightning(args)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=str(Path(args.ckpt_dir) / "checkpoints"),
        filename="model-{epoch:03d}-{val_iou:.5f}",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    print("Starting training...")
    trainer.fit(
        lit_model, train_loader, val_loader, ckpt_path=args.resume_training_checkpoint
    )


if __name__ == "__main__":
    main()
