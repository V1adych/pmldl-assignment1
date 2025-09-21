import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_area(xyxy):
    x0, y0, x1, y1 = xyxy.unbind(-1)
    return (x1 - x0).clamp(min=0) * (y1 - y0).clamp(min=0)


def generalized_box_iou(boxes1, boxes2):
    inter_x0 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y0 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x1 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y1 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = (inter_x1 - inter_x0).clamp(0) * (inter_y1 - inter_y0).clamp(0)

    area1 = box_area(boxes1)[:, None]
    area2 = box_area(boxes2)[None, :]
    union = area1 + area2 - inter + 1e-7
    iou = inter / union

    c_x0 = torch.min(boxes1[:, None, 0], boxes2[None, :, 0])
    c_y0 = torch.min(boxes1[:, None, 1], boxes2[None, :, 1])
    c_x1 = torch.max(boxes1[:, None, 2], boxes2[None, :, 2])
    c_y1 = torch.max(boxes1[:, None, 3], boxes2[None, :, 3])
    c_area = (c_x1 - c_x0).clamp(min=0) * (c_y1 - c_y0).clamp(min=0) + 1e-7

    giou = iou - (c_area - union) / c_area
    return giou


class HungarianMatcher(nn.Module):
    def __init__(self, class_cost=1.0, bbox_cost=5.0, giou_cost=2.0, num_classes=None):
        super().__init__()
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        self.num_classes = num_classes

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries, num_classes_p1 = outputs["pred_logits"].shape
        if self.num_classes is not None:
            assert num_classes_p1 == self.num_classes + 1

        out_prob = outputs["pred_logits"].softmax(-1)
        out_bbox = outputs["pred_boxes"]

        indices = []
        for b in range(bs):
            tgt_ids = targets[b]["labels"]
            tgt_bbox = targets[b]["boxes"]

            if tgt_ids.numel() == 0:
                indices.append(
                    (
                        torch.as_tensor([], dtype=torch.int64, device=out_prob.device),
                        torch.as_tensor([], dtype=torch.int64, device=out_prob.device),
                    )
                )
                continue

            cost_class = -out_prob[b][:, tgt_ids]

            cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)

            cost_giou = 1.0 - generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox[b]), box_cxcywh_to_xyxy(tgt_bbox)
            )

            C = (
                self.class_cost * cost_class
                + self.bbox_cost * cost_bbox
                + self.giou_cost * cost_giou
            )
            C = C.cpu()

            q_ind, t_ind = linear_sum_assignment(C)
            indices.append(
                (
                    torch.as_tensor(q_ind, dtype=torch.int64, device=out_prob.device),
                    torch.as_tensor(t_ind, dtype=torch.int64, device=out_prob.device),
                )
            )
        return indices


class DETRCriterion(nn.Module):
    def __init__(self, num_classes, matcher, eos_coef=0.1):
        """
        num_classes: number of foreground classes (no-object is added internally)
        matcher: a module returning list[(src_idx, tgt_idx)] per batch
        eos_coef: weight for the no-object class in CE
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs, targets, indices):
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)], dim=0
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            weight=self.empty_weight,
        )
        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none").sum() / max(
            target_boxes.shape[0], 1
        )

        giou = generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
        )
        loss_giou = (1.0 - torch.diag(giou)).sum() / max(target_boxes.shape[0], 1)
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices):
        pred_logits = outputs["pred_logits"]
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=pred_logits.device
        )
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        return {"cardinality_error": card_err}

    def forward(self, outputs, targets):
        """
        outputs: {'pred_logits': [B,Q,C+1], 'pred_boxes': [B,Q,4], optional 'aux_outputs': list of dicts}
        targets: list of len B, each {'labels': [N_i], 'boxes': [N_i,4] in cxcywh normalized}
        """
        outputs_no_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(outputs_no_aux, targets)

        losses = {}
        losses.update(self.loss_labels(outputs_no_aux, targets, indices))
        losses.update(self.loss_boxes(outputs_no_aux, targets, indices))
        losses.update(self.loss_cardinality(outputs_no_aux, targets, indices))

        if "aux_outputs" in outputs:
            for i, aux in enumerate(outputs["aux_outputs"]):
                idx_losses = {}
                idx_losses.update(
                    self.loss_labels(aux, targets, self.matcher(aux, targets))
                )
                idx_losses.update(
                    self.loss_boxes(aux, targets, self.matcher(aux, targets))
                )
                for k, v in idx_losses.items():
                    losses[f"{k}_{i}"] = v
        return losses
