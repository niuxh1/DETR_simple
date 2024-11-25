import torch
from torch.ao.quantization.backend_config.executorch import executorch_weight_qint8_neg_127_to_127_scale_min_2_neg_12
from torchvision.ops.boxes import box_area


def box_to_xy(x):
    x, y, w, h = x.unbind(-1)

    box = torch.stack([(x - 0.5 * w), (y - 0.5 * h), (x + 0.5 * w), (y + 0.5 * h)], dim=-1).type_as(x)
    return box


def box_to_cxcy(x):
    x0, y0, x1, y1 = x.unbind(-1)

    box = torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1).type_as(x)
    return box


def box_iou(box1, box2):
    box1_area = box_area(box1)
    box2_area = box_area(box2)

    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    iou = inter / (box1_area[:, None] + box2_area - inter)

    return iou, box1_area[:, None] + box2_area - inter


def g_iou(box1, box2):
    assert (box1[:, :2] >= box1[:, 2:]).all()
    assert (box2[:, :2] >= box2[:, 2:]).all()

    iou, union = box_iou(box1, box2)

    lt = torch.min(box1[:, None, :2], box2[:, :2])
    rb = torch.max(box1[:, None, 2:], box2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, device=torch.float)
    x = torch.arange(0, w, device=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], dim=-1)
