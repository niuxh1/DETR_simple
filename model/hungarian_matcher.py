import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from utils.box_ops import box_to_xy, g_iou


class hungarian_matcher(nn.Module):
    def __init__(self, class_cost: float = 1, bbox_cost: float = 1, giou_cost: float = 1):
        super().__init__()

        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost

    @torch.no_grad()
    def forward(self, outputs, targets):
        batch,num_query=outputs["logits"].shape[:2]

        out_prob = outputs["logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["boxes"].flatten(0, 1)

        target_ids=torch.cats([v["labels"] for v in targets])
        target_bbox=torch.cat([v["boxes"] for v in targets])

        class_cost=-out_prob[:,target_ids]
        bbox_cost=torch.cdist(out_bbox,target_bbox, p=1)
        giou_cost=-g_iou(box_to_xy(out_bbox),box_to_xy(target_bbox))

        cost_all=self.bbox_cost*bbox_cost+self.class_cost*class_cost+self.giou_cost*giou_cost
        cost_all=cost_all.view(batch,num_query,-1).cpu()

        sizes=[len(v["boxes"]) for v in targets]
        indices=[linear_sum_assignment(c[i]) for i,c in enumerate(cost_all.split(sizes,-1))]

        return [(torch.as_tensor(i,dtype=torch.int64),torch.as_tensor(j,dtype=torch.int64)) for i,j in indices]


def matcher(class_cost: float = 1, bbox_cost: float = 1, giou_cost: float = 1):
    m=hungarian_matcher(class_cost, bbox_cost, giou_cost)
    return m