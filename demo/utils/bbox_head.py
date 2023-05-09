import torch
import torch.nn.functional as F

from utils.bbox_decoder import TransFusionBBoxCoder


class BboxesHead():
    def __init__(
        self,
        num_proposals=128,
        num_classes=4,
        # loss
        loss_cls=dict(type="GaussianFocalLoss", reduction="mean"),
        # others
        bbox_coder=None,
    ):
        self.num_classes = num_classes
        self.num_proposals = num_proposals

        self.use_sigmoid_cls = loss_cls.get("use_sigmoid", False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1

        # ===
        pc_range = bbox_coder['pc_range']
        out_size_factor = bbox_coder['out_size_factor']
        voxel_size = torch.tensor(bbox_coder['voxel_size'])
        post_center_range = torch.tensor(bbox_coder['post_center_range'])
        score_threshold = bbox_coder['score_threshold']
        code_size = bbox_coder['code_size']
        self.bbox_coder = TransFusionBBoxCoder(pc_range, out_size_factor, voxel_size, post_center_range, score_threshold, code_size)
        # ===

    def get_bboxes(self, preds_dicts):
        # kevin onnx 
        # dict-->list
        # [center,height,dim,rot,vel,heatmap, query_heatmap_score, dense_heatmap, self_query_labels]
        # [     0     1,    2  3 ,  4,   5,            6,             7         ,                 8]
        # rets = []
        
        self_query_labels = preds_dicts[0][0][8]

        preds_dict = preds_dicts[0]
        batch_size = preds_dict[0][5].shape[0]
        batch_score = preds_dict[0][5][..., -self.num_proposals :].sigmoid()

        one_hot = F.one_hot(
            self_query_labels, num_classes=self.num_classes
        ).permute(0, 2, 1)
        batch_score = batch_score * preds_dict[0][6] * one_hot

        batch_center = preds_dict[0][0][..., -self.num_proposals :]
        batch_height = preds_dict[0][1][..., -self.num_proposals :]
        batch_dim = preds_dict[0][2][..., -self.num_proposals :]
        batch_rot = preds_dict[0][3][..., -self.num_proposals :]
        batch_vel = preds_dict[0][4][..., -self.num_proposals :]

        temp = self.bbox_coder.decode(
            batch_score,
            batch_rot,
            batch_dim,
            batch_center,
            batch_height,
            batch_vel,
            filter=True,
        )

        boxes3d = temp[0]["bboxes"]
        scores = temp[0]["scores"]
        labels = temp[0]["labels"]

        ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
        ret_layer= [ret]
        rets = [ret_layer]
        
        assert len(rets) == 1
        assert len(rets[0]) == 1
        
        res = [
            [
                rets[0][0]["bboxes"],
                rets[0][0]["scores"],
                rets[0][0]["labels"].int(),
            ]
        ]
        return res