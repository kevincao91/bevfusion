import torch

from mmdet.core.bbox.builder import BBOX_CODERS


@BBOX_CODERS.register_module()
class TransFusionBBoxCoder(object):
    def __init__(self,
                 pc_range,
                 out_size_factor:int,
                 voxel_size,
                 post_center_range,
                 score_threshold:float=0.0,
                 code_size:int=8,
                 ):
        self.pc_range = pc_range
        self.out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.score_threshold = score_threshold
        self.code_size = code_size


    def decode(self, heatmap, rot, dim, center, height, vel, filter:bool=False):
        """Decode bboxes.
        Args:
            heat (torch.Tensor): Heatmap with the shape of [B, num_cls, num_proposals].
            rot (torch.Tensor): Rotation with the shape of
                [B, 1, num_proposals].
            dim (torch.Tensor): Dim of the boxes with the shape of
                [B, 3, num_proposals].
            center (torch.Tensor): bev center of the boxes with the shape of
                [B, 2, num_proposals]. (in feature map metric)
            hieght (torch.Tensor): height of the boxes with the shape of
                [B, 2, num_proposals]. (in real world metric)
            vel (torch.Tensor): Velocity with the shape of [B, 2, num_proposals].
            filter: if False, return all box without checking score and center_range
        Returns:
            list[dict]: Decoded boxes.
        """
        # class label
        # final_preds = heatmap.max(1, keepdims=False).indices
        # final_scores = heatmap.max(1, keepdims=False).values
        final_preds = heatmap.max(1).indices
        final_scores = heatmap.max(1).values

        # kevin onnx
        center = center
        # ====


        # change size to real world metric
        center[:, 0, :] = center[:, 0, :] * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
        center[:, 1, :] = center[:, 1, :] * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
        # center[:, 2, :] = center[:, 2, :] * (self.post_center_range[5] - self.post_center_range[2]) + self.post_center_range[2]
        dim[:, 0, :] = dim[:, 0, :].exp()
        dim[:, 1, :] = dim[:, 1, :].exp()
        dim[:, 2, :] = dim[:, 2, :].exp()
        height = height - dim[:, 2:3, :] * 0.5  # gravity center to bottom center
        rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
        
        # kevin onnx
        # rot = torch.atan2(rots, rotc)
        rot = torch.atan(rots/(rotc+1e-6)) 
        # ===

        if vel is None:
            final_box_preds = torch.cat([center, height, dim, rot], dim=1).permute(0, 2, 1)
        else:
            final_box_preds = torch.cat([center, height, dim, rot, vel], dim=1).permute(0, 2, 1)

        # predictions_dicts = []
        # kevin onnx
        # for i in range(heatmap.shape[0]):
        boxes3d = final_box_preds[0]
        scores = final_scores[0]
        labels = final_preds[0]
        predictions_dict = {
            'bboxes': boxes3d,
            'scores': scores,
            'labels': labels
        }
        predictions_dicts = [predictions_dict]
        # ===

        if filter is False:
            return predictions_dicts

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=heatmap.device)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(2)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(2)

            # predictions_dicts = []
            # for i in range(heatmap.shape[0]):
            cmask = mask[0, :]
            if self.score_threshold:
                cmask &= thresh_mask[0]

            boxes3d = final_box_preds[0, cmask]
            scores = final_scores[0, cmask]
            labels = final_preds[0, cmask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels
            }
            predictions_dicts = [predictions_dict]
            # ===
        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')

        return predictions_dicts