import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from .test_mixins import BBoxTestMixin, MaskTestMixin, RPNTestMixin


@DETECTORS.register_module
class TwoStageDetector(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 selsa_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        # self.compute_similar = RelationNetwork()
        # self.compute_similar.apply(weights_init)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if selsa_head is not None:
            self.selsa_head = builder.build_selsa_head(selsa_head)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)

        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor
            self.mask_head = builder.build_head(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(TwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_selsa_head:
            self.selsa_head.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmedetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).cuda()
        # bbox head
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            outs = outs + (cls_score, bbox_pred)
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)
            outs = outs + (mask_pred, )
        return outs

    def forward_train(self,
                      imgs,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_meta (list[dict]): list of image info dict where each dict has:
                'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        imgs = imgs.split(3, 1)
        imgs = torch.cat(imgs)
        # cat_xes = self.extract_feat(imgs)
        cat_xes = self.extract_feat(imgs)  # 用gt在img中提取出feature
        # frame_ious = self.compute_ious(gt_bboxes[0], gt_labels[0])
        # _, sup_inds = frame_ious[1:].topk(3, 0)
        # ref = layer1_feat[0].unsqueeze(0).repeat(layer1_feat.size(0), 1, 1, 1)
        # relation_pair = -self.compute_similar(torch.cat((layer1_feat, ref), 1))
        # _, sup_inds = relation_pair[1:].topk(3, 0)
        # loss_silimar = {}
        # loss_silimar['loss_silimar'] = torch.mean(torch.abs(frame_ious.unsqueeze(1)+relation_pair))
        # loss_silimar['loss_silimar'] = max(1+relation_pair[0], torch.tensor(0.0, device=relation_pair.device))
        coses = {}
        # coses = ((1, 1), (2, 2))
        for i in range(1, cat_xes[0].size(0)):
            # cos = torch.mean(F.cosine_similarity(cat_xes[0][0].view(1, cat_xes[0].size(1), -1),
            #                                      cat_xes[0][i].view(1, cat_xes[0].size(1), -1), dim=2))
            l1_dis = -F.l1_loss(cat_xes[0][0], cat_xes[0][i])
            coses[i] = l1_dis.item() #l1_dis.item()+frame_ious[i] #调换key和value试试

        coses = sorted(coses.items(), key=lambda item: item[1])
        spa, spb = coses[0][0], coses[-1][0]
        # spa, spb = sup_inds[0].item()+1, sup_inds[-1].item()+1
        x = (cat_xes[0][0].unsqueeze(0),)
        x_ba = (torch.cat((cat_xes[0][spa].unsqueeze(0), cat_xes[0][spb].unsqueeze(0)), 0),)
        gt_dict = {0: [], 1: [], 2: [], 3:[], 4:[]}
        gt_labels_dict = {0: [], 1: [], 2: [], 3:[], 4:[]}
        i = 0
        for j in range(len(gt_bboxes[0])):
            if -1 in gt_bboxes[0][j]:
                i += 1
                continue
            gt_dict[i].append(gt_bboxes[0][j].unsqueeze(0))
            gt_labels_dict[i].append(gt_labels[0][j].unsqueeze(0))
        if len(gt_dict[1]) == 0 or len(gt_dict[2]) == 0:
            print(gt_bboxes)
        gt_ref = [torch.cat(gt_dict[0])]
        gt_ref_labels = [torch.cat(gt_labels_dict[0])]
        gt_ab = [torch.cat(gt_dict[spa]), torch.cat(gt_dict[spb])]
        gt_ab_labels = [torch.cat(gt_labels_dict[spa]), torch.cat(gt_labels_dict[spb])]
        img_meta_ba = [img_meta[0]['img_norm_cfg']['img_meta_ba'][spa - 1],
                       img_meta[0]['img_norm_cfg']['img_meta_ba'][spb - 1]]
        # img, bef_im, aft_im = imgs.split(3, 1)
        # x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_ref, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            # bbox_random_sampler = build_sampler(
            #     self.train_cfg.rcnn.random_sampler, context=self)
            num_imgs = len(img_meta)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[i],
                                                     gt_ref[i],
                                                     gt_bboxes_ignore[i],
                                                     gt_ref_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_ref[i],
                    gt_ref_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                # random_sampling_result = bbox_random_sampler.sample(
                #     assign_result,
                #     proposal_list[i],
                #     gt_ref[i],
                #     gt_ref_labels[i],
                #     feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # plot_iou_distribution(assign_result, sampling_results[0], 'hard')
        # plot_iou_distribution(assign_result, random_sampling_result, 'random')

        if self.with_selsa_head:
            # img_ba = torch.cat((bef_im, aft_im), 0)
            # x_ba = self.extract_feat(img_ba)
            # img_meta_ba = [*img_meta, *img_meta]
            if self.with_rpn:
                rpn_outs_ba = self.rpn_head(x_ba)
                proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                  self.test_cfg.rpn)
                proposal_inputs_ba = rpn_outs_ba + (img_meta_ba, proposal_cfg)
                proposal_list_ba = self.rpn_head.get_bboxes(*proposal_inputs_ba)
            else:
                proposal_list_ba = proposals

            if self.with_bbox or self.with_mask:
                sampler_ba = self.train_cfg.rcnn.sampler.copy()
                sampler_ba['pos_fraction'] = 0.5
                bbox_sampler_ba = build_sampler(
                    sampler_ba, context=self)
                num_imgs = len(img_meta_ba)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]
                sampling_results_ba = []
                for i in range(num_imgs):
                    assign_result_ba = bbox_assigner.assign(proposal_list_ba[i],
                                                         gt_ab[i],
                                                         gt_bboxes_ignore[0],
                                                         gt_ab_labels[i])
                    sampling_result_ba = bbox_sampler_ba.sample(
                        assign_result_ba,
                        proposal_list_ba[i],
                        gt_ab[i],
                        gt_ab_labels[i],
                        feats=[lvl_feat[i][None] for lvl_feat in x_ba])
                    sampling_results.append(sampling_result_ba)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            if self.with_shared_head:
                bbox_feats = self.shared_head(torch.cat((x[0], x_ba[0]), 0))

            bbox_feats = self.bbox_roi_extractor(
                [bbox_feats], rois)
            if self.with_selsa_head:
                # bbox_feats = self.selsa_head(bbox_feats)
                ohem_bbox_feats_idx = []
                with torch.no_grad():
                    new_bbox_feats = self.selsa_head(bbox_feats) #移到torch.no_grad()之外呢
                    for i, bbox_feat in enumerate(new_bbox_feats.chunk(3, 0)):
                        # bbox_feat = self.selsa_head(bbox_feat) #这里加上试试？
                        cls_score, _ = self.bbox_head(bbox_feat)
                        bbox_targets = self.bbox_head.get_target([sampling_results[i]],
                                                                 gt_bboxes, gt_labels,
                                                                 self.train_cfg.rcnn)
                        # topk_loss_inds = []
                        # pos_inds = (bbox_targets[0]>0).nonzero()
                        # neg_inds = (bbox_targets[0]==0).nonzero()
                        loss = self.bbox_head.loss(
                            cls_score=cls_score,
                            bbox_pred=None,
                            labels=bbox_targets[0],
                            label_weights=cls_score.new_ones(cls_score.size(0)),
                            bbox_targets=None,
                            bbox_weights=None,
                            reduction_override='none')['loss_cls']
                        _, loss_inds = loss.topk(300) #按照正负样本进行选取
                        # fraction = 0.25
                        # for inds in loss_inds:
                        #     if inds in pos_inds:
                        #         topk_loss_inds.append(inds)
                        # topk_loss_inds = topk_loss_inds[:int(300*fraction)]
                        # for inds in loss_inds:
                        #     if inds in neg_inds:
                        #         topk_loss_inds.append(inds)
                        # topk_loss_inds = topk_loss_inds[:300]
                        # if i == 0:
                        #     plot_iou_distribution(assign_result, sampling_results[0], 'PLSM', topk_loss_inds)
                        ohem_bbox_feats_idx.append(loss_inds)
                tmp_bbox_feats = None
                for i, bbox_feat in enumerate(bbox_feats.chunk(3, 0)): #把这部分移到with_grad()中去
                    if tmp_bbox_feats is None:
                        tmp_bbox_feats = bbox_feat[ohem_bbox_feats_idx[i]]
                    else:
                        tmp_bbox_feats = torch.cat((tmp_bbox_feats, bbox_feat[ohem_bbox_feats_idx[i]]), 0)
                bbox_feats = self.selsa_head(tmp_bbox_feats)
            # bbox_feats, _ = self.selsa_head(bbox_feats) #加上试试？
            bbox_feats = bbox_feats.chunk(3, 0)[0]
            # bbox_feats = self.selsa_head(bbox_feats) #加上试试？
            cls_score, bbox_pred = self.bbox_head(bbox_feats) #whether slice has a problem

            bbox_targets = self.bbox_head.get_target([sampling_results[0]],
                                                     gt_bboxes, gt_labels,
                                                     self.train_cfg.rcnn)
            # bbox_targets_all = self.bbox_head.get_target(sampling_results,
            #                                              [*gt_ref, *gt_ab], [*gt_ref_labels, *gt_ab_labels],
            #                                              self.train_cfg.rcnn)
            #
            # target = (bbox_targets_all[0].unsqueeze(1) == bbox_targets_all[0].unsqueeze(0)).int()
            # graph_loss = {}
            # similar = similar.squeeze(1)
            # # similar = (similar-similar.mean(1).unsqueeze(1)) / similar.std(1).unsqueeze(1)
            # similar = (similar-similar.min(1)[0].unsqueeze(1)) / torch.max((similar.max(1)[0].unsqueeze(1)-similar.min(1)[0].unsqueeze(1)), torch.ones_like(similar) * 1e-3)
            # # graph_loss['loss_graph'] = -(((1-target)*torch.log(1-similar) + target*torch.log(similar)).mean())*0.01
            # graph_loss['loss_graph'] = max(torch.mean(similar[target==0])-torch.mean(similar[target==1])+1, torch.tensor(0.0, device=similar.device)) #1
            # # graph_loss['loss_graph'] = max(torch.mean(similar[~target])/torch.mean(similar[target]), torch.tensor(0.0, device=similar.device)) #1
            # losses.update(graph_loss)
            ohem_bbox_targets = []
            for i in range(len(bbox_targets)):
                ohem_bbox_targets.append(bbox_targets[i][ohem_bbox_feats_idx[0]])
            # losses.update(loss_silimar)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *ohem_bbox_targets)
            losses.update(loss_bbox)

        # mask head forward and loss
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(
                            res.pos_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                    pos_inds.append(
                        torch.zeros(
                            res.neg_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]

            if mask_feats.shape[0] > 0:
                mask_pred = self.mask_head(mask_feats)
                mask_targets = self.mask_head.get_target(
                    sampling_results, gt_masks, self.train_cfg.rcnn)
                pos_labels = torch.cat(
                    [res.pos_gt_labels for res in sampling_results])
                loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                                pos_labels)
                losses.update(loss_mask)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        imgs = img.split(3, 1)
        imgs_cat = torch.cat(imgs, 0)
        x = self.extract_feat(imgs_cat)
        img_meta = [img_meta[0]] * int(x[0].size(0))

        if proposals is None:
            proposal_list = await self.async_test_rpn(x, img_meta,
                                                      self.test_cfg.rpn)
        else:
            proposal_list = proposals

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_meta,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    async def async_simple_test_frames(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."
        if hasattr(self, 'get_roi_feats') and self.get_roi_feats:
            x = self.extract_feat(img)

            if proposals is None:
                proposal_list = await self.async_test_rpn(x, img_meta,
                                                          self.test_cfg.rpn)
            else:
                proposal_list = proposals

            return await self.async_test_bboxes_frames(
                x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)

        det_bboxes, det_labels = await self.async_test_bboxes_frames(
            None, img_meta, None, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                None,
                img_meta,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test_frames(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        if hasattr(self, 'get_roi_feats') and self.get_roi_feats:
            x = self.extract_feat(img)

            if proposals is None:
                proposal_list = self.simple_test_rpn(x, img_meta,
                                                     self.test_cfg.rpn)
            else:
                proposal_list = proposals

            return self.simple_test_bboxes_frames(
                x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)

        det_bboxes, det_labels = self.simple_test_bboxes_frames(
            None, img_meta, None, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                None, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        imgs = img.split(3, 1)
        imgs_cat = torch.cat(imgs, 0)
        x = self.extract_feat(imgs_cat)
        img_meta = [img_meta[0]] * int(x[0].size(0))

        if proposals is None:
            proposal_list = self.simple_test_rpn(x, img_meta,
                                                 self.test_cfg.rpn)
        else:
            proposal_list = proposals

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results

    def bbox_overlaps(self, bboxes1, bboxes2, mode='iou', is_aligned=False):

        assert mode in ['iou', 'iof']

        rows = bboxes1.size(0)
        cols = bboxes2.size(0)
        if is_aligned:
            assert rows == cols

        if rows * cols == 0:
            return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

        if is_aligned:
            lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
            rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

            wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
            overlap = wh[:, 0] * wh[:, 1]
            area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
                    bboxes1[:, 3] - bboxes1[:, 1] + 1)

            if mode == 'iou':
                area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                        bboxes2[:, 3] - bboxes2[:, 1] + 1)
                ious = overlap / (area1 + area2 - overlap)
            else:
                ious = overlap / area1
        else:
            lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
            rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

            wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
            overlap = wh[:, :, 0] * wh[:, :, 1]
            area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
                    bboxes1[:, 3] - bboxes1[:, 1] + 1)

            if mode == 'iou':
                area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                        bboxes2[:, 3] - bboxes2[:, 1] + 1)
                ious = overlap / (area1[:, None] + area2 - overlap)
            else:
                ious = overlap / (area1[:, None])

        return ious[0]

    def compute_ious(self, gt_bboxes, gt_labels):
        gt_list = []
        tmp = []
        for i in range(len(gt_bboxes)):
            if -1 in gt_bboxes[i]:
                gt_list.append(tmp)
                tmp = []
            else:
                tmp.append([gt_bboxes[i], gt_labels[i]])
        gt_list.append(tmp)
        frame_ious = torch.tensor([1.0], device=gt_labels.device)
        for i in range(1, len(gt_list)):
            ious = torch.tensor([0.0], device=gt_labels.device)
            for j in range(len(gt_list[i])):
                for k in range(len(gt_list[0])):
                    if gt_list[i][j][1] == gt_list[0][k][1]:
                        cur_ious = self.bbox_overlaps(gt_list[i][j][0].unsqueeze(0), gt_list[0][k][0].unsqueeze(0))
                        ious = torch.cat((ious, cur_ious)) #判断iou是否等于0？
            if len(ious) == 1:
                frame_ious = torch.cat((frame_ious,ious))
            else:
                frame_ious = torch.cat((frame_ious,torch.max(ious[1:]).unsqueeze(0)))
        return frame_ious

# https://github.com/floodsung/LearningToCompare_FSL/blob/master/omniglot/omniglot_train_few_shot.py#L74
class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size=784,hidden_size=16):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(2048,256,kernel_size=3,padding=1),
                        nn.BatchNorm2d(256, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        # self.layer12 = nn.Sequential(
        #                 nn.Conv2d(256,64,kernel_size=3,padding=1),
        #                 nn.BatchNorm2d(64, momentum=1, affine=True),
        #                 nn.ReLU(),
        #                 nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(256,16,kernel_size=3,padding=1),
                        nn.BatchNorm2d(16, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))

        self.avg_pool = nn.AdaptiveAvgPool2d((7,7))
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        # x = F.interpolate(x, size=[32, 32], mode="bilinear", align_corners=True) #最近邻？缩小尺度增加卷积channel？
        out = self.layer1(x)
        # out = self.layer12(out)
        out = self.layer2(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out

def weights_init(m):
    import math
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


def plot_iou_distribution(assign_result, sampling_results, flag = None, loss_idx=None):
    iou_dict = {}
    if loss_idx is not None:
        boxes_inds = torch.cat([sampling_results.pos_inds, sampling_results.neg_inds])
        with open('./iou_distribution/'+flag+'_iou_distribution.txt', 'w') as f:
            for idx in loss_idx:
                f.write(str(assign_result.max_overlaps[boxes_inds[idx]].item())+'\n')
    else:
        with open('./iou_distribution/'+flag+'_iou_distribution.txt', 'w') as f:
            for idx in sampling_results.pos_inds:
                f.write(str(assign_result.max_overlaps[idx].item())+'\n')
            for idx in sampling_results.neg_inds:
                f.write(str(assign_result.max_overlaps[idx].item())+'\n')
            # for iou in assign_result.max_overlaps:
            #     f.write(str(iou.item())+'\n')