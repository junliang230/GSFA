import os.path as osp

import mmcv
import numpy as np
from torch.utils.data import Dataset

from .pipelines import Compose
from .registry import DATASETS


@DATASETS.register_module
class CustomDataset(Dataset):
    """Custom dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    CLASSES = None

    def __init__(self,
                 ann_file,
                 pipeline,
                 image_set=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 selsa_offset=None):
        self.ann_file = ann_file
        self.image_set = image_set
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.selsa_offset = selsa_offset

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals)
        self.img_infos = self.load_annotations(self.ann_file)
        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None
        # filter images too small
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # processing pipeline
        self.pipeline = Compose(pipeline)

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, ann_file):
        return mmcv.load(ann_file)

    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def get_triple_idx(self, filename, idx): #TODO: merge get_triple_idx into get_offset_idx
        if 'VID' in filename:
            self.selsa_offset['MAX_OFFSET'] = self.img_infos[idx]['frame_seg_len']
            self.selsa_offset['MIN_OFFSET'] = -self.img_infos[idx]['frame_seg_len']
            # self.selsa_offset['MAX_OFFSET'] = 10
            # self.selsa_offset['MIN_OFFSET'] = -10
            offsets = np.random.choice(self.selsa_offset['MAX_OFFSET'] - self.selsa_offset['MIN_OFFSET'] + 1, 4,replace=False) \
                      + self.selsa_offset['MIN_OFFSET']
            # offsets = np.random.choice(self.selsa_offset['MAX_OFFSET'], 2,replace=False)
            # offset_local = np.random.choice(24 + 1, 1,replace=False) - 12
            # offsets_all = np.array([list(offsets), list(offset_local)])
            # bef_id = min(max(self.img_infos[idx]['frame_seg_id'] + offsets[0], 0),self.img_infos[idx]['frame_seg_len'] - 1)
            # aft_id = min(max(self.img_infos[idx]['frame_seg_id'] + offsets[1], 0),self.img_infos[idx]['frame_seg_len'] - 1)
            idxes = []
            for offset in offsets:
                tmp_idx = min(max(self.img_infos[idx]['frame_seg_id'] + offset, 0),self.img_infos[idx]['frame_seg_len'] - 1)
                # tmp_idx = offset
                idxes.append(tmp_idx)
            return idxes
        elif 'DET' in filename:
            return [idx]*4
        else:
            return None

    def get_offset_idx(self, filename, idx, nums):
        if 'VID' in filename:
            self.selsa_offset['MAX_OFFSET'] = self.img_infos[idx]['frame_seg_len']
            self.selsa_offset['MIN_OFFSET'] = -self.img_infos[idx]['frame_seg_len']
            # offsets = np.random.choice(self.selsa_offset['MAX_OFFSET'] - self.selsa_offset['MIN_OFFSET'] + 1, nums,replace=False) \
            #           + self.selsa_offset['MIN_OFFSET']
            offsets = np.random.choice(self.selsa_offset['MAX_OFFSET'], min(self.selsa_offset['MAX_OFFSET'],nums), replace=False)
            idxes = []
            for offset in offsets:
                # offset_idx = min(max(self.img_infos[idx]['frame_seg_id'] + offset, 0),self.img_infos[idx]['frame_seg_len'] - 1)
                offset_idx = offset
                idxes.append(offset_idx)
            return idxes
        elif 'DET' in filename:
            idxes = []
            for i in range(nums):
                idxes.append(idx)
            return idxes
        else:
            return None

    def get_train_offset_img(self, idx, base_idx):
        import copy
        img_info = copy.deepcopy(self.img_infos[base_idx])
        filename = '%s/%06d'%(img_info['filename'][:img_info['filename'].rfind('/')],idx) + '.JPEG'
        id = '%s/%06d'%(img_info['id'][:img_info['id'].rfind('/')],idx)
        img_info['filename'] = filename
        img_info['id'] = id
        ann_info = self.get_offset_ann_info(img_info)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        data = self.pipeline(results)
        return data

    def get_test_offset_img(self, idx, base_idx):
        import copy
        img_info = copy.deepcopy(self.img_infos[base_idx])
        filename = '%s/%06d'%(img_info['filename'][:img_info['filename'].rfind('/')],idx) + '.JPEG'
        id = '%s/%06d'%(img_info['id'][:img_info['id'].rfind('/')],idx)
        img_info['filename'] = filename
        img_info['id'] = id
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        data = self.pipeline(results)
        return data

    def __getitem__(self, idx):
        if self.test_mode:
            if self.test_frames:
                return self.prepare_test_img(idx)

            import torch
            from mmcv.parallel import DataContainer as DC
            data = self.prepare_test_img(idx)
            idxes = self.get_offset_idx(data['img_meta'][0].data['filename'], idx, 20)#maybe some idxes are same #TODO: set in cfg
            test_imgs = torch.tensor(data['img'][0])
            for idx_ in idxes:
                support_img = self.get_test_offset_img(idx_, idx)['img'][0]
                test_imgs = torch.cat((test_imgs, support_img), 0)

            data['img'][0] = test_imgs #TODO: maybe need to consider batchsize>1 in test

            return data
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue

            import torch
            from mmcv.parallel import DataContainer as DC
            triple_idxes = self.get_triple_idx(data['img_meta'].data['filename'], idx) #TODO: in cfg
            triple_img = torch.tensor(data['img'].data)
            triple_gt = torch.tensor(data['gt_bboxes'].data)
            triple_gt_labels = torch.tensor(data['gt_labels'].data)
            triple_img_meta = data['img_meta'].data
            triple_img_meta['img_norm_cfg']['img_meta_ba'] = []
            for k, idx_ in enumerate(triple_idxes):
                if idx_ == idx:
                    train_img = self.prepare_train_img(idx)
                    support_img = train_img['img'].data.numpy()
                    support_gt = train_img['gt_bboxes'].data.numpy()
                    support_gt_labels = train_img['gt_labels'].data.numpy()
                    support_img_meta = train_img['img_meta'].data
                    support_img = np.ascontiguousarray(support_img.transpose(1, 2, 0))
                    if support_img_meta['pad_shape'][0] != triple_img_meta['pad_shape'][0] or support_img_meta['pad_shape'][1] != triple_img_meta['pad_shape'][1]:
                        support_img = support_img[:support_img_meta['img_shape'][0], :support_img_meta['img_shape'][1],...]
                        support_img, w_scale, h_scale = mmcv.imresize(support_img,(data['img'].data.size(2), data['img'].data.size(1)), return_scale=True)
                        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
                        support_img_meta['scale_factor'] = support_img_meta['scale_factor']*scale_factor
                        support_img_meta['pad_shape'] = triple_img_meta['pad_shape']
                        support_img_meta['img_shape'] = triple_img_meta['pad_shape']
                        support_gt = support_gt * scale_factor
                        img_shape = [support_img.shape[0], support_img.shape[1]]
                        support_gt[:, 0::2] = np.clip(support_gt[:, 0::2], 0, img_shape[1] - 1)
                        support_gt[:, 1::2] = np.clip(support_gt[:, 1::2], 0, img_shape[0] - 1)
                        # for bbox in support_gt:
                        #     import cv2
                        #     bbox_int = bbox.astype(np.float).astype(np.int32)
                        #     left_top = (bbox_int[0], bbox_int[1])
                        #     right_bottom = (bbox_int[2], bbox_int[3])
                        #     cv2.rectangle(support_img, left_top, right_bottom, (255, 100, 100), thickness=2)  # blue
                        # cv2.imwrite('demo.jpg', support_img)
                    triple_img = torch.cat((triple_img, torch.from_numpy(support_img).permute(2, 0, 1)), 0)
                    triple_gt = torch.cat((triple_gt, torch.tensor([[-1, -1, -1, -1.0]]), torch.from_numpy(support_gt)), 0)
                    triple_gt_labels = torch.cat((triple_gt_labels, torch.tensor([-1]), torch.from_numpy(support_gt_labels)), 0)
                    triple_img_meta['img_norm_cfg']['img_meta_ba'].append(support_img_meta)
                else:
                    while True:
                        data_ = self.get_train_offset_img(idx_, idx)
                        if data_ is None:
                            idx_ = self.get_triple_idx(data['img_meta'].data['filename'], idx)[k]
                            continue
                        else:
                            break
                    support_img = data_['img'].data.numpy()
                    support_gt = data_['gt_bboxes'].data.numpy()
                    support_gt_labels = data_['gt_labels'].data.numpy()
                    support_img_meta = data_['img_meta'].data
                    support_img = np.ascontiguousarray(support_img.transpose(1, 2, 0))
                    if support_img_meta['pad_shape'][0] != triple_img_meta['pad_shape'][0] or support_img_meta['pad_shape'][1] != triple_img_meta['pad_shape'][1]:
                        support_img = support_img[:support_img_meta['img_shape'][0], :support_img_meta['img_shape'][1],...]
                        support_img, w_scale, h_scale = mmcv.imresize(support_img,(data['img'].data.size(2), data['img'].data.size(1)), return_scale=True)
                        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
                        support_img_meta['scale_factor'] = support_img_meta['scale_factor']*scale_factor
                        support_img_meta['pad_shape'] = triple_img_meta['pad_shape']
                        support_img_meta['img_shape'] = triple_img_meta['pad_shape']
                        support_gt = support_gt * scale_factor
                        img_shape = [support_img.shape[0], support_img.shape[1]]
                        support_gt[:, 0::2] = np.clip(support_gt[:, 0::2], 0, img_shape[1] - 1)
                        support_gt[:, 1::2] = np.clip(support_gt[:, 1::2], 0, img_shape[0] - 1)
                        # for bbox in support_gt:
                        #     import cv2
                        #     bbox_int = bbox.astype(np.float).astype(np.int32)
                        #     left_top = (bbox_int[0], bbox_int[1])
                        #     right_bottom = (bbox_int[2], bbox_int[3])
                        #     cv2.rectangle(support_img, left_top, right_bottom, (255, 100, 100), thickness=2)  # blue
                        # cv2.imwrite('demo.jpg', support_img)
                    triple_img = torch.cat((triple_img, torch.from_numpy(support_img).permute(2, 0, 1)), 0)
                    triple_gt = torch.cat((triple_gt, torch.tensor([[-1, -1, -1, -1.0]]), torch.from_numpy(support_gt)), 0)
                    triple_gt_labels = torch.cat((triple_gt_labels, torch.tensor([-1]), torch.from_numpy(support_gt_labels)), 0)
                    triple_img_meta['img_norm_cfg']['img_meta_ba'].append(support_img_meta)
            del data['img']
            del data['gt_bboxes']
            del data['gt_labels']
            data['img'] = DC(triple_img, stack=True)
            data['gt_bboxes'] = DC(triple_gt)
            data['gt_labels'] = DC(triple_gt_labels)
            return data

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)
