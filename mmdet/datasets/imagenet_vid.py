import os.path as osp
import os
import xml.etree.ElementTree as ET
import random
import mmcv
import numpy as np

from .custom import CustomDataset
from .registry import DATASETS
from string import digits

@DATASETS.register_module
class VIDDataset(CustomDataset):
    CLASSES = ['airplane', 'antelope', 'bear', 'bicycle',
                    'bird', 'bus', 'car', 'cattle',
                    'dog', 'domestic_cat', 'elephant', 'fox',
                    'giant_panda', 'hamster', 'horse', 'lion',
                    'lizard', 'monkey', 'motorcycle', 'rabbit',
                    'red_panda', 'sheep', 'snake', 'squirrel',
                    'tiger', 'train', 'turtle', 'watercraft',
                    'whale', 'zebra']
    classes_map = ['n02691156', 'n02419796', 'n02131653', 'n02834778',
                        'n01503061', 'n02924116', 'n02958343', 'n02402425',
                        'n02084071', 'n02121808', 'n02503517', 'n02118333',
                        'n02510455', 'n02342885', 'n02374451', 'n02129165',
                        'n01674464', 'n02484322', 'n03790512', 'n02324045',
                        'n02509815', 'n02411705', 'n01726692', 'n02355227',
                        'n02129604', 'n04468005', 'n01662784', 'n04530566',
                        'n02062744', 'n02391049']

    def __init__(self, min_size=None, **kwargs):
        super(VIDDataset, self).__init__(**kwargs)

        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.classes_map)}
        self.min_size = min_size

    def load_annotations(self, ann_file):
        img_infos = []
        image_sets = [iset for iset in self.image_set.split('+')]
        for image_set in image_sets:
            det_vid = image_set.split('_')[0]
            ann_file = osp.join(self.ann_file, 'ImageSets', image_set + '.txt')
            with open(ann_file) as f:
                img_ids = [x.strip().split(' ') for x in f.readlines()]
            if len(img_ids[0]) == 2:
                for img_id in img_ids:
                    xml_path = osp.join(self.ann_file, 'Annotations', det_vid, img_id[0] + '.xml')
                    filename = osp.join('Data', det_vid, img_id[0] + '.JPEG')
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    size = root.find('size')
                    width = int(float(size.find('width').text))
                    height = int(float(size.find('height').text))
                    if height <= 0 or width <= 0:
                        continue
                    img_infos.append(
                        dict(id=img_id[0], filename=filename, width=width, height=height, det_vid=det_vid))
            else:
                for img_id in img_ids:
                    xml_path = osp.join(self.ann_file, 'Annotations', det_vid, '%s/%06d'%(img_id[0],int(img_id[2])) + '.xml')
                    filename = osp.join('Data', det_vid, '%s/%06d'%(img_id[0],int(img_id[2])) + '.JPEG')
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    size = root.find('size')
                    width = int(float(size.find('width').text))
                    height = int(float(size.find('height').text))
                    if height <= 0 or width <= 0:
                        continue
                    img_infos.append(
                        dict(id='%s/%06d'%(img_id[0],int(img_id[2])), filename=filename, width=width, height=height,
                             frame_id=int(img_id[1]), frame_seg_id=int(img_id[2]), frame_seg_len=int(img_id[3]), det_vid=det_vid))
        return img_infos[45:46]

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        xml_path = osp.join(self.ann_file, 'Annotations', self.img_infos[idx]['det_vid'], img_id + '.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.classes_map:
                continue
            label = self.cat2label[name]

            difficult = False
            bnd_box = obj.find('bndbox')
            bbox = [
                int(bnd_box.find('xmin').text),
                int(bnd_box.find('ymin').text),
                int(bnd_box.find('xmax').text),
                int(bnd_box.find('ymax').text)
            ]

            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w <= 0 or w > self.img_infos[idx]['width'] or h <= 0 or h > self.img_infos[idx]['height']:
                continue

            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2)
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2)
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann

    def get_offset_ann_info(self, img_info):
        img_id = img_info['id']
        xml_path = osp.join(self.ann_file, 'Annotations', img_info['det_vid'], img_id + '.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.classes_map:
                continue
            label = self.cat2label[name]

            difficult = False
            bnd_box = obj.find('bndbox')
            bbox = [
                int(bnd_box.find('xmin').text),
                int(bnd_box.find('ymin').text),
                int(bnd_box.find('xmax').text),
                int(bnd_box.find('ymax').text)
            ]

            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w <= 0 or w > img_info['width'] or h <= 0 or h > img_info['height']:
                continue

            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2)
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2)
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann
