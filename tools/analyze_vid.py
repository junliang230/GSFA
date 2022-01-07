import os.path as osp
import os
import xml.etree.ElementTree as ET
import random
import mmcv
import numpy as np

from  mmdet.datasets.custom import CustomDataset
from string import digits

class VIDDataset():
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
        self.data_root = '/media/data1/jliang/dataset/ILSVRC/'
        self.ann_file = '/media/data1/jliang/dataset/ILSVRC/'
        self.image_set = 'VID_val_videos' #VID_train_15frames+DET_train_30classes
        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.classes_map)}
        self.min_size = min_size

    def load_annotations(self):
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
        return img_infos

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
            occluded = obj.find('occluded').text
            generated = obj.find('generated').text
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
            occluded = occluded,
            generated = generated,
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))

        return ann

    def get_ann_info_fromid(self, idx, id):
        img_id = id
        xml_path = osp.join(self.ann_file, 'Annotations', self.img_infos[idx]['det_vid'], img_id + '.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        occluded = 0
        generated = 0
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.classes_map:
                continue
            label = self.cat2label[name]
            occluded += int(obj.find('occluded').text)
            generated += int(obj.find('generated').text)
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
            occluded = occluded,
            generated = generated,
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))

        return ann

    def generate_shuffle_val_frames(self):
        import numpy as np
        img_infos = []
        last_seg_len = None
        video_shuffle = True
        with open('/home/hschen/Sequence-Level-Semantics-Aggregation/data/ILSVRC/ImageSets/VID_val_frames_shuffle.txt', 'w') as f:
            for i in range(len(self.img_infos)):
                img_info = self.img_infos[i].copy()
                cur_seg_len = img_info['frame_seg_len']
                last_seg_len = cur_seg_len if last_seg_len is None else cur_seg_len+last_seg_len
                video_index = np.arange(cur_seg_len)
                if video_shuffle:
                    np.random.shuffle(video_index)
                for cur_frameid in range(len(video_index)):
                    l0 = img_info['id'][:img_info['id'].rfind('/')]
                    l1 = str(video_index[cur_frameid] + last_seg_len - cur_seg_len)
                    l2 = str(video_index[cur_frameid])
                    l3 = str(cur_seg_len)
                    f.write(l0+' '+l1+' '+l2+' '+l3+'\n')

    def analyze_valset(self):
        seqes_val = {}
        for i in range(len(vid.img_infos)):
            seq_val = {'occluded':0, 'generated':0}
            seq_len = vid.img_infos[i]['frame_seg_len']
            base_id = vid.img_infos[i]['id'][:vid.img_infos[i]['id'].rfind('/')]
            class_seq = {}
            for j in range(seq_len):
                ann = self.get_ann_info_fromid(i, '%s/%06d'%(base_id, j))
                for label in ann['labels']:
                    class_seq[self.CLASSES[label-1]] = class_seq.get(self.CLASSES[label-1], 0) + 1
                seq_val['occluded'] += int(ann['occluded'])
                seq_val['generated'] += int(ann['generated'])
            seq_val['class'] = sorted(class_seq.items(), key=lambda item:item[1], reverse=True)[0][0]
            seqes_val[base_id] = seq_val
            print(base_id+':', seq_val)
        print('======================================================================\n', seqes_val)

    def filter_result(self, frame_results):
        filtered_result = {}
        idx_result = {}
        for idx, frame_result in enumerate(frame_results):
            label = int(frame_result[0])
            score = float(frame_result[1])
            if label not in filtered_result:
                filtered_result[label] = score
                idx_result[label] = idx
            elif score > filtered_result[label]:
                filtered_result[label] = score
                idx_result[label] = idx

        final_result = []
        for k, v in idx_result.items():
            final_result.append(frame_results[v])

        return final_result

    def show_result(self):
        import cv2
        import pickle
        data_root = '/media/data1/jliang/dataset/ILSVRC/'
        result_path = '/media/data2/jliang/detection/mmdetection/result'
        val_videos_path = os.path.join(data_root, 'ImageSets/VID_val_videos.txt')
        with open("/media/data2/jliang/detection/mmdetection/img_info_baseline.pkl", 'rb') as f:
            imgs_info_baseline = pickle.load(f, encoding='bytes')
        with open("/media/data2/jliang/detection/mmdetection/result_baseline.pkl", 'rb') as f:
            results_baseline = pickle.load(f, encoding='bytes')
        with open("/media/data2/jliang/detection/mmdetection/img_info_pixel.pkl", 'rb') as f:
            imgs_info_pixel = pickle.load(f, encoding='bytes')
        with open("/media/data2/jliang/detection/mmdetection/result_pixel.pkl", 'rb') as f:
            results_pixel = pickle.load(f, encoding='bytes')
        with open("/media/data2/jliang/detection/mmdetection/img_info_newdcn.pkl", 'rb') as f:
            imgs_info_newdcn = pickle.load(f, encoding='bytes')
        with open("/media/data2/jliang/detection/mmdetection/result_newdcn.pkl", 'rb') as f:
            results_newdcn = pickle.load(f, encoding='bytes')
        with open("/media/data2/jliang/detection/mmdetection/img_info_pixel_newdcn.pkl", 'rb') as f:
            imgs_info_pixel_newdcn = pickle.load(f, encoding='bytes')
        with open("/media/data2/jliang/detection/mmdetection/result_pixel_newdcn.pkl", 'rb') as f:
            results_pixel_newdcn = pickle.load(f, encoding='bytes')
        with open(val_videos_path, 'r') as f:
            val_videos = f.readlines()

        thickness = 2
        text_thickness = 1
        font_scale = 1
        score_thr = 0.3
        name_seqs = ['val/ILSVRC2015_val_00057000']
        for name_seq in name_seqs:
            for val_video in val_videos:
                val_video = val_video.replace('\n', '')
                val_video = val_video.split(' ')
                if val_video[0] == name_seq:
                    video_b = int(val_video[1])-1
                    video_len = int(val_video[-1])
                    for i in range(video_b, video_b+video_len):
                        img_info = imgs_info_newdcn[i]
                        result = results_newdcn[i]
                        img_path = os.path.join(data_root, img_info['filename'])
                        _, seq_name, img_name = img_info['id'].split('/')
                        seq_result_path = os.path.join(result_path, seq_name, 'newdcn')
                        if not os.path.exists(seq_result_path):
                            os.makedirs(seq_result_path)
                        img = cv2.imread(img_path)
                        for i in range(len(result)):
                            res = result[i]
                            if len(res) != 0:
                                for box in res:
                                    x1, y1, x2, y2 = box[:-1].astype(np.int)
                                    score = box[-1]
                                    if score > score_thr:
                                        cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 255), thickness=thickness)  # red
                                        label_text = self.CLASSES[i] if self.CLASSES is not None else 'cls {}'.format(
                                            i - 1)
                                        label_text += '|{:.02f}'.format(float(score))
                                        (text_width, text_height) = \
                                        cv2.getTextSize(label_text, cv2.FONT_HERSHEY_COMPLEX, fontScale=font_scale,
                                                        thickness=text_thickness)[0]
                                        box_coords = ((x1, y1 - 2),
                                                      (x1 + text_width - 2, y1 - text_height - 2))
                                        cv2.rectangle(img, box_coords[0], box_coords[1], (100, 100, 255), cv2.FILLED)
                                        cv2.putText(img, label_text, (x1, y1 - 2),
                                                    cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 255, 255), thickness=text_thickness)
                        cv2.imwrite(os.path.join(seq_result_path, img_name)+'.jpg', img)

                    #pixel_newdcn
                    for i in range(video_b, video_b+video_len):
                        img_info = imgs_info_pixel_newdcn[i]
                        result = results_pixel_newdcn[i]
                        img_path = os.path.join(data_root, img_info['filename'])
                        _, seq_name, img_name = img_info['id'].split('/')
                        seq_result_path = os.path.join(result_path, seq_name, 'pixel_newdcn')
                        if not os.path.exists(seq_result_path):
                            os.makedirs(seq_result_path)
                        img = cv2.imread(img_path)
                        for i in range(len(result)):
                            res = result[i]
                            if len(res) != 0:
                                for box in res:
                                    x1, y1, x2, y2 = box[:-1].astype(np.int)
                                    score = box[-1]
                                    if score > score_thr:
                                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 100, 100), thickness=thickness)  # blue
                                        label_text = self.CLASSES[i] if self.CLASSES is not None else 'cls {}'.format(
                                            i - 1)
                                        label_text += '|{:.02f}'.format(float(score))
                                        (text_width, text_height) = \
                                        cv2.getTextSize(label_text, cv2.FONT_HERSHEY_COMPLEX, fontScale=font_scale,
                                                        thickness=text_thickness)[0]
                                        box_coords = ((x1, y1 - 2),
                                                      (x1 + text_width - 2, y1 - text_height - 2))
                                        cv2.rectangle(img, box_coords[0], box_coords[1], (255, 100, 100), cv2.FILLED)
                                        cv2.putText(img, label_text, (x1, y1 - 2),
                                                    cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 255, 255), thickness=text_thickness)
                        cv2.imwrite(os.path.join(seq_result_path, img_name)+'.jpg', img)

                    #baseline
                    for i in range(video_b, video_b + video_len):
                        img_info = imgs_info_baseline[i]
                        result = results_baseline[i]
                        img_path = os.path.join(data_root, img_info['filename'])
                        _, seq_name, img_name = img_info['id'].split('/')
                        seq_result_path = os.path.join(result_path, seq_name, 'baseline')
                        if not os.path.exists(seq_result_path):
                            os.makedirs(seq_result_path)
                        img = cv2.imread(img_path)
                        for i in range(len(result)):
                            res = result[i]
                            if len(res) != 0:
                                for box in res:
                                    x1, y1, x2, y2 = box[:-1].astype(np.int)
                                    score = box[-1]
                                    if score > score_thr:
                                        cv2.rectangle(img, (x1, y1), (x2, y2), (100, 255, 100),
                                                      thickness=thickness)  # green
                                        label_text = self.CLASSES[
                                            i] if self.CLASSES is not None else 'cls {}'.format(
                                            i - 1)
                                        label_text += '|{:.02f}'.format(float(score))
                                        (text_width, text_height) = \
                                            cv2.getTextSize(label_text, cv2.FONT_HERSHEY_COMPLEX,
                                                            fontScale=font_scale,
                                                            thickness=text_thickness)[0]
                                        box_coords = ((x1, y1 - 2),
                                                      (x1 + text_width - 2, y1 - text_height - 2))
                                        cv2.rectangle(img, box_coords[0], box_coords[1], (100, 255, 100),
                                                      cv2.FILLED)
                                        cv2.putText(img, label_text, (x1, y1 - 2),
                                                    cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 255, 255),
                                                    thickness=text_thickness)
                        cv2.imwrite(os.path.join(seq_result_path, img_name) + '.jpg', img)

                    #pixel
                    for i in range(video_b, video_b + video_len):
                        img_info = imgs_info_pixel[i]
                        result = results_pixel[i]
                        img_path = os.path.join(data_root, img_info['filename'])
                        _, seq_name, img_name = img_info['id'].split('/')
                        seq_result_path = os.path.join(result_path, seq_name, 'pixel')
                        if not os.path.exists(seq_result_path):
                            os.makedirs(seq_result_path)
                        img = cv2.imread(img_path)
                        for i in range(len(result)):
                            res = result[i]
                            if len(res) != 0:
                                for box in res:
                                    x1, y1, x2, y2 = box[:-1].astype(np.int)
                                    score = box[-1]
                                    if score > score_thr:
                                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 100, 255),
                                                      thickness=thickness)  # magenta
                                        label_text = self.CLASSES[
                                            i] if self.CLASSES is not None else 'cls {}'.format(
                                            i - 1)
                                        label_text += '|{:.02f}'.format(float(score))
                                        (text_width, text_height) = \
                                            cv2.getTextSize(label_text, cv2.FONT_HERSHEY_COMPLEX,
                                                            fontScale=font_scale,
                                                            thickness=text_thickness)[0]
                                        box_coords = ((x1, y1 - 2),
                                                      (x1 + text_width - 2, y1 - text_height - 2))
                                        cv2.rectangle(img, box_coords[0], box_coords[1], (255, 100, 255),
                                                      cv2.FILLED)
                                        cv2.putText(img, label_text, (x1, y1 - 2),
                                                    cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 255, 255),
                                                    thickness=text_thickness)
                        cv2.imwrite(os.path.join(seq_result_path, img_name) + '.jpg', img)

                    cat_result_path = os.path.join(result_path, seq_name)
                    if not os.path.exists(os.path.join(cat_result_path, 'concat')):
                        os.makedirs(os.path.join(cat_result_path, 'concat'))
                    cat_result_path_list = os.listdir(cat_result_path)
                    if 'concat' in cat_result_path_list:
                        cat_result_path_list.remove('concat')
                    imgs_list = os.listdir(os.path.join(cat_result_path, cat_result_path_list[0]))
                    for img_name in imgs_list:
                        img1 = cv2.imread(os.path.join(cat_result_path, 'baseline', img_name))
                        img2 = cv2.imread(os.path.join(cat_result_path, 'pixel', img_name))
                        img3 = cv2.imread(os.path.join(cat_result_path, 'newdcn', img_name))
                        img4 = cv2.imread(os.path.join(cat_result_path, 'pixel_newdcn', img_name))
                        cat_img = np.concatenate((np.concatenate((img1, img2), 1), np.concatenate((img3, img4), 1)), 0)
                        # for i in range(len(cat_result_path_list)):
                        #     img_path = os.path.join(cat_result_path, cat_result_path_list[i], img_name)
                        #     img = cv2.imread(img_path)
                        #     if cat_img is None:
                        #         cat_img = img
                        #     else:
                        #         cat_img = np.concatenate((cat_img, img), 1)
                        cv2.imwrite(os.path.join(cat_result_path, 'concat', img_name), cat_img)
                    break



        #                 # cv2.imwrite(result_path_concat+'/%06d.jpg'%j, np.concatenate((img1, img2, img3), 1))
        #                 cv2.imwrite(result_path_concat+'/%06d.jpg'%j, np.concatenate( (np.concatenate((img1, img3), 1), np.concatenate((img4, img5), 1)), 0) )

    def plot_specific_img(self):
        import cv2
        import pickle
        data_root = '/media/data2/jliang/dataset/ILSVRC/'
        result_path = '/media/data1/jliang/detection/mmdetection/single_result'
        val_videos_path = os.path.join(data_root, 'ImageSets/VID_val_videos.txt')
        with open("../img_info_baseline.pkl", 'rb') as f:
            imgs_info_baseline = pickle.load(f, encoding='bytes')
        with open("../result_baseline.pkl", 'rb') as f:
            results_baseline = pickle.load(f, encoding='bytes')
        with open("../img_info_pixel.pkl", 'rb') as f:
            imgs_info_pixel = pickle.load(f, encoding='bytes')
        with open("../result_pixel.pkl", 'rb') as f:
            results_pixel = pickle.load(f, encoding='bytes')
        with open("../img_info_newdcn.pkl", 'rb') as f:
            imgs_info_newdcn = pickle.load(f, encoding='bytes')
        with open("../result_newdcn.pkl", 'rb') as f:
            results_newdcn = pickle.load(f, encoding='bytes')
        with open("../img_info_pixel_newdcn.pkl", 'rb') as f:
            imgs_info_pixel_newdcn = pickle.load(f, encoding='bytes')
        with open("../result_pixel_newdcn.pkl", 'rb') as f:
            results_pixel_newdcn = pickle.load(f, encoding='bytes')
        with open(val_videos_path, 'r') as f:
            val_videos = f.readlines()

        thickness = 3
        text_thickness = 2
        font_scale = 1
        score_thr = 0.3
        name_seqs = ['val/ILSVRC2015_val_00057000']
        choice = 'all'
        specific_img_name = 104
        for name_seq in name_seqs:
            for val_video in val_videos:
                val_video = val_video.replace('\n', '')
                val_video = val_video.split(' ')
                if val_video[0] == name_seq:
                    video_b = int(val_video[1]) - 1
                    video_len = int(val_video[-1])

                    if choice == 'all':
                        for i in range(video_b, video_b + video_len):
                            img_info = imgs_info_newdcn[i]
                            result = results_newdcn[i]
                            img_path = os.path.join(data_root, img_info['filename'])
                            _, seq_name, img_name = img_info['id'].split('/')
                            if int(img_name) != specific_img_name:
                                continue
                            seq_result_path = os.path.join(result_path, seq_name, 'newdcn')
                            if not os.path.exists(seq_result_path):
                                os.makedirs(seq_result_path)
                            img = cv2.imread(img_path)
                            for i in range(len(result)):
                                res = result[i]
                                if len(res) != 0:
                                    for box in res:
                                        x1, y1, x2, y2 = box[:-1].astype(np.int)
                                        score = box[-1]
                                        if score > 0.35:
                                            cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 255),
                                                          thickness=thickness)  # red
                                            label_text = self.CLASSES[i] if self.CLASSES is not None else 'cls {}'.format(
                                                i - 1)
                                            label_text += ' {:.02f}'.format(float(score))
                                            (text_width, text_height) = \
                                                cv2.getTextSize(label_text, cv2.FONT_HERSHEY_COMPLEX, fontScale=font_scale,
                                                                thickness=text_thickness)[0]
                                            box_coords = ((x1, y1 - 2),
                                                          (x1 + text_width - 2, y1 - text_height - 2))
                                            cv2.rectangle(img, box_coords[0], box_coords[1], (100, 100, 255), cv2.FILLED)
                                            cv2.putText(img, label_text, (x1, y1 - 2),
                                                        cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 255, 255),
                                                        thickness=text_thickness)
                            cv2.imwrite(os.path.join(seq_result_path, img_name) + '.jpg', img)

                    if choice == 'all':
                        # pixel_newdcn
                        for i in range(video_b, video_b + video_len):
                            img_info = imgs_info_pixel_newdcn[i]
                            result = results_pixel_newdcn[i]
                            img_path = os.path.join(data_root, img_info['filename'])
                            _, seq_name, img_name = img_info['id'].split('/')
                            if int(img_name) != specific_img_name:
                                continue
                            seq_result_path = os.path.join(result_path, seq_name, 'pixel_newdcn')
                            if not os.path.exists(seq_result_path):
                                os.makedirs(seq_result_path)
                            img = cv2.imread(img_path)
                            for i in range(len(result)):
                                res = result[i]
                                if len(res) != 0:
                                    for box in res:
                                        x1, y1, x2, y2 = box[:-1].astype(np.int)
                                        score = box[-1]
                                        if score > score_thr:
                                            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 100, 100),
                                                          thickness=thickness)  # blue
                                            label_text = self.CLASSES[i] if self.CLASSES is not None else 'cls {}'.format(
                                                i - 1)
                                            label_text += ' {:.02f}'.format(float(score))
                                            (text_width, text_height) = \
                                                cv2.getTextSize(label_text, cv2.FONT_HERSHEY_COMPLEX, fontScale=font_scale,
                                                                thickness=text_thickness)[0]
                                            box_coords = ((x1, y1 - 2),
                                                          (x1 + text_width - 2, y1 - text_height - 2))
                                            cv2.rectangle(img, box_coords[0], box_coords[1], (255, 100, 100), cv2.FILLED)
                                            cv2.putText(img, label_text, (x1, y1 - 2),
                                                        cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 255, 255),
                                                        thickness=text_thickness)
                            cv2.imwrite(os.path.join(seq_result_path, img_name) + '.jpg', img)

                    if choice == 'all':
                        # baseline
                        for i in range(video_b, video_b + video_len):
                            img_info = imgs_info_baseline[i]
                            result = results_baseline[i]
                            img_path = os.path.join(data_root, img_info['filename'])
                            _, seq_name, img_name = img_info['id'].split('/')
                            # specific_img_name = 111
                            if int(img_name) != specific_img_name:
                                continue
                            seq_result_path = os.path.join(result_path, seq_name, 'baseline')
                            if not os.path.exists(seq_result_path):
                                os.makedirs(seq_result_path)
                            img = cv2.imread(img_path)
                            for i in range(len(result)):
                                res = result[i]
                                if len(res) != 0:
                                    for box in res:
                                        x1, y1, x2, y2 = box[:-1].astype(np.int)
                                        score = box[-1]
                                        if score > 0.3:
                                            cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 255),
                                                          thickness=thickness)  # green
                                            label_text = self.CLASSES[
                                                i] if self.CLASSES is not None else 'cls {}'.format(
                                                i - 1)
                                            label_text += ' {:.02f}'.format(float(score))
                                            (text_width, text_height) = \
                                                cv2.getTextSize(label_text, cv2.FONT_HERSHEY_COMPLEX,
                                                                fontScale=font_scale,
                                                                thickness=text_thickness)[0]
                                            box_coords = ((x1, y1 - 2),
                                                          (x1 + text_width - 2, y1 - text_height - 2))
                                            cv2.rectangle(img, box_coords[0], box_coords[1], (100, 100, 255),
                                                          cv2.FILLED)
                                            cv2.putText(img, label_text, (x1, y1 - 2),
                                                        cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 255, 255),
                                                        thickness=text_thickness)
                            cv2.imwrite(os.path.join(seq_result_path, img_name) + '.jpg', img)

                    if choice == 'all':
                        # pixel
                        for i in range(video_b, video_b + video_len):
                            img_info = imgs_info_pixel[i]
                            result = results_pixel[i]
                            img_path = os.path.join(data_root, img_info['filename'])
                            _, seq_name, img_name = img_info['id'].split('/')
                            # specific_img_name = 45
                            if int(img_name) != specific_img_name:
                                continue
                            seq_result_path = os.path.join(result_path, seq_name, 'pixel')
                            if not os.path.exists(seq_result_path):
                                os.makedirs(seq_result_path)
                            img = cv2.imread(img_path)
                            for i in range(len(result)):
                                res = result[i]
                                if len(res) != 0:
                                    for box in res:
                                        x1, y1, x2, y2 = box[:-1].astype(np.int)
                                        score = box[-1]
                                        if score > 0.3:
                                            cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 255),
                                                          thickness=thickness)  # magenta
                                            label_text = self.CLASSES[
                                                i] if self.CLASSES is not None else 'cls {}'.format(
                                                i - 1)
                                            label_text += ' {:.02f}'.format(float(score))
                                            (text_width, text_height) = \
                                                cv2.getTextSize(label_text, cv2.FONT_HERSHEY_COMPLEX,
                                                                fontScale=font_scale,
                                                                thickness=text_thickness)[0]
                                            box_coords = ((x1, y1 - 2),
                                                          (x1 + text_width - 2, y1 - text_height - 2))
                                            cv2.rectangle(img, box_coords[0], box_coords[1], (100, 100, 255),
                                                          cv2.FILLED)
                                            cv2.putText(img, label_text, (x1, y1 - 2),
                                                        cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 255, 255),
                                                        thickness=text_thickness)
                            cv2.imwrite(os.path.join(seq_result_path, img_name) + '.jpg', img)
                    break

    def show_result_2th(self):
        import cv2
        import pickle
        data_root = '/media/data1/jliang/dataset/ILSVRC/'
        result_path = '/media/data2/jliang/detection/mmdetection/result_2th'
        val_videos_path = os.path.join(data_root, 'ImageSets/VID_val_videos.txt')
        with open("/media/data2/jliang/detection/mmdetection/img_info_baseline.pkl", 'rb') as f:
            imgs_info_baseline = pickle.load(f, encoding='bytes')
        with open("/media/data2/jliang/detection/mmdetection/result_baseline.pkl", 'rb') as f:
            results_baseline = pickle.load(f, encoding='bytes')
        with open("/media/data2/jliang/detection/mmdetection/img_info_frame.pkl", 'rb') as f:
            imgs_info_frame = pickle.load(f, encoding='bytes')
        with open("/media/data2/jliang/detection/mmdetection/result_frame.pkl", 'rb') as f:
            results_frame = pickle.load(f, encoding='bytes')
        with open("/media/data2/jliang/detection/mmdetection/img_info_proposal.pkl", 'rb') as f:
            imgs_info_proposal = pickle.load(f, encoding='bytes')
        with open("/media/data2/jliang/detection/mmdetection/result_proposal.pkl", 'rb') as f:
            results_proposal = pickle.load(f, encoding='bytes')
        with open("/media/data2/jliang/detection/mmdetection/img_info_frame_proposal.pkl", 'rb') as f:
            imgs_info_frame_proposal = pickle.load(f, encoding='bytes')
        with open("/media/data2/jliang/detection/mmdetection/result_frame_proposal.pkl", 'rb') as f:
            results_frame_proposal = pickle.load(f, encoding='bytes')
        with open(val_videos_path, 'r') as f:
            val_videos = f.readlines()

        thickness = 2
        text_thickness = 1
        font_scale = 1
        score_thr = 0.3
        name_seqs = ['val/ILSVRC2015_val_00010000','val/ILSVRC2015_val_00010001','val/ILSVRC2015_val_00010002','val/ILSVRC2015_val_00010003']
        for name_seq in name_seqs:
            for val_video in val_videos:
                val_video = val_video.replace('\n', '')
                val_video = val_video.split(' ')
                if val_video[0] == name_seq:
                    video_b = int(val_video[1])-1
                    video_len = int(val_video[-1])
                    for i in range(video_b, video_b+video_len):
                        img_info = imgs_info_proposal[i]
                        result = results_proposal[i]
                        img_path = os.path.join(data_root, img_info['filename'])
                        _, seq_name, img_name = img_info['id'].split('/')
                        seq_result_path = os.path.join(result_path, seq_name, 'proposal')
                        if not os.path.exists(seq_result_path):
                            os.makedirs(seq_result_path)
                        img = cv2.imread(img_path)
                        for i in range(len(result)):
                            res = result[i]
                            if len(res) != 0:
                                for box in res:
                                    x1, y1, x2, y2 = box[:-1].astype(np.int)
                                    score = box[-1]
                                    if score > score_thr:
                                        cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 255), thickness=thickness)  # red
                                        label_text = self.CLASSES[i] if self.CLASSES is not None else 'cls {}'.format(
                                            i - 1)
                                        label_text += '|{:.02f}'.format(float(score))
                                        (text_width, text_height) = \
                                        cv2.getTextSize(label_text, cv2.FONT_HERSHEY_COMPLEX, fontScale=font_scale,
                                                        thickness=text_thickness)[0]
                                        box_coords = ((x1, y1 - 2),
                                                      (x1 + text_width - 2, y1 - text_height - 2))
                                        cv2.rectangle(img, box_coords[0], box_coords[1], (100, 100, 255), cv2.FILLED)
                                        cv2.putText(img, label_text, (x1, y1 - 2),
                                                    cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 255, 255), thickness=text_thickness)
                        cv2.imwrite(os.path.join(seq_result_path, img_name)+'.jpg', img)

                    #frame_proposal
                    for i in range(video_b, video_b+video_len):
                        img_info = imgs_info_frame_proposal[i]
                        result = results_frame_proposal[i]
                        img_path = os.path.join(data_root, img_info['filename'])
                        _, seq_name, img_name = img_info['id'].split('/')
                        seq_result_path = os.path.join(result_path, seq_name, 'frame_proposal')
                        if not os.path.exists(seq_result_path):
                            os.makedirs(seq_result_path)
                        img = cv2.imread(img_path)
                        for i in range(len(result)):
                            res = result[i]
                            if len(res) != 0:
                                for box in res:
                                    x1, y1, x2, y2 = box[:-1].astype(np.int)
                                    score = box[-1]
                                    if score > score_thr:
                                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 100, 100), thickness=thickness)  # blue
                                        label_text = self.CLASSES[i] if self.CLASSES is not None else 'cls {}'.format(
                                            i - 1)
                                        label_text += '|{:.02f}'.format(float(score))
                                        (text_width, text_height) = \
                                        cv2.getTextSize(label_text, cv2.FONT_HERSHEY_COMPLEX, fontScale=font_scale,
                                                        thickness=text_thickness)[0]
                                        box_coords = ((x1, y1 - 2),
                                                      (x1 + text_width - 2, y1 - text_height - 2))
                                        cv2.rectangle(img, box_coords[0], box_coords[1], (255, 100, 100), cv2.FILLED)
                                        cv2.putText(img, label_text, (x1, y1 - 2),
                                                    cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 255, 255), thickness=text_thickness)
                        cv2.imwrite(os.path.join(seq_result_path, img_name)+'.jpg', img)

                    #baseline
                    for i in range(video_b, video_b + video_len):
                        img_info = imgs_info_baseline[i]
                        result = results_baseline[i]
                        img_path = os.path.join(data_root, img_info['filename'])
                        _, seq_name, img_name = img_info['id'].split('/')
                        seq_result_path = os.path.join(result_path, seq_name, 'baseline')
                        if not os.path.exists(seq_result_path):
                            os.makedirs(seq_result_path)
                        img = cv2.imread(img_path)
                        for i in range(len(result)):
                            res = result[i]
                            if len(res) != 0:
                                for box in res:
                                    x1, y1, x2, y2 = box[:-1].astype(np.int)
                                    score = box[-1]
                                    if score > score_thr:
                                        cv2.rectangle(img, (x1, y1), (x2, y2), (100, 255, 100),
                                                      thickness=thickness)  # green
                                        label_text = self.CLASSES[
                                            i] if self.CLASSES is not None else 'cls {}'.format(
                                            i - 1)
                                        label_text += '|{:.02f}'.format(float(score))
                                        (text_width, text_height) = \
                                            cv2.getTextSize(label_text, cv2.FONT_HERSHEY_COMPLEX,
                                                            fontScale=font_scale,
                                                            thickness=text_thickness)[0]
                                        box_coords = ((x1, y1 - 2),
                                                      (x1 + text_width - 2, y1 - text_height - 2))
                                        cv2.rectangle(img, box_coords[0], box_coords[1], (100, 255, 100),
                                                      cv2.FILLED)
                                        cv2.putText(img, label_text, (x1, y1 - 2),
                                                    cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 255, 255),
                                                    thickness=text_thickness)
                        cv2.imwrite(os.path.join(seq_result_path, img_name) + '.jpg', img)

                    #frame
                    for i in range(video_b, video_b + video_len):
                        img_info = imgs_info_frame[i]
                        result = results_frame[i]
                        img_path = os.path.join(data_root, img_info['filename'])
                        _, seq_name, img_name = img_info['id'].split('/')
                        seq_result_path = os.path.join(result_path, seq_name, 'frame')
                        if not os.path.exists(seq_result_path):
                            os.makedirs(seq_result_path)
                        img = cv2.imread(img_path)
                        for i in range(len(result)):
                            res = result[i]
                            if len(res) != 0:
                                for box in res:
                                    x1, y1, x2, y2 = box[:-1].astype(np.int)
                                    score = box[-1]
                                    if score > score_thr:
                                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 100, 255),
                                                      thickness=thickness)  # magenta
                                        label_text = self.CLASSES[
                                            i] if self.CLASSES is not None else 'cls {}'.format(
                                            i - 1)
                                        label_text += '|{:.02f}'.format(float(score))
                                        (text_width, text_height) = \
                                            cv2.getTextSize(label_text, cv2.FONT_HERSHEY_COMPLEX,
                                                            fontScale=font_scale,
                                                            thickness=text_thickness)[0]
                                        box_coords = ((x1, y1 - 2),
                                                      (x1 + text_width - 2, y1 - text_height - 2))
                                        cv2.rectangle(img, box_coords[0], box_coords[1], (255, 100, 255),
                                                      cv2.FILLED)
                                        cv2.putText(img, label_text, (x1, y1 - 2),
                                                    cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 255, 255),
                                                    thickness=text_thickness)
                        cv2.imwrite(os.path.join(seq_result_path, img_name) + '.jpg', img)

                    cat_result_path = os.path.join(result_path, seq_name)
                    if not os.path.exists(os.path.join(cat_result_path, 'concat')):
                        os.makedirs(os.path.join(cat_result_path, 'concat'))
                    cat_result_path_list = os.listdir(cat_result_path)
                    if 'concat' in cat_result_path_list:
                        cat_result_path_list.remove('concat')
                    imgs_list = os.listdir(os.path.join(cat_result_path, cat_result_path_list[0]))
                    for img_name in imgs_list:
                        img1 = cv2.imread(os.path.join(cat_result_path, 'baseline', img_name))
                        img2 = cv2.imread(os.path.join(cat_result_path, 'frame', img_name))
                        img3 = cv2.imread(os.path.join(cat_result_path, 'proposal', img_name))
                        img4 = cv2.imread(os.path.join(cat_result_path, 'frame_proposal', img_name))
                        cat_img = np.concatenate((np.concatenate((img1, img2), 1), np.concatenate((img3, img4), 1)), 0)
                        # for i in range(len(cat_result_path_list)):
                        #     img_path = os.path.join(cat_result_path, cat_result_path_list[i], img_name)
                        #     img = cv2.imread(img_path)
                        #     if cat_img is None:
                        #         cat_img = img
                        #     else:
                        #         cat_img = np.concatenate((cat_img, img), 1)
                        cv2.imwrite(os.path.join(cat_result_path, 'concat', img_name), cat_img)
                    break



        #                 # cv2.imwrite(result_path_concat+'/%06d.jpg'%j, np.concatenate((img1, img2, img3), 1))
        #                 cv2.imwrite(result_path_concat+'/%06d.jpg'%j, np.concatenate( (np.concatenate((img1, img3), 1), np.concatenate((img4, img5), 1)), 0) )

    def plot_specific_img_2th(self):
        import cv2
        import pickle
        data_root = '/media/data1/jliang/dataset/ILSVRC/'
        result_path = '/media/data2/jliang/detection/mmdetection/single_result_2th'
        val_videos_path = os.path.join(data_root, 'ImageSets/VID_val_videos.txt')
        with open("/media/data2/jliang/detection/mmdetection/img_info_baseline.pkl", 'rb') as f:
            imgs_info_baseline = pickle.load(f, encoding='bytes')
        with open("/media/data2/jliang/detection/mmdetection/result_baseline.pkl", 'rb') as f:
            results_baseline = pickle.load(f, encoding='bytes')
        with open("/media/data2/jliang/detection/mmdetection/img_info_frame.pkl", 'rb') as f:
            imgs_info_frame = pickle.load(f, encoding='bytes')
        with open("/media/data2/jliang/detection/mmdetection/result_frame.pkl", 'rb') as f:
            results_frame = pickle.load(f, encoding='bytes')
        with open("/media/data2/jliang/detection/mmdetection/img_info_proposal.pkl", 'rb') as f:
            imgs_info_proposal = pickle.load(f, encoding='bytes')
        with open("/media/data2/jliang/detection/mmdetection/result_proposal.pkl", 'rb') as f:
            results_proposal = pickle.load(f, encoding='bytes')
        with open("/media/data2/jliang/detection/mmdetection/img_info_frame_proposal.pkl", 'rb') as f:
            imgs_info_frame_proposal = pickle.load(f, encoding='bytes')
        with open("/media/data2/jliang/detection/mmdetection/result_frame_proposal.pkl", 'rb') as f:
            results_frame_proposal = pickle.load(f, encoding='bytes')
        with open(val_videos_path, 'r') as f:
            val_videos = f.readlines()

        thickness = 3
        text_thickness = 2
        font_scale = 1
        score_thr = 0.3
        name_seqs = ['val/ILSVRC2015_val_00057000']
        choice = 'all'
        specific_img_name = 118
        for name_seq in name_seqs:
            for val_video in val_videos:
                val_video = val_video.replace('\n', '')
                val_video = val_video.split(' ')
                if val_video[0] == name_seq:
                    video_b = int(val_video[1]) - 1
                    video_len = int(val_video[-1])

                    if choice == 'all':
                        for i in range(video_b, video_b + video_len):
                            img_info = imgs_info_proposal[i]
                            result = results_proposal[i]
                            img_path = os.path.join(data_root, img_info['filename'])
                            _, seq_name, img_name = img_info['id'].split('/')
                            if int(img_name) != specific_img_name:
                                continue
                            seq_result_path = os.path.join(result_path, seq_name, 'proposal')
                            if not os.path.exists(seq_result_path):
                                os.makedirs(seq_result_path)
                            img = cv2.imread(img_path)
                            for i in range(len(result)):
                                res = result[i]
                                if len(res) != 0:
                                    for box in res:
                                        x1, y1, x2, y2 = box[:-1].astype(np.int)
                                        score = box[-1]
                                        if score > 0.3:
                                            label_text = self.CLASSES[i] if self.CLASSES is not None else 'cls {}'.format(
                                                i - 1)
                                            if label_text == 'lizard':continue
                                            cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 255),
                                                          thickness=thickness)  # red
                                            label_text += ' {:.02f}'.format(float(score))
                                            (text_width, text_height) = \
                                                cv2.getTextSize(label_text, cv2.FONT_HERSHEY_COMPLEX, fontScale=font_scale,
                                                                thickness=text_thickness)[0]
                                            box_coords = ((x1, y1 - 2),
                                                          (x1 + text_width - 2, y1 - text_height - 2))
                                            cv2.rectangle(img, box_coords[0], box_coords[1], (100, 100, 255), cv2.FILLED)
                                            cv2.putText(img, label_text, (x1, y1 - 2),
                                                        cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 255, 255),
                                                        thickness=text_thickness)
                            cv2.imwrite(os.path.join(seq_result_path, img_name) + '.jpg', img)

                    if choice == 'all':
                        # frame_proposal
                        for i in range(video_b, video_b + video_len):
                            img_info = imgs_info_frame_proposal[i]
                            result = results_frame_proposal[i]
                            img_path = os.path.join(data_root, img_info['filename'])
                            _, seq_name, img_name = img_info['id'].split('/')
                            if int(img_name) != specific_img_name:
                                continue
                            seq_result_path = os.path.join(result_path, seq_name, 'frame_proposal')
                            if not os.path.exists(seq_result_path):
                                os.makedirs(seq_result_path)
                            img = cv2.imread(img_path)
                            for i in range(len(result)):
                                res = result[i]
                                if len(res) != 0:
                                    for box in res:
                                        x1, y1, x2, y2 = box[:-1].astype(np.int)
                                        score = box[-1]
                                        if score > 0.5:
                                            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 100, 100),
                                                          thickness=thickness)  # blue
                                            label_text = self.CLASSES[i] if self.CLASSES is not None else 'cls {}'.format(
                                                i - 1)
                                            label_text += ' {:.02f}'.format(float(score))
                                            (text_width, text_height) = \
                                                cv2.getTextSize(label_text, cv2.FONT_HERSHEY_COMPLEX, fontScale=font_scale,
                                                                thickness=text_thickness)[0]
                                            box_coords = ((x1, y1 - 2),
                                                          (x1 + text_width - 2, y1 - text_height - 2))
                                            cv2.rectangle(img, box_coords[0], box_coords[1], (255, 100, 100), cv2.FILLED)
                                            cv2.putText(img, label_text, (x1, y1 - 2),
                                                        cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 255, 255),
                                                        thickness=text_thickness)
                            cv2.imwrite(os.path.join(seq_result_path, img_name) + '.jpg', img)

                    if choice == 'all':
                        # baseline
                        for i in range(video_b, video_b + video_len):
                            img_info = imgs_info_baseline[i]
                            result = results_baseline[i]
                            img_path = os.path.join(data_root, img_info['filename'])
                            _, seq_name, img_name = img_info['id'].split('/')
                            # specific_img_name = 111
                            if int(img_name) != specific_img_name:
                                continue
                            seq_result_path = os.path.join(result_path, seq_name, 'baseline')
                            if not os.path.exists(seq_result_path):
                                os.makedirs(seq_result_path)
                            img = cv2.imread(img_path)
                            for i in range(len(result)):
                                res = result[i]
                                if len(res) != 0:
                                    for box in res:
                                        x1, y1, x2, y2 = box[:-1].astype(np.int)
                                        score = box[-1]
                                        if score > 0.60:
                                            label_text = self.CLASSES[
                                                i] if self.CLASSES is not None else 'cls {}'.format(
                                                i - 1)
                                            if label_text == 'lizard':continue
                                            cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 255),
                                                          thickness=thickness)  # green
                                            label_text += ' {:.02f}'.format(float(score))
                                            (text_width, text_height) = \
                                                cv2.getTextSize(label_text, cv2.FONT_HERSHEY_COMPLEX,
                                                                fontScale=font_scale,
                                                                thickness=text_thickness)[0]
                                            box_coords = ((x1, y1 - 2),
                                                          (x1 + text_width - 2, y1 - text_height - 2))
                                            cv2.rectangle(img, box_coords[0], box_coords[1], (100, 100, 255),
                                                          cv2.FILLED)
                                            cv2.putText(img, label_text, (x1, y1 - 2),
                                                        cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 255, 255),
                                                        thickness=text_thickness)
                            cv2.imwrite(os.path.join(seq_result_path, img_name) + '.jpg', img)

                    if choice == 'all':
                        # frame
                        for i in range(video_b, video_b + video_len):
                            img_info = imgs_info_frame[i]
                            result = results_frame[i]
                            img_path = os.path.join(data_root, img_info['filename'])
                            _, seq_name, img_name = img_info['id'].split('/')
                            # specific_img_name = 60
                            if int(img_name) != specific_img_name:
                                continue
                            seq_result_path = os.path.join(result_path, seq_name, 'frame')
                            if not os.path.exists(seq_result_path):
                                os.makedirs(seq_result_path)
                            img = cv2.imread(img_path)
                            for i in range(len(result)):
                                res = result[i]
                                if len(res) != 0:
                                    for box in res:
                                        x1, y1, x2, y2 = box[:-1].astype(np.int)
                                        score = box[-1]
                                        if score > 0.3:
                                            label_text = self.CLASSES[
                                                i] if self.CLASSES is not None else 'cls {}'.format(
                                                i - 1)
                                            if label_text == 'lizard':continue
                                            cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 255),
                                                          thickness=thickness)  # magenta
                                            label_text += ' {:.02f}'.format(float(score))
                                            (text_width, text_height) = \
                                                cv2.getTextSize(label_text, cv2.FONT_HERSHEY_COMPLEX,
                                                                fontScale=font_scale,
                                                                thickness=text_thickness)[0]
                                            box_coords = ((x1, y1 - 2),
                                                          (x1 + text_width - 2, y1 - text_height - 2))
                                            cv2.rectangle(img, box_coords[0], box_coords[1], (100, 100, 255),
                                                          cv2.FILLED)
                                            cv2.putText(img, label_text, (x1, y1 - 2),
                                                        cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 255, 255),
                                                        thickness=text_thickness)
                            cv2.imwrite(os.path.join(seq_result_path, img_name) + '.jpg', img)
                    break

    def plot_video_gt(self):
        videoSeqs = ['val/ILSVRC2015_val_00010000']
        self.img_infos = vid.load_annotations()
        import numpy as np
        import cv2
        img_infos = []
        last_seg_len = None
        video_shuffle = False
        for i in range(len(self.img_infos)):
            if self.img_infos[i]['id'][:-7] != videoSeqs[0]:
                continue
            img_info = self.img_infos[i].copy()
            cur_seg_len = img_info['frame_seg_len']
            last_seg_len = 0 if last_seg_len is None else cur_seg_len
            video_index = np.arange(cur_seg_len)
            if video_shuffle:
                np.random.shuffle(video_index)
            for cur_frameid in range(len(video_index)):
                filename = '%s/%06d' % (img_info['filename'][:img_info['filename'].rfind('/')], video_index[cur_frameid]) + '.JPEG'
                id = '%s/%06d' % (img_info['id'][:img_info['id'].rfind('/')], video_index[cur_frameid])
                frame_id = video_index[cur_frameid] + last_seg_len

                img_info['filename'] = filename
                img_info['id'] = id
                img_info['frame_id'] = frame_id

                img_infos.append(img_info.copy())

        self.img_infos = img_infos
        annotations = [self.get_ann_info(i) for i in range(len(self.img_infos))]
        data_root = '/media/data1/jliang/dataset/ILSVRC/'
        result_path = '/media/data2/jliang/detection/mmdetection/video_gt'
        thickness = 2
        text_thickness = 1
        font_scale = 1
        for i in range(len(self.img_infos)):
            img_path = os.path.join(data_root, self.img_infos[i]['filename'])
            _, seq_name, img_name = self.img_infos[i]['id'].split('/')
            seq_result_path = os.path.join(result_path, seq_name)
            if not os.path.exists(seq_result_path):
                os.makedirs(seq_result_path)
            img = cv2.imread(img_path)
            res = annotations[i]
            boxes = res['bboxes']
            labels = res['labels']
            if len(boxes) != 0:
                for i in range(len(boxes)):
                    box = boxes[i]
                    label = labels[i]
                    x1, y1, x2, y2 = box.astype(np.int)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (100, 255, 100),
                                    thickness=thickness)  # green
                    label_text = self.CLASSES[label-1] if self.CLASSES is not None else 'cls {}'.format(i - 1)
                    (text_width, text_height) = \
                        cv2.getTextSize(label_text, cv2.FONT_HERSHEY_COMPLEX,
                                        fontScale=font_scale,
                                        thickness=text_thickness)[0]
                    box_coords = ((x1, y1 - 2),
                                    (x1 + text_width - 2, y1 - text_height - 2))
                    cv2.rectangle(img, box_coords[0], box_coords[1], (100, 255, 100),
                                    cv2.FILLED)
                    cv2.putText(img, label_text, (x1, y1 - 2),
                                cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 255, 255),
                                thickness=text_thickness)
            cv2.imwrite(os.path.join(seq_result_path, img_name) + '.jpg', img)


if __name__ == '__main__':
    vid = VIDDataset()
    # vid.img_infos = vid.load_annotations()
    # vid.analyze_valset()
    # vid.show_result_2th()
    vid.plot_video_gt()
    # vid.plot_specific_img_2th()
    # vid.generate_shuffle_val_frames()
    # class_dict = {}
    # for i in range(len(vid.img_infos)):
    #     ann = vid.get_ann_info(i)
    #     if i > 14945:
    #         break
    #     labels = ann['labels']
    #     for label in labels:
    #         class_dict[vid.CLASSES[label-1]] = class_dict.get(vid.CLASSES[label-1], 0) + 1
    # print(class_dict)

