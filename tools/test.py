import argparse
import os
import os.path as osp
import pickle
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmdet.core import coco_eval, results2json, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from tools.voc_eval import voc_eval

def single_gpu_test_frames(model, data_loader, show=False):
    model.eval()
    dataset = data_loader.dataset
    cur_frameid = 0
    results = []
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            if cur_frameid == 0:  # new video
                key_frame_flag = 0
                cur_seg_len = None
                model.module.data_list.clear()
                model.module.feat_list.clear()
                model.module.rois_list.clear()
            else:  # normal frame
                key_frame_flag = 2
            cur_frameid += 1
            if cur_frameid == cur_seg_len:
                cur_frameid = 0
                key_frame_flag = 1
            model.module.get_roi_feats = True
            if key_frame_flag == 0:
                feat, img_meta, rois = model(return_loss=False, rescale=not show, **data)
                cur_seg_len = img_meta[0]['img_info']['frame_seg_len']
                while len(model.module.data_list) < dataset.KEY_FRAME_INTERVAL * dataset.sample_stride + 1:
                    model.module.data_list.append(img_meta)
                    model.module.feat_list.append(feat)
                    model.module.rois_list.append(rois)
            elif key_frame_flag == 2:
                if len(model.module.data_list) < dataset.all_frame_interval - 1:
                    feat, img_meta, rois = model(return_loss=False, rescale=not show, **data)
                    model.module.data_list.append(img_meta)
                    model.module.feat_list.append(feat)
                    model.module.rois_list.append(rois)
                else:
                    feat, img_meta, rois = model(return_loss=False, rescale=not show, **data)
                    model.module.data_list.append(img_meta)
                    model.module.feat_list.append(feat)
                    model.module.rois_list.append(rois)

                    model.module.get_roi_feats = False
                    result = model(return_loss=False, rescale=not show, **data)
                    results.append(result)

                    if show:
                        model.module.show_result(data, result)

                    batch_size = data['img'][0].size(0)
                    for _ in range(batch_size):
                        prog_bar.update()
            elif key_frame_flag == 1:       # last frame of a video
                end_counter = 0
                feat, img_meta, rois = model(return_loss=False, rescale=not show, **data)

                roidb_frame_seg_len = cur_seg_len
                while len(model.module.data_list) < dataset.all_frame_interval - 1:
                    model.module.data_list.append(img_meta)
                    model.module.feat_list.append(feat)
                    model.module.rois_list.append(rois)

                model.module.get_roi_feats = False
                while end_counter < min(roidb_frame_seg_len, dataset.KEY_FRAME_INTERVAL * dataset.sample_stride + 1):
                    model.module.data_list.append(img_meta)
                    model.module.feat_list.append(feat)
                    model.module.rois_list.append(rois)
                    result = model(return_loss=False, rescale=not show, **data)
                    results.append(result)
                    end_counter += 1
                    if show:
                        model.module.show_result(data, result)

                    batch_size = data['img'][0].size(0)
                    for _ in range(batch_size):
                        prog_bar.update()
    return results

def single_gpu_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results

def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results

#init dataset and model
def init_for_test(dataset, model):
    import numpy as np
    from collections import deque
    img_infos = []
    last_seg_len = None
    video_shuffle = True
    for i in range(len(dataset)):
        img_info = dataset.img_infos[i].copy()
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

    # with open("img_info_frame.pkl", 'wb') as f:
    #     pickle.dump(img_infos, f)
    dataset.img_infos = img_infos
    dataset.KEY_FRAME_INTERVAL = 10
    dataset.sample_stride = 1
    dataset.all_frame_interval = dataset.KEY_FRAME_INTERVAL * dataset.sample_stride * 2 + 1

    model.data_list = deque(maxlen=dataset.all_frame_interval)
    model.feat_list = deque(maxlen=dataset.all_frame_interval)
    model.rois_list = deque(maxlen=dataset.all_frame_interval)
    # model.layer_list = deque(maxlen=dataset.all_frame_interval)
    model.get_roi_feats = True

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--json_out',
        help='output result file name without extension',
        type=str)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():
    args = parse_args()

    assert args.out or args.show or args.json_out, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out" or "--show" or "--json_out"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if args.json_out is not None and args.json_out.endswith('.json'):
        args.json_out = args.json_out[:-5]

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    dataset.test_frames = True
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        # workers_per_gpu=cfg.data.workers_per_gpu,
        workers_per_gpu=5,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        if dataset.test_frames:
            init_for_test(dataset, model)
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test_frames(model, data_loader, args.show)
        else:
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test(model, data_loader, args.show)
    else:
        model = MMDistributedDataParallel(model.cuda())
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if args.out and rank == 0:
        print('\nwriting results to {}'.format(args.out))
        mmcv.dump(outputs, args.out)
        eval_types = args.eval
        if eval_types:
            print('Starting evaluate {}'.format(' and '.join(eval_types)))
            if eval_types == ['proposal_fast']:
                result_file = args.out
                coco_eval(result_file, eval_types, dataset.coco)
            else:
                if not isinstance(outputs[0], dict):
                    voc_eval(args.out, dataset, 0.5)
                    # result_files = results2json(dataset, outputs, args.out)
                    # coco_eval(result_files, eval_types, dataset.coco)
                else:
                    for name in outputs[0]:
                        print('\nEvaluating {}'.format(name))
                        outputs_ = [out[name] for out in outputs]
                        result_file = args.out + '.{}'.format(name)
                        result_files = results2json(dataset, outputs_,
                                                    result_file)
                        coco_eval(result_files, eval_types, dataset.coco)

    # Save predictions in the COCO json format
    # if args.json_out and rank == 0:
    #     if not isinstance(outputs[0], dict):
    #         results2json(dataset, outputs, args.json_out)
    #     else:
    #         for name in outputs[0]:
    #             outputs_ = [out[name] for out in outputs]
    #             result_file = args.json_out + '.{}'.format(name)
    #             results2json(dataset, outputs_, result_file)


if __name__ == '__main__':
    main()
