import time
import argparse
import os
import os.path as osp

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmpose.apis import multi_gpu_test, single_gpu_test
from mmpose.core import wrap_fp16_model
from mmpose.datasets import build_dataloader, build_dataset

import sys
sys.path.append('tools')
from models import build_posenet


def parse_args():
    parser = argparse.ArgumentParser(description='mmpose test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--eval',
        default=None,
        nargs='+',
        help='evaluation metric, which depends on the dataset,'
        ' e.g., "mAP" for MSCOCO')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
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

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None

    # 初始化模型
    model = build_posenet(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()
    model = model.cuda()

    # 创建随机数据
    input_size = (3, 256, 192)  # 根据模型需要调整
    random_input = torch.randn(1, *input_size).cuda()

    # 创建占位符
    num_joints = 17  # 假设关键点数量为17，根据模型调整
    heatmap_size = (1, num_joints, input_size[1] // 4, input_size[2] // 4)  # 假设输出热图尺寸是输入的1/4
    target = torch.zeros(heatmap_size).cuda()
    target_weight = torch.ones((1, num_joints, 1)).cuda()

    # GPU预热
    for _ in range(100):
        _ = model(random_input, target, target_weight)

    # 测量推理时间
    iterations = 900
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            start_time1 = time.time()
            _ = model(random_input, target, target_weight)
            print(time.time() - start_time1)

    elapsed_time = time.time() - start_time
    avg_time_per_iteration = elapsed_time / iterations
    print(f"Average inference time per iteration (over 9000 iterations): {avg_time_per_iteration:.3f} seconds")

if __name__ == '__main__':
    main()