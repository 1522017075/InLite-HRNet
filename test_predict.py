import json
import argparse
import os
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

import sys
sys.path.append('tools')
from models import build_posenet


import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from draw_utils import draw_keypoints

import draw_3D_pose

def parse_args():
    parser = argparse.ArgumentParser(description='mmpose test model')
    parser.add_argument('--config',default='configs/top_down/dite_hrnet/coco/inlitehrnetv1_18_coco_256x192.py', help='test config file path')
    parser.add_argument('--checkpoint', default='weights/inlitehrnetv1_18_coco_256x192.pth', help='checkpoint file')
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


def predict_single_person():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")


    # 预测图片时是否进行翻转（水平翻转前后两次的结果进行综合判断）
    flip_test = False
    # 输入图片缩放到这个高宽
    resize_hw = (256, 192)
    # img_path = "./person.png"
    img_path = "./pic/person1.png"
    keypoint_json_path = "person_keypoints.json"
    assert os.path.exists(img_path), f"file: {img_path} does not exist."
    assert os.path.exists(keypoint_json_path), f"file: {keypoint_json_path} does not exist."

    data_transform = transforms.Compose([
        transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=resize_hw),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # read json file
    with open(keypoint_json_path, "r") as f:
        person_info = json.load(f)

    # read single-person image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 记录原始高宽，需要还原回去
    img_tensor, target = data_transform(img, {"box": [0, 0, img.shape[1] - 1, img.shape[0] - 1]})
    # 增加一个batch维度
    img_tensor = torch.unsqueeze(img_tensor, dim=0)

    # create model
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

    # 创建占位符
    num_joints = 17  # 假设关键点数量为17，根据模型调整
    heatmap_size = (1, num_joints, input_size[1] // 4, input_size[2] // 4)  # 假设输出热图尺寸是输入的1/4
    target_heatmap = torch.zeros(heatmap_size).cuda()
    target_weight = torch.ones((1, num_joints, 1)).cuda()

    with torch.inference_mode():
        # outputs = model(img_tensor.to(device))
        outputs = model(img_tensor.to(device), target_heatmap, target_weight)

        # draw_3D_pose.draw_3d(outputs)

        if flip_test:
            flip_tensor = transforms.flip_images(img_tensor)
            flip_outputs = torch.squeeze(
                transforms.flip_back(model(flip_tensor.to(device), target_heatmap, target_weight), person_info["flip_pairs"]),
            )
            # feature is not aligned, shift flipped heatmap for higher accuracy
            # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
            flip_outputs[..., 1:] = flip_outputs.clone()[..., 0: -1]
            outputs = (outputs + flip_outputs) * 0.5

        keypoints, scores = transforms.get_final_preds(outputs, [target["reverse_trans"]], True)
        keypoints = np.squeeze(keypoints)
        scores = np.squeeze(scores)

        plot_img = draw_keypoints(img, keypoints, scores, thresh=0.2, r=3)
        plt.imshow(plot_img)
        plt.show()
        plot_img.save("./result/test_result1.jpg")


if __name__ == '__main__':
    predict_single_person()