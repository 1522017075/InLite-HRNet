import json
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


import transforms
import cv2
import matplotlib.pyplot as plt
from draw_utils import draw_keypoints
from draw_link_tool import imshow_keypoints

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


def predict_single_person():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")


    # 预测图片时是否进行翻转（水平翻转前后两次的结果进行综合判断）
    flip_test = True
    # 输入图片缩放到这个高宽
    resize_hw = (256, 192)
    img_path = "./person.png"
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

    with torch.inference_mode():
        outputs = model(img_tensor.to(device))

        if flip_test:
            flip_tensor = transforms.flip_images(img_tensor)
            flip_outputs = torch.squeeze(
                transforms.flip_back(model(flip_tensor.to(device)), person_info["flip_pairs"]),
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


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

counter = 0
fps = 0
start_time = time.time()
keypoint_json_path = "person_keypoints.json"
with open(keypoint_json_path, "r") as f:
    person_info = json.load(f)

# 使用while循环不断读取摄像头的图像帧
def predict_mp4(flip_test=False):
    global counter, start_time, fps, person_info
    # 处理成可以输入的256*192
    resize_hw = (256, 192)
    data_transform = transforms.Compose([
        transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=resize_hw),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # model
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
    target = torch.zeros(heatmap_size).cuda()
    target_weight = torch.ones((1, num_joints, 1)).cuda()

    while True:
        # 读取一帧图像，ret是一个布尔值，表示是否成功读取，frame是一个图像矩阵
        ret, frame = cap.read()
        if ret :
            # 使用yolov5模型对图像进行预测，返回一个结果对象
            # results = model(frame)
            # 使用results.print()函数打印检测结果
            # results.print()
            # # 使用results.render()函数在图像上绘制检测框和标签
            # results.render()
            # 使用cv2.imshow()函数显示处理后的图像
            # 可能寄掉
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 记录原始高宽，需要还原回去
            frame, target = data_transform(img, {"box": [0, 0, img.shape[1] - 1, img.shape[0] - 1]})
            # 增加一个batch维度
            frame = torch.unsqueeze(frame, dim=0)

            # 使用模型得到输出结果
            with torch.inference_mode():
                outputs = model(frame.cuda(), target, target_weight)
                # outputs = model(frame.to(device))

                # 2D转3D
                # keypoints_2d形状为 (17, height, width)
                # keypoints_2d = extract_keypoints_from_heatmaps(outputs)
                # keypoints_3d = estimate_3d_pose(keypoints_2d, bone_lengths)
                # print(keypoints_3d)


                # 预测图片时是否进行翻转（水平翻转前后两次的结果进行综合判断）
                if flip_test:
                    flip_tensor = transforms.flip_images(frame)
                    flip_outputs = torch.squeeze(
                        transforms.flip_back(model(flip_tensor.cuda()), person_info["flip_pairs"]),
                    )
                    # feature is not aligned, shift flipped heatmap for higher accuracy
                    # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
                    flip_outputs[..., 1:] = flip_outputs.clone()[..., 0: -1]
                    outputs = (outputs + flip_outputs) * 0.5

                keypoints, scores = transforms.get_final_preds(outputs, [target["reverse_trans"]], True)
                keypoints = np.squeeze(keypoints)
                scores = np.reshape(scores, (17, 1))
                pose_result = np.concatenate((keypoints, scores), 1)
                pose_result = np.array([pose_result])
                # scores = np.squeeze(scores)
                # frame = draw_keypoints(img, keypoints, scores, thresh=0.2, r=3)
                frame = imshow_keypoints(img, pose_result)

                frame = np.array(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 显示fps帧数
        counter += 1  # 计算帧数
        if (time.time() - start_time) != 0:  # 实时显示帧数
            frame = cv2.putText(frame, "FPS {0}".format(float('%.1f' % (counter / (time.time() - start_time)))),
                                (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                                3)
            # src = cv2.resize(frame, (256, 192), interpolation=cv2.INTER_CUBIC)  # 窗口大小
            cv2.imshow('ZhangZiYang Model', frame)
            # print("frame_width:", frame_width, " frame_height", frame_height, " FPS: ",
            #       counter / (time.time() - start_time))
            counter = 0
            start_time = time.time()

        # 使用cv2.waitKey()函数设置延迟时间，单位是毫秒，如果按下q键，就退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 使用cv2.destroyAllWindows()函数关闭所有窗口
    cv2.destroyAllWindows()
    # 使用cv2.release()函数释放摄像头资源
    cap.release()

if __name__ == '__main__':
    predict_mp4()