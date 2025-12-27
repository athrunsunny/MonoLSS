import os
import pickle
import copy
import json
import numpy as np
import torch
import cv2
from tqdm import tqdm
from PIL import Image
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion
from mmdet3d.core.bbox import LiDARInstance3DBoxes, get_box_type, CameraInstance3DBoxes


class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]  # 类别
        self.trucation = float(label[1])  # 截断程度
        self.occlusion = float(label[2])  # 遮挡状态  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.alpha = float(label[3])  # 观察角度
        self.box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])),
                              dtype=np.float32)  # 2D边界框的左上角x坐标，左上角y坐标，右下角xy坐标
        self.h = float(label[8])  # 3D 高度
        self.w = float(label[9])  # 3D 宽度
        self.l = float(label[10])  # 3D 长度
        self.pos = np.array((float(label[11]), float(label[12]), float(label[13])),
                            dtype=np.float32)  # 3D位置 相机坐标系X坐标，相机坐标系Y坐标（Y向下），相机坐标系Z坐标（深度）
        self.dis_to_cam = np.linalg.norm(self.pos)
        self.ry = float(label[14])  # 旋转角 弧度，绕相机坐标系Y轴的旋转角（-pi~pi）
        self.score = float(label[15]) if label.__len__() == 16 else -1.0
        self.level_str = None
        self.level = self.get_obj_level()

    def get_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if self.trucation == -1:
            self.level_str = 'DontCare'
            return 0

        if height >= 40 and self.trucation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 1  # Easy
        elif height >= 25 and self.trucation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 2  # Moderate
        elif height >= 25 and self.trucation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 3  # Hard
        else:
            self.level_str = 'UnKnown'
            return 4


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def project_to_image(pts_3d, P):
    '''
    将相机坐标系下的3D边界框的角点, 投影到图像平面上, 得到它们在图像上的2D坐标
    输入: pts_3d是一个nx3的矩阵, 包含了待投影的3D坐标点(每行一个点), P是相机的投影矩阵, 通常是一个3x4的矩阵。
    输出: 返回一个nx2的矩阵, 包含了投影到图像平面上的2D坐标点。
      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)  => normalize projected_pts_2d(2xn)
      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)   => normalize projected_pts_2d(nx2)
    '''
    n = pts_3d.shape[0]  # 获取3D点的数量
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))  # 将每个3D点的坐标扩展为齐次坐标形式（4D），通过在每个点的末尾添加1，创建了一个nx4的矩阵。
    pts_2d = np.dot(pts_3d_extend,
                    np.transpose(P))  # 将扩展的3D坐标点矩阵与投影矩阵P相乘，得到一个nx3的矩阵，其中每一行包含了3D点在图像平面上的投影坐标。每个点的坐标表示为[x, y, z]。
    pts_2d[:, 0] /= pts_2d[:, 2]  # 将投影坐标中的x坐标除以z坐标，从而获得2D图像上的x坐标。
    pts_2d[:, 1] /= pts_2d[:, 2]  # 将投影坐标中的y坐标除以z坐标，从而获得2D图像上的y坐标。
    return pts_2d[:, 0:2]  # 返回一个nx2的矩阵,其中包含了每个3D点在2D图像上的坐标。


def compute_box_3d(obj, P, return2d_mat=False):
    '''
    计算对象的3D边界框在图像平面上的投影
    输入: obj代表一个物体标签信息,  P代表相机的投影矩阵-内参。
    输出: 返回两个值, corners_3d表示3D边界框在 相机坐标系 的8个角点的坐标-3D坐标。
                                     corners_2d表示3D边界框在 图像上 的8个角点的坐标-2D坐标。
    '''
    # 计算一个绕Y轴旋转的旋转矩阵R，用于将3D坐标从世界坐标系转换到相机坐标系。obj.ry是对象的偏航角
    R = roty(obj.ry)

    # 物体实际的长、宽、高
    l = obj.l
    w = obj.w
    h = obj.h

    # 存储了3D边界框的8个角点相对于对象中心的坐标。这些坐标定义了3D边界框的形状。
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # 1、将3D边界框的角点坐标从对象坐标系转换到相机坐标系。它使用了旋转矩阵R
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # 3D边界框的坐标进行平移
    corners_3d[0, :] = corners_3d[0, :] + obj.pos[0]
    corners_3d[1, :] = corners_3d[1, :] + obj.pos[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.pos[2]

    if not return2d_mat:
        return np.transpose(corners_3d)

    # 2、检查对象是否在相机前方，因为只有在相机前方的对象才会被绘制。
    # 如果对象的Z坐标（深度）小于0.1，就意味着对象在相机后方，那么corners_2d将被设置为None，函数将返回None。
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # 3、将相机坐标系下的3D边界框的角点，投影到图像平面上，得到它们在图像上的2D坐标。
    corners_2d = project_to_image(np.transpose(corners_3d), P)
    return corners_2d, np.transpose(corners_3d)


cv2Palette = (
    (255, 0, 0),
    (0, 60, 255),
    (0, 255, 255),
    (10, 40, 50),
    (0, 255, 100),
    (40, 125, 255)
)


def draw_projected_box3d(image, qs, color=(0, 60, 255), thickness=2, conf=None, cls=-1):
    '''
    qs: 包含8个3D边界框角点坐标的数组, 形状为(8, 2)。图像坐标下的3D框, 8个顶点坐标。
    '''
    ''' Draw 3d bounding box in image
        qs: (8,2) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
          
            4 -------- 5
           /|         /|
          7 -------- 6 .
          | |        | |
          . 0 -------- 1
          |/         |/
          3 -------- 2
    '''
    qs = qs.astype(np.int32)  # 将输入的顶点坐标转换为整数类型，以便在图像上绘制。

    # # 这个循环迭代4次，每次处理一个边界框的一条边。
    # for k in range(0, 4):
    #     # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
    #
    #     # 定义了要绘制的边的起始点和结束点的索引。在这个循环中，它用于绘制边界框的前四条边。
    #     i, j = k, (k + 1) % 4
    #     cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
    #
    #     # 定义了要绘制的边的起始点和结束点的索引。在这个循环中，它用于绘制边界框的后四条边，与前四条边平行
    #     i, j = k + 4, (k + 1) % 4 + 4
    #     cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
    #
    #     # 定义了要绘制的边的起始点和结束点的索引。在这个循环中，它用于绘制连接前四条边和后四条边的边界框的边。
    #     i, j = k, k + 4
    #     cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

    shift = 4
    pointMultiplier = 2 ** shift
    corners = qs.astype(np.int32)

    lineColor = cv2Palette[cls]
    if conf is not None:
        text = f"{conf:.3f}"
        org = ((int(corners[5, 0])), int(corners[5, 1] - 2))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        thickness = 1
        cv2.PutText(image, text, org, font, font_scale, (0, 60, 255), thickness)

    corners = corners * pointMultiplier
    cv2.polylines(image, [corners[[2, 3, 7, 6]].astype(int)], True, lineColor, 1, cv2.LINE_AA, shift)
    cv2.polylines(image, [corners[:4].astype(int)], True, lineColor, 1, cv2.LINE_AA, shift)
    cv2.polylines(image, [corners[4:].astype(int)], True, lineColor, 1, cv2.LINE_AA, shift)
    cv2.polylines(image, [corners[[0, 1, 5, 4]].astype(int)], True, (255, 255, 255), 1, cv2.LINE_AA, shift)

    return image


###########################################################################################
classes = ['Car']


def inverse_pose(T_wc, feizhengding=False):
    """
    Args:
        T_wc (np.array): world coordinate to camera coordinata.shape(4,4)
    Returns:
        T_wc_inv (np.array): camera coordinate to world coordinata.shape(4,4)
    """
    if isinstance(T_wc, list):
        T_wc = np.array(T_wc)
    R = T_wc[:3, :3]
    t = T_wc[:3, 3]
    if feizhengding:
        R_inv = np.linalg.inv(R)
    else:
        R_inv = R.T  # 旋转矩阵的正交性质
    t_inv = -R_inv @ t
    T_wc_inv = np.eye(4)
    T_wc_inv[:3, :3] = R_inv
    T_wc_inv[:3, 3] = t_inv
    return T_wc_inv


def plot_rect3d_on_img(img,
                       num_rects,
                       rect_corners,
                       labels,
                       color=(0, 255, 0),
                       thickness=1,
                       shift=4
                       ):
    pointMultiplier = 2 ** shift
    for i in range(num_rects):
        corners = rect_corners[i].astype(np.int32)
        singleCorner = corners * pointMultiplier
        lineColor = cv2Palette[0]

        cv2.polylines(img, [corners[[2, 3, 7, 6]].astype(int)], True, lineColor, 1, cv2.LINE_AA, shift)
        cv2.polylines(img, [corners[:4].astype(int)], True, lineColor, 1, cv2.LINE_AA, shift)
        cv2.polylines(img, [corners[4:].astype(int)], True, lineColor, 1, cv2.LINE_AA, shift)
        cv2.polylines(img, [corners[[0, 1, 5, 4]].astype(int)], True, (255, 255, 255), 1, cv2.LINE_AA, shift)

    return img.astype(np.int32)


def draw_lidar_bbox3d_on_img(bboxes3d,
                             labels,
                             raw_img,
                             lidar2img_rt,
                             img_metas,
                             color=(0, 255, 0),
                             thickness=1,
                             return_point=False):
    img = raw_img.copy()
    H, W, C = img.shape
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate([corners_3d.reshape(-1, 3), np.ones((num_bbox * 8, 1))], axis=-1)
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()

    pts_2d = pts_4d @ lidar2img_rt.T
    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    if return_point:
        return imgfov_pts_2d
    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, labels, color, thickness)


def convert_yaw_lidar_to_cam(yaw_lidar, lidar2cam_rotation):
    """
    Args:
        yaw_lidar:LiDAR坐标系下的航向角（弧度） [N]
        lidar2cam_rotation:LiDAR到到相机的旋转矩阵 [3, 3]
    Returns:
        yaw_cam:相机坐标系下的航向角 [N]
    """
    # 示例：假设LiDAR的yaw绕Z轴， 相机yaw也是绕Z轴，但是Y轴相反
    yaw_cam = -yaw_lidar  # 需要根据实际标定调整符号

    # 通用情况：通过旋转矩阵计算
    # 提取X-Y平面的旋转分量
    r11, r12 = lidar2cam_rotation[0, 0], lidar2cam_rotation[0, 1]
    r21, r22 = lidar2cam_rotation[1, 0], lidar2cam_rotation[1, 1]
    delta_theta = torch.atan2(r21, r11)
    yaw_cam = delta_theta - yaw_lidar  # 外参矩阵引入的初始旋转偏移

    yaw_cam = (yaw_cam + torch.pi) % (2 * torch.pi) - torch.pi
    return yaw_cam.numpy()


if __name__ == '__main__':
    data_type = 'train'
    pkl_file = f'bevdetv3-self_new_infos_{data_type}.pkl'
    source_path = f''  # 存放标注pkl的路径
    save_path = f''  # 保存结果路径
    os.makedirs(save_path, exist_ok=True)

    op = source_path
    data_root = op + '/bevdetv3-self_new_infos_train.pkl'
    out_dir = f'{op}/vis'
    os.makedirs(out_dir, exist_ok=True)

    root_path = f''  # 存放图片路径

    dair_datainfo_file_path = os.path.join(source_path, pkl_file)
    with open(dair_datainfo_file_path, 'rb') as fb:
        list_datainfo = pickle.load(fb)

    start = 25000
    for n, data in enumerate(tqdm(list_datainfo['infos'][start:])):
        if n % 500 != 0:
            continue

        frame = data
        boxes3D = frame['gt_boxes']
        labels = frame['gt_names']
        labels = [classes.index(item) for item in labels]
        box_type_3d, box_mode_3d = get_box_type('LiDAR')
        gt_bboxes = LiDARInstance3DBoxes(
            np.concatenate([boxes3D, frame['gt_velocity']], -1).astype(np.float32), box_dim=7 + 2,
            origin=(0.5, 0.5, 0.5)).convert_to(box_mode_3d)

        image_paths = []
        lidar2img_rts = []
        cam_intrinsic = []
        lidar2cam_rts = []
        for cam_type, cam_info in frame['cams'].items():
            image_paths.append(cam_info['data_path'])
            # obtain lidar to image transformation matrix
            # 坐标系的转化: 从小到大直接乘, 从大到小则inverse
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])  # 旋转矩阵是正交矩阵R-1==R.T
            # 左乘旋转矩阵绕固定坐标系旋转，右乘旋转矩阵绕自身坐标系旋转。
            # 绕固定坐标系旋转讨论的是向量的旋转，绕自身坐标系旋转讨论的是坐标变换。
            # 实际上为lidar2cam_t = cam2lidar_t @ cam2lidar_r
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T  # lidar2cam_r.T为相机到雷达的旋转矩阵
            lidar2cam_rt[3, :3] = -lidar2cam_t  # 雷达到相机与相机到雷达方向相反，还差一个负号
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic  # 构建相机内参的齐次方阵
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)  # 得到雷达到图像坐标系的旋转矩阵
            lidar2img_rts.append(lidar2img_rt)
            cam_intrinsic.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)  # 雷达到相机坐标系的旋转矩阵

        results = dict(
            lidar2cam=lidar2cam_rts,
            lidar2img=lidar2img_rts,
            img_filename=image_paths,
            img=[cv2.cvtColor(np.asarray(Image.open(item)), cv2.COLOR_BGR2RGB) for item in image_paths],
            intrinsics=cam_intrinsic,
            gt_bboxes_3d=gt_bboxes,
            gt_labels_3d=labels,
            world2img=[frame['cams']['CAM_FRONT']['lidar2img']],
            boxes3D=frame['gt_boxes'],
        )

        for i in range(len(results['lidar2img'])):
            img = results['img'][i]
            world2image = results['lidar2img'][i]
            new_img = draw_lidar_bbox3d_on_img(gt_bboxes, labels, img, world2image, dict())

        lidar2img = results['lidar2img']
        intrinsics = results['intrinsics']
        extrinsics = [np.linalg.inv(intrin) @ l2i for intrin, l2i in zip(intrinsics, lidar2img)]
        cam2world = [inverse_pose(item, feizhengding=True) for item in extrinsics]

        draw_bev_img = False
        if draw_bev_img:
            pass
        else:
            whole_img = new_img

        whole_img = cv2.resize(whole_img, (int(whole_img.shape[1] / 2), int(whole_img.shape[0] / 2)))
        frame_name = frame['cams']['CAM_FRONT']['data_path'].split('/')[-1].split('.jpg')[0]
        img_filename = os.path.join(out_dir, frame_name + '_' + str(n + start) + '_alignbev.jpg')
        cv2.imwrite(img_filename, whole_img)

        # 获取2d框
        for i in range(len(results['lidar2img'])):
            img = results['img'][i]

            gt_bboxes_3d = results['gt_bboxes_3d']
            gt_bbox = gt_bboxes_3d.tensor
            second_column = gt_bbox[:, 1]
            sorted_values, sorted_indices = torch.sort(second_column, descending=True)
            sorted_gt_bbox = gt_bbox[sorted_indices]

            # 计算2d框
            lidar2img_rt = results['lidar2img'][i]
            corners_3d = gt_bboxes_3d.corners
            num_bbox = corners_3d.shape[0]
            pts_4d = np.concatenate([corners_3d.reshape(-1, 3), np.ones((num_bbox * 8, 1))], axis=-1)
            lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
            if isinstance(lidar2img_rt, torch.Tensor):
                lidar2img_rt = lidar2img_rt.cpu().numpy()

            pts_2d = pts_4d @ lidar2img_rt.T
            pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
            pts_2d[:, 0] /= pts_2d[:, 2]
            pts_2d[:, 1] /= pts_2d[:, 2]
            imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

            x_coords = imgfov_pts_2d[..., 0]
            y_coords = imgfov_pts_2d[..., 1]
            H,W,C = img.shape

            tmp_bbox = []
            mask = np.ones(gt_bboxes_3d.tensor.shape[0], dtype=bool)
            for k in range(num_bbox):
                x_min = int(np.min(x_coords[k])) if np.min(x_coords[k]) > 0 else 0
                x_max = int(np.max(x_coords[k])) if np.max(x_coords[k]) < W else W
                y_min = int(np.min(y_coords[k])) if np.min(y_coords[k]) > 0 else 0
                y_max = int(np.max(y_coords[k])) if np.max(y_coords[k]) < H else H

                if x_min > x_max:
                    mask[k] = 0
                    continue

                if y_min > y_max:
                    mask[k] = 0
                    continue

                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
                tmp_bbox.append((x_min, y_min, x_max, y_max))

            save_path = os.path.join(out_dir, str(n + start) + '.jpg')
            cv2.imwrite(save_path, img)

        # 获取相机坐标系下的3D标注框
        for i in range(len(results['lidar2img'])):
            gt_bboxes_3d = results['gt_bboxes_3d']
            gt_bbox = gt_bboxes_3d.tensor
            second_column = gt_bbox[:, 1]
            sorted_values, sorted_indices = torch.sort(second_column, descending=True)
            sorted_gt_bbox = gt_bbox[sorted_indices]

            lidar2cam_rt = results['lidar2cam'][i]
            centers = gt_bboxes_3d.gravity_center
            dims = gt_bboxes_3d.dims # (w,l,h)
            yaws = gt_bboxes_3d.yaw

            centers_hom = torch.cat([centers, torch.ones_like(centers[:, :1])], dim=1)
            centers_cam = centers_hom @ lidar2cam_rt.T
            centers_cam = centers_cam[:, :3]

            dims_cam = dims[:, [0,2,1]] # 相机坐标系下 w（右），h （下）， l （前）

            yaws_cam = -yaws

            camera_boxes_tensor = torch.cat([centers_cam, dims_cam, yaws_cam.unsqueeze(1)],dim=1)
            camera_boxes = CameraInstance3DBoxes(camera_boxes_tensor, box_dim=7)

        img = results['img'][0]
        cam2img = results['intrinsics'][0]
        gt_bboxes_3d = results['gt_bboxes_3d']

        corners_3d = gt_bboxes_3d.corners
        num_bbox = corners_3d.shape[0]
        pts_4d = np.concatenate([corners_3d.reshape(-1, 3), np.ones((num_bbox * 8, 1))], axis=-1)
        lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
        if isinstance(lidar2img_rt, torch.Tensor):
            lidar2img_rt = lidar2img_rt.cpu().numpy()

        pts_2d = pts_4d @ lidar2img_rt.T @ np.linalg.inv(cam2img.T)
        pts_2d = pts_2d.reshape(num_bbox, 8, 4)[:, :, :3]
        pts_2d = torch.from_numpy(pts_2d).float()

        for i in range(len(mask)):
            if mask[i]:
                intrinsics = cam2img[:3, :]
                cam_3d = camera_boxes_tensor[i]
                cam_3d = cam_3d.cpu().numpy().tolist()
                pic_2d = tmp_bbox[i]

                label = classes[labels[i]]

                trucation = 0.0
                occ = 0
                alpha = 0.0

                box_yaw_new = (gt_bboxes_3d.yaw[i].numpy() - np.pi/2)

                dim = gt_bboxes_3d.dims[i]
                l = dim[1].item()
                h = dim[2].item()
                w = dim[0].item()

                cam_3d_bbox = np.array([label, trucation, occ,alpha, pic_2d[0], pic_2d[1], pic_2d[2], pic_2d[3],
                                        1.5, w, l, cam_3d[0], cam_3d[1] + 0.5, cam_3d[2], box_yaw_new]).tolist()

                res_str = ' '.join([str(x) for x in cam_3d_bbox])
                print(res_str)
                object_ = Object3d(res_str)
                box3d_pts_2d, box3d_pts_3d = compute_box_3d(object_, intrinsics, return2d_mat=True)
                img = draw_projected_box3d(img, box3d_pts_2d, conf=object_.score, cls=0)
        cv2.imwrite(f'{out_dir}/image_with_3Dbbox{n+start}.jpg', img.astype(np.uint8))

# 可视化上看，自有数据集的目标转换到相机坐标系下之后，目标的朝向会朝相机中心偏，导致一些目标在相机坐标系下的3D框航向角有问题，
# 可能是因为实际的雷达到相机的外参非正定矩阵导致的问题






