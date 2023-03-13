import os
import json
import numpy as np
from PIL import Image
from fire import Fire

from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, transform_matrix

from mmdet.apis import init_detector, inference_detector


def main(
        # nuscene config
        version='v1.0-mini',
        dataroot='/home/liqi/data/nuscenes',
        # mmdet config
        config_file='scripts/mmdet/configs/mask2former/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic.py',
        checkpoint_file='scripts/mmdet/checkpoints/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic_20220407_104949-d4919c44.pth',
        ):
    det_model = init_detector(config_file, checkpoint_file, device='cuda:0')

    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

    for i, instance in enumerate(nusc.instance):
        category = nusc.get('category', instance['category_token'])['name']
        if 'car' not in category:
            continue

        anno_record = nusc.get('sample_annotation', instance['first_annotation_token'])

        output_dir = f"{dataroot}/nerf/{version}/inst_{i:06}"

        meta = {}
        frames = []
        image_idx = 0
        while True:
            sample_record = nusc.get('sample', anno_record['sample_token'])
            cams = [key for key in sample_record['data'].keys() if 'CAM' in key]

            for cam in cams:
                data_path, boxes, camera_intrinsic = nusc.get_sample_data(sample_record['data'][cam],
                                                                          selected_anntokens=[anno_record['token']])
                if len(boxes) == 0:
                    continue

                assert len(boxes) == 1

                pts_points = view_points(boxes[0].corners(), view=camera_intrinsic, normalize=True)[:2, :]
                img = np.array(Image.open(data_path))
                bound = np.stack([pts_points.min(1), pts_points.max(1)]).astype('int32')
                bound[bound < 0] = 0
                bound[1, 0] = min(bound[1, 0], img.shape[1])
                bound[1, 1] = min(bound[1, 1], img.shape[0])
                bound_mask = np.zeros(img.shape[:2], dtype=bool)
                bound_mask[bound[0, 1]:bound[1, 1], bound[0, 0]:bound[1, 0]] = 1
                result = inference_detector(det_model, img)

                pan = result['pan_results']
                pan_pred = []
                for x in np.unique(pan[bound_mask]):
                    if x > 1000 and x % 1000 in [2, 3, 4, 5]:
                        pan_pred.append(x)
                if len(pan_pred) == 0:
                    continue

                pan_mask = pan == pan_pred[np.argmax([(pan[bound_mask] == x).sum() for x in pan_pred])]
                pan_mask[~bound_mask] = False

                patch = img[bound[0, 1]:bound[1, 1], bound[0, 0]:bound[1, 0]]
                mask = pan_mask[bound[0, 1]:bound[1, 1], bound[0, 0]:bound[1, 0]].astype('uint8') * 255
                if (mask > 0).sum() < 150:
                    continue

                if not os.path.exists(os.path.join(output_dir, "images")):
                    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

                Image.fromarray(patch).save(f"{output_dir}/images/{image_idx:05}_patch.png")
                Image.fromarray(mask).save(f"{output_dir}/images/{image_idx:05}_mask.png")
                frame = {}

                sample_data = nusc.get('sample_data', sample_record['data'][cam])
                calibrated_sensor = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
                cam_trans = calibrated_sensor['translation']
                cam_rot = Quaternion(calibrated_sensor['rotation'])
                cam_matrix = transform_matrix(translation=cam_trans, rotation=cam_rot)

                ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
                ego_trans = ego_pose['translation']
                ego_rot = Quaternion(ego_pose['rotation'])
                ego_matrix = transform_matrix(translation=ego_trans, rotation=ego_rot)

                obj_trans = anno_record['translation']
                obj_rot = Quaternion(anno_record['rotation'])
                obj_matrix = transform_matrix(translation=obj_trans, rotation=obj_rot)

                frame['rgb_path'] = f"images/{image_idx:05}_patch.png"
                frame['mask_path'] = f"images/{image_idx:05}_mask.png"
                mat = np.linalg.inv(obj_matrix) @ ego_matrix @ cam_matrix
                # mat = mat @ np.array([[-1, 0, 0, 0],
                #                       [0, 1, 0, 0],
                #                       [0, 0, -1, 0],
                #                       [0, 0, 0, 1]])

                frame['transform_matrix'] = mat.tolist()
                frame['w'] = img.shape[1]
                frame['h'] = img.shape[0]
                frame['fl_x'] = camera_intrinsic[0, 0]
                frame['fl_y'] = camera_intrinsic[1, 1]
                frame['cx'] = camera_intrinsic[0, 2]
                frame['cy'] = camera_intrinsic[1, 2]
                frame['x_min'] = int(bound[0, 0])
                frame['x_max'] = int(bound[1, 0])
                frame['y_min'] = int(bound[0, 1])
                frame['y_max'] = int(bound[1, 1])
                frame['ego_trans'] = ego_pose['translation']
                frame['ego_rot'] = ego_pose['rotation']
                frame['obj_trans'] = anno_record['translation']
                frame['obj_rot'] = anno_record['rotation']
                frame['obj_size'] = anno_record['size']
                frame['cam_trans'] = calibrated_sensor['translation']
                frame['cam_rot'] = calibrated_sensor['rotation']

                frame['cam_to_obj_rot'] = Quaternion(matrix=mat).elements.tolist()
                frame['cam_to_obj_trans'] = mat[:, 3].tolist()
                frames.append(frame)

                image_idx = image_idx + 1

            if anno_record['next'] != '':
                anno_record = nusc.get('sample_annotation', anno_record['next'])
            else:
                break

        if image_idx == 0:
            continue

        meta['frames'] = frames
        meta['aabb_scale'] = 128

        with open(f'{output_dir}/transforms.json', 'w') as f:
            json.dump(meta, f, indent=4)

        print(f'instance {i} processed')

if __name__ == '__main__':
    Fire(main)
