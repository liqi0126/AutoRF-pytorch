import os
import cv2
import numpy as np
import json
from PIL import Image
from functools import partial

import torch
import torchvision.transforms as T

import nuscenes_util

DATA_DIR = '/data1/liqi/nuscenes/nerf'

img_transform = T.Compose([T.Resize((128, 128)), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class NuScenes(torch.utils.data.Dataset):
    def __init__(self, version='v1.0-mini'):
        super().__init__()
        self.version = version
        self.filelist = sorted(os.listdir(f"{DATA_DIR}/{version}"))

        self.cam_pos = torch.eye(4)[None, ...]
        self.cam_pos[:, 1, 1] = -1
        self.cam_pos[:, 2, 2] = -1

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        idx = self.filelist[idx]
        data_path = f"{DATA_DIR}/{self.version}/{idx}"
        with open(f'{data_path}/transforms.json', 'r') as f:
            meta = json.load(f)

        rgbs = []
        masks = []
        cam_rays = []
        for frame in meta['frames']:
            rgb = Image.open(f"{data_path}/{frame['rgb_path']}")
            mask = Image.open(f"{data_path}/{frame['mask_path']}")
            rgb = T.ToTensor()(rgb)
            mask = T.ToTensor()(mask)

            h = frame['h']
            w = frame['w']
            fl_x = frame['fl_x']
            fl_y = frame['fl_y']
            cx = frame['cx']
            cy = frame['cy']
            x_min = frame['x_min']
            x_max = frame['x_max']
            y_min = frame['y_min']
            y_max = frame['y_max']
            obj_size = frame['obj_size']
            cam_to_obj_trans = frame['cam_to_obj_trans']
            cam_to_obj_rot = frame['cam_to_obj_rot']

            render_rays = nuscenes_util.gen_rays(
                self.cam_pos, w, h,
                torch.tensor([fl_x, fl_y]), 0, np.inf,
                torch.tensor([cx, cy])
            )[0].numpy()

            cam_ray = render_rays[y_min:y_max, x_min:x_max, :].reshape(-1, 8)

            ray_o = nuscenes_util.camera2object(cam_ray[:, :3], cam_to_obj_trans, cam_to_obj_rot, obj_size, pos=True)
            ray_d = nuscenes_util.camera2object(cam_ray[:, 3:6], cam_to_obj_trans, cam_to_obj_rot, obj_size, pos=False)

            z_in, z_out, intersect = nuscenes_util.ray_box_intersection(ray_o, ray_d)

            bounds = np.ones((*ray_o.shape[:-1], 2)) * -1
            bounds[intersect, 0] = z_in
            bounds[intersect, 1] = z_out

            cam_ray = np.concatenate([ray_o, ray_d, bounds], -1)
            cam_ray = torch.FloatTensor(cam_ray)

            rgbs.append(rgb)
            masks.append(mask)
            cam_rays.append(cam_ray)

        return rgbs, masks, cam_rays

    def __getcar__(self, idx):
        idx = self.filelist[idx]
        data_path = f"{DATA_DIR}/{self.version}/{idx}"
        with open(f'{data_path}/transforms.json', 'r') as f:
            meta = json.load(f)

        rgbs = []
        for frame in meta['frames']:
            rgb = Image.open(f"{data_path}/{frame['rgb_path']}")
            rgb = T.ToTensor()(rgb)
            rgb = img_transform(rgb)
            rgbs.append(rgb)
        return torch.stack(rgbs)

    def __getscene__(self, idx, translation=None, rotation=None):
        idx = self.filelist[idx]
        data_path = f"{DATA_DIR}/{self.version}/{idx}"
        with open(f'{data_path}/transforms.json', 'r') as f:
            meta = json.load(f)

        rgbs = []
        for frame in meta['frames']:
            rgb = Image.open(f"{data_path}/{frame['rgb_path']}")
            rgb = T.ToTensor()(rgb)
            rgb = img_transform(rgb)
            rgbs.append(rgb)

        rgbs = torch.stack(rgbs)

        H = meta['frames'][0]['h']
        W = meta['frames'][0]['w']
        fl_x = meta['frames'][0]['fl_x']
        fl_y = meta['frames'][0]['fl_y']
        cx = meta['frames'][0]['cx']
        cy = meta['frames'][0]['cy']
        obj_size = meta['frames'][0]['obj_size']
        cam_to_obj_trans = np.array([0., 0., -10.])
        cam_to_obj_rot = np.array([1., 0., 0., 0.])
        obj_size = np.array([3, 2, 2])

        render_rays = nuscenes_util.gen_rays(
           self.cam_pos, W, H,
           torch.tensor([fl_x, fl_y]), 0, np.inf,
           torch.tensor([cx, cy])
        )[0].flatten(0, 1).numpy()

        # manipulate 3d boxes
        # if translation is not None:
        #     objs = translate(objs, translation)
        #
        # if rotation is not None:
        #     objs = rotate(objs, rotation)

        # get rays from 3d boxes
        ray_o = nuscenes_util.camera2object(render_rays[:, :3], cam_to_obj_trans, cam_to_obj_rot, obj_size, pos=True)
        ray_d = nuscenes_util.camera2object(render_rays[:, 3:6], cam_to_obj_trans, cam_to_obj_rot, obj_size, pos=False)

        z_in, z_out, intersect = nuscenes_util.ray_box_intersection(ray_o, ray_d)

        bounds = np.ones((*ray_o.shape[:-1], 2)) * -1
        bounds[intersect, 0] = z_in
        bounds[intersect, 1] = z_out

        scene_render_rays = np.concatenate([ray_o, ray_d, bounds], -1)
        _, nc = scene_render_rays.shape
        scene_render_rays = scene_render_rays.reshape(H, W, nc)

        return H, W, \
            torch.FloatTensor(scene_render_rays), \
            rgbs, \
            torch.from_numpy(intersect), \
            torch.tensor(cam_to_obj_trans), \
            torch.tensor(cam_to_obj_rot), \
            torch.tensor(obj_size)


def collate_lambda_train(batch, ray_batch_size=1024):
    imgs_bs, masks_bs, rays_bs, rgbs_bs = [], [], [], []
    len_bs = []
    for imgs, masks, cam_rays in batch:
        post_imgs, post_masks, post_rays, post_rgbs = [], [], [], []

        ratio = []
        for mask in masks:
            ratio.append(mask.sum().item())
        ratio = torch.tensor(ratio)
        ratio /= sum(ratio)
        ray_cnts = (ratio * ray_batch_size).int()
        ray_cnts[0] = ray_batch_size - ray_cnts[1:].sum().item()

        for img, mask, cam_ray, ray_cnt in zip(imgs, masks, cam_rays, ray_cnts):
            _, H, W = img.shape

            pix_inds = torch.randint(0, H * W, (ray_cnt,))

            rgb_gt = img.permute(1, 2, 0).flatten(0, 1)[pix_inds, ...]
            mask_gt = mask.permute(1, 2, 0).flatten(0, 1)[pix_inds, ...]
            ray = cam_ray.view(-1, cam_ray.shape[-1])[pix_inds]

            post_imgs.append(img_transform(img))
            post_masks.append(mask_gt)
            post_rays.append(ray)
            post_rgbs.append(rgb_gt)

        post_imgs = torch.stack(post_imgs)
        post_rgbs = torch.cat(post_rgbs)
        post_masks = torch.cat(post_masks)
        post_rays = torch.cat(post_rays)

        len_bs.append(len(imgs))
        imgs_bs.append(post_imgs)
        rgbs_bs.append(post_rgbs)
        masks_bs.append(post_masks)
        rays_bs.append(post_rays)

    max_len = max(len_bs)
    for i in range(len(len_bs)):
        imgs_bs[i] = torch.cat([imgs_bs[i], torch.zeros((max_len - imgs_bs[i].shape[0], *imgs_bs[i].shape[1:]))])

    len_bs = torch.tensor(len_bs)
    imgs_bs = torch.stack(imgs_bs)
    rgbs_bs = torch.stack(rgbs_bs, 1)
    masks_bs = torch.stack(masks_bs, 1)
    rays_bs = torch.stack(rays_bs, 1)

    return len_bs, imgs_bs, rays_bs, rgbs_bs, masks_bs


if __name__ == '__main__':
    nuscenes = NuScenes()

    rgbs1, masks1, rays1 = nuscenes.__getitem__(1)
    rgbs2, masks2, rays2 = nuscenes.__getitem__(2)

    # import ipdb; ipdb.set_trace()

    train_data_loader = torch.utils.data.DataLoader(
        nuscenes,
        batch_size=12,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=partial(collate_lambda_train, ray_batch_size=1024)
    )

    for i, (lens, imgs, rays, rgbs, msks) in enumerate(train_data_loader):
        print(i)
