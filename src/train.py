import os

import shutil

import cv2

import kitti_util
import numpy as np
from models import PixelNeRFNet
import torch.nn.functional as F
from renderer import NeRFRenderer

import torch
import random

from PIL import Image

from nuscenes import NuScenes, collate_lambda_train
import nuscenes_util

from functools import partial

import argparse
import imageio
import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ray_batch_size", type=int, default=2048)
    parser.add_argument("--print_interval", type=int, default=5)
    parser.add_argument("--vis_interval", type=int, default=100)
    parser.add_argument("--ckpt", type=str, default='200.ckpt')
    parser.add_argument("--ckpt_interval", default=5, help='checkpoint interval (in epochs)')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=float, default=1000000)
    parser.add_argument("--save_path", type=str, default='output_nuscenes')
    parser.add_argument("--viz_path", type=str, default='viz_nuscenes')
    parser.add_argument("--demo", action="store_true")
    return parser.parse_args()

def make_canvas(patches):
    image = patches.pop(0)
    banner = list()
    hmax = max([p.shape[0] for p in patches]) + 10
    for p in patches:
        H, W, _ = p.shape
        a = (hmax - H) // 2
        b = hmax - H - a
        pp = np.pad(p, ((a, b), (0, 0), (0, 0)))
        banner.append(pp)
    banner = np.concatenate(banner, 1)
    imW, bnW = image.shape[1], banner.shape[1]
    a = (bnW - imW) // 2
    b = bnW - imW - a
    image = np.pad(image, ((0, 0), (a, b), (0, 0)))
    canvas = np.concatenate([image, banner], 0)
    return canvas

class PixelNeRFTrainer():
    def __init__(self, args, net, renderer, train_dataset, test_dataset, device):
        super().__init__()
        self.args = args
        self.device = device
        self.net = net
        self.renderer = renderer
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
            collate_fn = partial(collate_lambda_train, ray_batch_size=args.ray_batch_size)
        )

        os.makedirs(self.args.save_path, exist_ok = True)


        self.num_epochs = args.epochs

        self.optim = torch.optim.Adam(net.parameters(), lr=args.lr)

        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self.optim, milestones=[100, 150], gamma=0.1
        )

    def train_step(self, data, is_train=True):
        img_cnt, src_images, all_rays, all_rgb_gt, all_mask_gt = data

        img_cnt = img_cnt.to(self.device)
        src_images = src_images.to(self.device)
        all_rays = all_rays.to(self.device)
        all_rgb_gt = all_rgb_gt.to(self.device)
        all_mask_gt = all_mask_gt.to(self.device)

        images = torch.cat([src_images[i, :img_cnt[i]] for i in range(len(img_cnt))])
        latent = self.net.encode(images)
        cum = torch.cat((torch.tensor([0]).to(img_cnt.device), torch.cumsum(img_cnt, 0)))
        latent = torch.stack([latent[cum[i]:cum[i+1]].mean(0) for i in range(len(img_cnt))])

        render_dict = self.renderer(self.net, all_rays, latent)

        render_rgb = render_dict['rgb']

        intersect = render_dict['intersect']

        render_rgb = render_rgb[intersect, ...]
        all_rgb_gt = all_rgb_gt[intersect, ...]
        all_mask_gt = all_mask_gt[intersect, ...]

        loss = F.mse_loss(render_rgb, all_rgb_gt * all_mask_gt, reduction='mean')
        #loss = loss.sum() / all_mask_gt.sum()

        if is_train:
            loss.backward()

        return loss


    def vis_step(self, data):
        src_images, all_rays = data

        all_rays = all_rays.to(device)
        src_images = src_images.to(device)

        self.net.eval()
        pred_rgb = list()
        with torch.no_grad():
            latent = self.net.encode(src_images)
            for batch_rays in torch.split(all_rays, self.args.batch_size):
                pred_rgb.append( self.renderer(self.net, batch_rays.flatten(0, 1), latent)['rgb'] )

        self.net.train()

        pred_rgb = torch.cat(pred_rgb, 0).view(-1, 3)

        return pred_rgb


    def vis_scene(self, idx, translation=None, rotation=None):
        H, W, all_rays, rois, intersect, cam_to_obj_trans, cam_to_obj_rot, obj_size = self.test_dataset.__getscene__(idx, translation, rotation)

        self.net.eval()

        all_rays = all_rays.to(device)
        src_images = rois.to(device)
        intersect = intersect.to(device)
        cam_to_obj_trans = cam_to_obj_trans.to(device)
        cam_to_obj_rot = cam_to_obj_rot.to(device)
        obj_size = obj_size.to(device)

        all_rays = all_rays.view(H*W, -1, 8)
        valid_rays = all_rays[intersect, ...]

        _, Nb, _ = valid_rays.shape
        Nk = self.renderer.n_coarse

        with torch.no_grad():
            latents = self.net.encode(src_images)
            latents = latents.mean(0)[None]

            rgb_map = list()
            for batch_rays in tqdm.tqdm(torch.split(valid_rays, self.args.batch_size)):

                rays = batch_rays.view(-1, 8)  # (N * B, 8)
                z_coarse = self.renderer.sample_from_ray(rays)
                empty_space = z_coarse == -1

                rgbs, sigmas = self.renderer.nerf_predict(self.net, rays, z_coarse, latents)

                pts_o = rays[:, None, :3] + z_coarse[:, :, None] * rays[:, None, 3:6]
                pts_o = pts_o.view(-1, Nb, Nk, 3).permute(1, 0, 2, 3).contiguous()

                pts_w = nuscenes_util.object2camera(pts_o.view(Nb, -1, 3), cam_to_obj_trans, cam_to_obj_rot, obj_size)
                pts_w = pts_w.view(Nb, -1, Nk, 3).permute(1, 0, 2, 3).contiguous()

                z_world = torch.norm(pts_w, p=2, dim=-1).view_as(z_coarse)
                z_world[empty_space] = -1

                z_world = z_world.view(-1, Nb*Nk)

                z_sort = torch.sort(z_world, 1).values
                z_args = torch.searchsorted(z_sort, z_world)

                rgbs[empty_space, ...] = 0
                sigmas[empty_space] = 0

                rgbs = rgbs.view(-1, Nb * Nk, 3)
                sigmas = sigmas.view(-1, Nb * Nk)

                rgbs_sort = torch.zeros_like(rgbs).scatter_(1, z_args[:, :, None].repeat(1, 1, 3), rgbs)
                sigmas_sort = torch.zeros_like(sigmas).scatter_(1, z_args, sigmas)

                rgb, depth, weights = self.renderer.volume_render(rgbs_sort, sigmas_sort, z_sort)

                # rgb = self.renderer(self.net, batch_rays, latents)['rgb'][:, 0, :]

                rgb_map.append(rgb)

            rgb_map = torch.cat(rgb_map, 0)

            canvas = torch.zeros(H*W, 3).type_as(all_rays)
            canvas[intersect, :] = rgb_map
            canvas = (canvas.view(H, W, 3).cpu().numpy() * 255).astype(np.uint8)

            return canvas

    def vis_car(self, idx):
        rgbs = self.test_dataset.__getcar__(idx)

        cam_pos = torch.eye(4)[None, ...]
        cam_pos[:, 1, 1] = -1
        cam_pos[:, 2, 2] = -1

        render_rays = nuscenes_util.gen_rays(
            cam_pos, 1600, 900,
            torch.tensor([1250, 1250]), 0, np.inf,
            torch.tensor([800, 450])
        )[0].flatten(0,1).numpy()

        ray_o = render_rays[:, :3]
        ray_o[:, 0] = -1.
        ray_o[:, 1] = -1.
        ray_o[:, 2] = 1.
        ray_d = render_rays[:, 3:6]

        z_in, z_out, intersect = nuscenes_util.ray_box_intersection(ray_o, ray_d)
        bounds =  np.ones((*ray_o.shape[:-1], 2)) * -1
        bounds [intersect, 0] = z_in
        bounds [intersect, 1] = z_out
        all_rays = torch.cat([ray_o, ray_d, bounds])

        self.net.eval()
        rgbs = rgbs.to(self.device)
        with torch.no_grad():
            latent = self.net.encode(rgbs).mean(0)


    def train(self):
        for epoch in range(self.num_epochs):
            batch = 0
            for data in self.train_data_loader:
                losses = self.train_step(data)

                self.optim.step()
                self.optim.zero_grad()

                if batch % self.args.print_interval == 0:
                    print("E", epoch, "B", batch, "loss", losses.item(),"lr", self.optim.param_groups[0]["lr"])

                batch += 1

            if (epoch + 1) % self.args.ckpt_interval == 0:
                torch.save(
                    self.net.state_dict(),
                    os.path.join(
                        self.args.save_path,"epoch_%d.ckpt" % (epoch + 1),
                    )
                )
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()


if __name__ == "__main__":

    args = get_args()

    device = torch.device("cuda:0")

    net = PixelNeRFNet().to(device=device)

    renderer = NeRFRenderer().to(device=device)

    trainer = PixelNeRFTrainer(
        args, net, renderer,
        NuScenes(),
        NuScenes(),
        device
    )

    if args.demo:
        trainer.net.load_state_dict(torch.load(os.path.join(args.save_path, args.ckpt)))

        canvas = trainer.vis_scene(8)

        Image.fromarray(canvas).save('rgb.png')

        #  with imageio.get_writer(os.path.join(args.save_path, 'car.gif'), mode='I', duration=0.5) as writer:
        #      for z in np.arange(0, 36+1)/36*2 * np.pi:
        #          canvas = trainer.vis_scene(0, rotation=z)
        #          canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        #          writer.append_data(canvas)
        #  writer.close()
    else:
        trainer.train()

