# -*- coding: utf-8 -*-

import argparse
import logging
import random
import os
import copy
import cv2
from math import pi
from tqdm import tqdm
from types import SimpleNamespace

from fastprogress import progress_bar

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as T

from models import PixelNeRFNet
from renderer import NeRFRenderer

import wandb

device = torch.device("cuda:0")
DATA_DIR = '/home/liqi/data/KITTI/training'
img_transform = T.Compose([
    T.ToTensor(),
    T.Resize((128, 128)),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

config = SimpleNamespace(run_name="DDPM_conditional",
                         epochs=100,
                         noise_steps=1000,
                         seed=42,
                         batch_size=10,
                         img_size=64,
                         num_classes=10,
                         dataset_path='datasets/cifar10-64',
                         train_folder="train",
                         val_folder="test",
                         device="cuda",
                         slice_size=1,
                         do_validation=True,
                         fp16=True,
                         log_every_epoch=10,
                         num_workers=10,
                         lr=5e-3)

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s",
                    level=logging.INFO,
                    datefmt="%I:%M:%S")


def mk_folders(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


def set_seed(s, reproducible=False):
    "Set random seed for `random`, `torch`, and `numpy` (where available)"
    try:
        torch.manual_seed(s)
    except NameError:
        pass
    try:
        torch.cuda.manual_seed_all(s)
    except NameError:
        pass
    try:
        np.random.seed(s % (2**32 - 1))
    except NameError:
        pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(),
                                             ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class KITTI(torch.utils.data.Dataset):
    def __init__(self, ):
        super().__init__()
        self.filelist = [
            f[:-10] for f in os.listdir(f"{DATA_DIR}/nerf") if "label" in f
        ]
        self.filelist.sort()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        idx = self.filelist[idx]
        img = cv2.imread(f'{DATA_DIR}/nerf/%s_patch.png' % idx)
        img = img_transform(img)
        return img

    def __getviews__(self, idx):
        import kitti_util
        idx = self.filelist[idx]
        img = cv2.imread(f'{DATA_DIR}/nerf/%s_patch.png' % idx)
        img = img_transform(img)

        render_rays = kitti_util.gen_rays(
            self.cam_pos, H, canvas.shape[0],
            torch.tensor([calib.f_u, calib.f_v]), 0, np.inf,
            torch.tensor([calib.c_u, calib.c_v])
        )[0].numpy()


        test_data = list()
        out_shape = list()
        for ry in ry_list:
            _,_,_,_,_,_,_, l, h, w, _ = [float(a) for a in obj]
            xmin, ymin, xmax, ymax = box3d_to_image_roi(txyz + [l, h, w, ry], calib.P, canvas.shape)

            cam_rays = render_rays[int(ymin):int(ymax), int(xmin):int(xmax), :].reshape(-1, 8)

            objs = np.array(txyz + [l, h, w, ry]).reshape(1, 7)

            ray_o = kitti_util.world2object(np.zeros((len(cam_rays), 3)), objs)
            ray_d = kitti_util.world2object(cam_rays[:, 3:6], objs, use_dir=True)

            z_in, z_out, intersect = kitti_util.ray_box_intersection(ray_o, ray_d)

            bounds =  np.ones((*ray_o.shape[:-1], 2)) * -1
            bounds [intersect, 0] = z_in
            bounds [intersect, 1] = z_out

            cam_rays = np.concatenate([ray_o, ray_d, bounds], -1)

            out_shape.append( [int(ymax)-int(ymin), int(xmax)-int(xmin) ])

            test_data.append( collate_lambda_test(img, cam_rays) )

        return img, test_data, out_shape


class Net(nn.Module):
    def __init__(self,
                 hidden_size=128,
                 n_blocks=8,
                 n_blocks_view=1,
                 skips=[4],
                 n_freq_posenc=10,
                 n_freq_posenc_views=4,
                 z_dim=128,
                 rgb_out_dim=3):
        super().__init__()
        self.n_freq_posenc = n_freq_posenc
        self.n_freq_posenc_views = n_freq_posenc_views
        self.skips = skips
        self.z_dim = z_dim

        self.n_blocks = n_blocks
        self.n_blocks_view = n_blocks_view

        dim_embed = 3 * self.n_freq_posenc * 2
        dim_embed_view = 3 * self.n_freq_posenc_views * 2

        # Density Prediction Layers
        self.fc_in = nn.Linear(dim_embed, hidden_size)

        if z_dim > 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        self.blocks = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for i in range(n_blocks - 1)])
        n_skips = sum([i in skips for i in range(n_blocks - 1)])

        if n_skips > 0:
            self.fc_z_skips = nn.ModuleList(
                [nn.Linear(z_dim, hidden_size) for i in range(n_skips)])
            self.fc_p_skips = nn.ModuleList(
                [nn.Linear(dim_embed, hidden_size) for i in range(n_skips)])

        self.sigma_out = nn.Linear(hidden_size, 1)

        # Feature Prediction Layers
        self.fc_z_view = nn.Linear(z_dim, hidden_size)
        self.feat_view = nn.Linear(hidden_size, hidden_size)
        self.fc_view = nn.Linear(dim_embed_view, hidden_size)
        self.feat_out = nn.Linear(hidden_size, rgb_out_dim)

        self.blocks_view = nn.ModuleList([
            nn.Linear(dim_embed_view + hidden_size, hidden_size)
            for _ in range(n_blocks_view - 1)
        ])

        self.fc_shape = nn.Sequential(nn.Linear(512, 128), nn.ReLU())

        self.fc_app = nn.Sequential(nn.Linear(512, 128), nn.ReLU())

        self.fc_rgb = nn.Sequential(nn.Linear(3, 128), nn.ReLU())
        self.fc_sigma = nn.Sequential(nn.Linear(1, 128), nn.ReLU())

        self.fc_z_rgb = nn.Linear(z_dim, hidden_size)
        self.fc_z_sigma = nn.Linear(z_dim, hidden_size)

    def transform_points(self, p, views=False):
        L = self.n_freq_posenc_views if views else self.n_freq_posenc
        p_transformed = torch.cat([
            torch.cat([torch.sin((2**i) * pi * p),
                       torch.cos((2**i) * pi * p)],
                      dim=-1) for i in range(L)
        ],
                                  dim=-1)
        return p_transformed

    def forward(self, rgb_t, sigma_t, p_in, ray_d, latent):
        B, N, _ = p_in.shape

        z_sigma = self.fc_sigma(sigma_t)
        z_rgb = self.fc_rgb(rgb_t)

        z_shape = self.fc_shape(latent)
        z_app = self.fc_app(latent)

        z_shape = z_shape[:, None, :].repeat(1, N, 1)
        z_app = z_app[:, None, :].repeat(1, N, 1)

        p = self.transform_points(p_in)
        net = self.fc_in(p)

        net = net + self.fc_z(z_shape)
        net = net + self.fc_z_sigma(z_sigma)

        net = F.relu(net)

        skip_idx = 0
        for idx, layer in enumerate(self.blocks):
            net = F.relu(layer(net))
            if (idx + 1) in self.skips and (idx < len(self.blocks) - 1):
                net = net + self.fc_z_skips[skip_idx](z_shape)
                net = net + self.fc_p_skips[skip_idx](p)
                skip_idx += 1
        sigma_out = self.sigma_out(net)

        net = self.feat_view(net)
        net = net + self.fc_z_view(z_app)
        net = net + self.fc_z_rgb(z_rgb)

        ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
        ray_d = self.transform_points(ray_d, views=True)
        net = net + self.fc_view(ray_d)

        net = F.relu(net)
        if self.n_blocks_view > 1:
            for layer in self.blocks_view:
                net = F.relu(layer(net))

        feat_out = self.feat_out(net)

        return feat_out, sigma_out


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.data_loader = torch.utils.data.DataLoader(
            KITTI(),
            batch_size=4,
            shuffle=True,
            num_workers=8,
            pin_memory=False,
        )

        self.model = Net().to(device=device)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)

        self.nerf = PixelNeRFNet().to(device=device)
        self.nerf.load_state_dict(
            torch.load(os.path.join("output/", "200.ckpt")))
        self.nerf.eval()
        self.renderer = NeRFRenderer().to(device=device)

    def prepare(self, args):
        mk_folders(args.run_name)
        self.optimizer = optim.AdamW(self.model.parameters(),
                                     lr=args.lr,
                                     eps=1e-5)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=args.lr,
            steps_per_epoch=len(self.data_loader),
            epochs=args.epochs)
        self.loss_fn = nn.MSELoss()
        self.ema = EMA(0.995)
        self.scaler = torch.cuda.amp.GradScaler()

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n, ))

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def add_noise(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None,
                                                                     None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise


    def save_model(self, run_name, epoch=-1):
        "Save model locally and on wandb"
        torch.save(self.model.state_dict(), os.path.join("models", run_name, f"ckpt.pt"))
        torch.save(self.ema_model.state_dict(), os.path.join("models", run_name, f"ema_ckpt.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join("models", run_name, f"optim.pt"))
        at = wandb.Artifact("model", type="model", description="Model weights for DDPM conditional", metadata={"epoch": epoch})
        at.add_dir(os.path.join("models", run_name))
        wandb.log_artifact(at)

    def train_step(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.step_ema(self.ema_model, self.model)
        self.scheduler.step()

    def one_epoch(self, train=True):
        avg_rgb_loss = 0.
        avg_sigma_loss = 0.
        avg_loss = 0.

        if train:
            self.model.train()
        else:
            self.model.eval()

        pbar = progress_bar(self.data_loader, leave=False)
        for i, img in enumerate(pbar):
            img = img.cuda()
            with torch.no_grad():
                latents = self.nerf.encode(img)
                xyz = torch.rand((latents.shape[0], 512, 3)).cuda()
                xyz = 2 * (xyz - .5)

                view_dirs = torch.rand((latents.shape[0], 512, 3)).cuda()
                view_dirs = view_dirs / torch.norm(
                    view_dirs, dim=-1, keepdim=True)

                rgb, sigma = self.nerf.decoder(xyz, view_dirs, latents)
                rgb, sigma = torch.sigmoid(rgb), F.softplus(sigma)
                sigma = (sigma - 25.) / 25.  # rescale

                t = self.sample_timesteps(rgb.shape[0]).to(device)
                rgb_t, rgb_noise = self.add_noise(rgb, t)
                sigma_t, sigma_noise = self.add_noise(sigma, t)

            with torch.autocast("cuda") and (torch.inference_mode() if not train else torch.enable_grad()):
                predicted_rgb_noise, predicted_sigma_noise = self.model(
                    rgb_t, sigma_t, xyz, view_dirs, latents)
                rgb_loss = self.loss_fn(rgb_noise, predicted_rgb_noise)
                sigma_loss = self.loss_fn(sigma_noise, predicted_sigma_noise)

                total_loss = rgb_loss + sigma_loss

                avg_rgb_loss += rgb_loss
                avg_sigma_loss += sigma_loss
                avg_loss += total_loss

                if train:
                    self.train_step(total_loss)

                    wandb.log({
                        "train_rgb_loss": rgb_loss.item(),
                        "train_sigma_loss": sigma_loss.item(),
                        "train_loss": total_loss.item()
                    })
                    pbar.comment = f"RGB={rgb_loss.item():2.3f} SIGMA={sigma_loss.item():2.3f} TOTAL={total_loss.item():2.3f}"

        return avg_rgb_loss.mean().item(), avg_sigma_loss.mean().item(), avg_loss.mean().item()


    def log_images(self):
        "Log images to wandb and save them to disk"

        import nuscenes_util
        import numpy as np

        W = 900
        H = 1600
        fl_x = 800
        fl_y = 800
        cx = 800
        cy = 400

        rotation = 0
        angle = rotation * np.pi / 180
        obj_to_cam_trans = [0, 2, 10]
        obj_to_cam_rot = np.array([[np.cos(angle), 0, np.sin(angle)],
                                   [0, 1, 0],
                                   [-np.sin(angle), 0, np.cos(angle)]]) @ \
                         np.array([[1, 0, 0],
                                   [0, 0, -1],
                                   [0, 1, 0]])
        cam_to_obj_rot = obj_to_cam_rot.T
        cam_to_obj_trans = -obj_to_cam_rot.T @ obj_to_cam_trans

        render_rays = nuscenes_util.gen_rays(
           self.cam_pos, W, H,
           torch.tensor([fl_x, fl_y]), 0, np.inf,
           torch.tensor([cx, cy])
        )[0].flatten(0, 1).numpy()

        obj_size = [2, 4, 2]

        ray_o = nuscenes_util.camera2object(render_rays[:, :3], cam_to_obj_trans, cam_to_obj_rot, obj_size, pos=True)
        ray_d = nuscenes_util.camera2object(render_rays[:, 3:6], cam_to_obj_trans, cam_to_obj_rot, obj_size, pos=False)
        z_in, z_out, intersect = nuscenes_util.ray_box_intersection(ray_o, ray_d)

        bounds = np.ones((*ray_o.shape[:-1], 2)) * -1
        bounds[intersect, 0] = z_in
        bounds[intersect, 1] = z_out

        scene_render_rays = np.concatenate([ray_o, ray_d, bounds], -1)
        _, nc = scene_render_rays.shape
        scene_render_rays = scene_render_rays.reshape(H, W, nc)


    def fit(self, args):
        for epoch in progress_bar(range(args.epochs), total=args.epochs, leave=True):
            logging.info(f"Starting epoch {epoch}:")
            _ = self.one_epoch(train=True)
            # validation
            if args.do_validation:
                avg_rgb_loss, avg_sigma_loss, avg_loss = self.one_epoch(train=False)
                wandb.log({"val_rgb_loss": avg_rgb_loss,
                           "val_sigma_loss": avg_sigma_loss,
                           "val_loss": avg_loss})

            # log predicitons
            # if epoch % args.log_every_epoch == 0:
                # self.log_images()

        # save model
        self.save_model(run_name=self.run_name, epoch=epoch)


def parse_args(config):
    parser = argparse.ArgumentParser(description='Process hyper-parameters')
    parser.add_argument('--run_name',
                        type=str,
                        default=config.run_name,
                        help='name of the run')
    parser.add_argument('--epochs',
                        type=int,
                        default=config.epochs,
                        help='number of epochs')
    parser.add_argument('--seed',
                        type=int,
                        default=config.seed,
                        help='random seed')
    parser.add_argument('--batch_size',
                        type=int,
                        default=config.batch_size,
                        help='batch size')
    parser.add_argument('--image_size',
                        type=int,
                        default=config.img_size,
                        help='image size')
    parser.add_argument('--num_classes',
                        type=int,
                        default=config.num_classes,
                        help='number of classes')
    parser.add_argument('--dataset_path',
                        type=str,
                        default=config.dataset_path,
                        help='path to dataset')
    parser.add_argument('--device',
                        type=str,
                        default=config.device,
                        help='device')
    parser.add_argument('--lr',
                        type=float,
                        default=config.lr,
                        help='learning rate')
    parser.add_argument('--slice_size',
                        type=int,
                        default=config.slice_size,
                        help='slice size')
    parser.add_argument('--noise_steps',
                        type=int,
                        default=config.noise_steps,
                        help='noise steps')
    args = vars(parser.parse_args())

    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)


if __name__ == "__main__":
    parse_args(config)

    set_seed(config.seed)

    diffuser = Diffusion()

    with wandb.init(project="train_nerf_diffusion", group="train", config=config):
        diffuser.prepare(config)
        diffuser.fit(config)
