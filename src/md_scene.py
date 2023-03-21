
import os
import torch

import tqdm

import imageio
import numpy as np

from metadrive import MetaDriveEnv

from models import PixelNeRFNet
from renderer import NeRFRenderer

W = 1600
H = 900

CX = 800
CY = 450

FL_X = 800
FL_Y = 800

device = torch.device("cuda:0")

def unproj_map(width, height, f, c=None, device="cpu"):
    """
    Get camera unprojection map for given image size.
    [y,x] of output tensor will contain unit vector of camera ray of that pixel.
    :param width image width
    :param height image height
    :param f focal length, either a number or tensor [fx, fy]
    :param c principal point, optional, either None or tensor [fx, fy]
    if not specified uses center of image
    :return unproj map (height, width, 3)
    """
    if c is None:
        c = [width * 0.5, height * 0.5]
    else:
        c = c.squeeze()
    if isinstance(f, float):
        f = [f, f]
    elif len(f.shape) == 0:
        f = f[None].expand(2)
    elif len(f.shape) == 1:
        f = f.expand(2)
    Y, X = torch.meshgrid(
        torch.arange(height, dtype=torch.float32) - float(c[1]),
        torch.arange(width, dtype=torch.float32) - float(c[0]),
    )
    X = X.to(device=device) / float(f[0])
    Y = Y.to(device=device) / float(f[1])
    Z = torch.ones_like(X)
    unproj = torch.stack((X, -Y, -Z), dim=-1)
    unproj /= torch.norm(unproj, dim=-1).unsqueeze(-1)
    return unproj


def gen_rays(poses, width, height, focal, z_near, z_far, c=None, ndc=False):
    """
    Generate camera rays
    :return (B, H, W, 8)
    """
    num_images = poses.shape[0]
    device = poses.device
    cam_unproj_map = (
        unproj_map(width, height, focal.squeeze(), c=c, device=device)
        .unsqueeze(0)
        .repeat(num_images, 1, 1, 1)
    )
    cam_centers = poses[:, None, None, :3, 3].expand(-1, height, width, -1)
    cam_raydir = torch.matmul(
        poses[:, None, None, :3, :3], cam_unproj_map.unsqueeze(-1)
    )[:, :, :, :, 0]


    cam_nears = (
        torch.tensor(z_near, device=device)
        .view(1, 1, 1, 1)
        .expand(num_images, height, width, -1)
    )
    cam_fars = (
        torch.tensor(z_far, device=device)
        .view(1, 1, 1, 1)
        .expand(num_images, height, width, -1)
    )
    return torch.cat(
        (cam_centers, cam_raydir, cam_nears, cam_fars), dim=-1
    )  # (B, H, W, 8)


def ray_box_intersection(ray_o, ray_d, aabb_min=None, aabb_max=None):
    """Returns 1-D intersection point along each ray if a ray-box intersection is detected
    If box frames are scaled to vertices between [-1., -1., -1.] and [1., 1., 1.] aabbb is not necessary
    Args:
        ray_o: Origin of the ray in each box frame, [rays, boxes, 3]
        ray_d: Unit direction of each ray in each box frame, [rays, boxes, 3]
        (aabb_min): Vertex of a 3D bounding box, [-1., -1., -1.] if not specified
        (aabb_max): Vertex of a 3D bounding box, [1., 1., 1.] if not specified
    Returns:
        z_ray_in:
        z_ray_out:
        intersection_map: Maps intersection values in z to their ray-box intersection
    """
    # Source: https://medium.com/@bromanz/another-view-on-the-classic-ray-aabb-intersection-algorithm-for-bvh-traversal-41125138b525
    # https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
    if aabb_min is None:
        aabb_min = np.ones_like(ray_o) * -1. # tf.constant([-1., -1., -1.])
    if aabb_max is None:
        aabb_max = np.ones_like(ray_o) # tf.constant([1., 1., 1.])

    inv_d = np.reciprocal(ray_d)

    t_min = (aabb_min - ray_o) * inv_d
    t_max = (aabb_max - ray_o) * inv_d

    t0 = np.minimum(t_min, t_max)
    t1 = np.maximum(t_min, t_max)

    t_near = np.maximum(np.maximum(t0[..., 0], t0[..., 1]), t0[..., 2])
    t_far = np.minimum(np.minimum(t1[..., 0], t1[..., 1]), t1[..., 2])

    # Check if rays are inside boxes
    intersection_map = t_far > t_near # np.where(t_far > t_near)[0]

    # Check that boxes are in front of the ray origin
    positive_far = (t_far * intersection_map) > 0
    intersection_map = np.logical_and(intersection_map, positive_far)

    if not intersection_map.shape[0] == 0:
        z_ray_in = t_near[intersection_map]
        z_ray_out = t_far[intersection_map]
    else:
        return None, None, None

    return z_ray_in, z_ray_out, intersection_map


def camera2object(pts, cam_to_obj_trans, cam_to_obj_rot, obj_size, pos=True):
    pts_o = np.einsum('BNi,Ai ->ABN', cam_to_obj_rot, pts)
    if pos:
        pts_o += cam_to_obj_trans

    pts_o /= (obj_size / 2)
    if not pos:
        pts_o = pts_o / np.linalg.norm(pts_o, axis=-1, keepdims=True)
    return pts_o


def object2camera(pts, cam_to_obj_trans, cam_to_obj_rot, obj_size):
    pts_w = pts
    pts_w *= (obj_size[:, None] / 2)
    pts_w -= cam_to_obj_trans[:, None]
    pts_w = torch.einsum('BNi,Bki ->BkN', torch.transpose(cam_to_obj_rot, 1, 2), pts_w)
    return pts_w


def get_car_poses(env):
    poses = {}
    poses['ego'] = {'pos': np.array([*env.vehicle.position, 1]), 'yaw': env.vehicle.heading_theta}
    cars = env.engine.traffic_manager.spawned_objects
    poses['traffic'] = {}
    for key in cars:
        poses['traffic'][key] = {'pos': np.array([*cars[key].position, 1]), 'yaw': cars[key].heading_theta}
    return poses


def rotation_matrix(yaw):
    return np.array([np.cos(yaw), -np.sin(yaw), 0, np.sin(yaw), np.cos(yaw), 0, 0, 0, 1]).reshape((3, 3))

def get_mat(trans, rot):
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = trans
    return mat

def get_rays(env, obj_size):
    poses = get_car_poses(env)

    ego_pos = np.array(poses['ego']['pos'])
    ego_yaw = np.array(poses['ego']['yaw'])
    ego_rot = rotation_matrix(ego_yaw)
    ego_mat = get_mat(ego_pos, ego_rot)

    print(ego_pos)

    obj_pos = np.array([poses['traffic'][car]['pos'] for car in poses['traffic']])
    obj_yaw = np.array([poses['traffic'][car]['yaw'] for car in poses['traffic']])
    obj_rot = np.array([rotation_matrix(yaw) for yaw in obj_yaw])
    obj_rot = obj_rot @ np.array([[1, 0, 0],
                                  [0, 0, -1],
                                  [0, 1, 0]])

    obj_mat = np.array([get_mat(pos, rot) for pos, rot in zip(obj_pos, obj_rot)])

    cam_mat = np.array([[0, 0, 1, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, -3],
                        [0, 0, 0, 1]])

    cam_to_obj_mat = np.linalg.inv(obj_mat) @ ego_mat @ cam_mat
    cam_to_obj_trans = cam_to_obj_mat[:, :3, 3]
    cam_to_obj_rot = cam_to_obj_mat[:, :3, :3]

    cam_pos = torch.eye(4)[None, ...]
    cam_pos[:, 1, 1] = -1
    cam_pos[:, 2, 2] = -1

    render_rays = gen_rays(
        cam_pos, W, H,
        torch.tensor([FL_X, FL_Y]), 0, np.inf,
        torch.tensor([CX, CY])
    )[0].flatten(0, 1).numpy()

    ray_o = camera2object(render_rays[..., :3], cam_to_obj_trans, cam_to_obj_rot, obj_size, pos=True)
    ray_d = camera2object(render_rays[..., 3:6], cam_to_obj_trans, cam_to_obj_rot, obj_size, pos=False)

    z_in, z_out, intersect = ray_box_intersection(ray_o, ray_d)

    bounds = np.ones((*ray_o.shape[:-1], 2)) * -1
    bounds[intersect, 0] = z_in
    bounds[intersect, 1] = z_out

    cam_ray = np.concatenate([ray_o, ray_d, bounds], -1)
    cam_ray = torch.FloatTensor(cam_ray)

    intersect = np.any(intersect, 1)
    intersect = torch.from_numpy(intersect)
    cam_to_obj_trans = torch.Tensor(cam_to_obj_trans)
    cam_to_obj_rot = torch.Tensor(cam_to_obj_rot)
    obj_size = torch.Tensor(obj_size)
    return cam_ray, intersect, cam_to_obj_trans, cam_to_obj_rot, obj_size


def vis_scene(net, renderer, all_rays, rois, intersect, cam_to_obj_trans, cam_to_obj_rot, obj_size):
    net.eval()

    all_rays = all_rays.to(device)
    src_images = rois.to(device)
    intersect = intersect.to(device)
    cam_to_obj_trans = cam_to_obj_trans.to(device)
    cam_to_obj_rot = cam_to_obj_rot.to(device)
    obj_size = obj_size.to(device)

    all_rays = all_rays.view(H*W, -1, 8)
    valid_rays = all_rays[intersect, ...]

    _, Nb, _ = valid_rays.shape
    Nk = renderer.n_coarse

    with torch.no_grad():
        latents = net.encode(src_images)

        rgb_map = list()
        for batch_rays in tqdm.tqdm(torch.split(valid_rays, 1024)):

            rays = batch_rays.view(-1, 8)  # (N * B, 8)
            z_coarse = renderer.sample_from_ray(rays)
            empty_space = z_coarse == -1

            rgbs, sigmas = renderer.nerf_predict(net, rays, z_coarse, latents)

            pts_o = rays[:, None, :3] + z_coarse[:, :, None] * rays[:, None, 3:6]
            pts_o = pts_o.view(-1, Nb, Nk, 3).permute(1, 0, 2, 3).contiguous()
            pts_w = object2camera(pts_o.view(Nb, -1, 3), cam_to_obj_trans, cam_to_obj_rot, obj_size)
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

            rgb, depth, weights = renderer.volume_render(rgbs_sort, sigmas_sort, z_sort)

            rgb_map.append(rgb)

        rgb_map = torch.cat(rgb_map, 0)

        canvas = torch.zeros(H*W, 3).type_as(all_rays)
        canvas[intersect, :] = rgb_map
        canvas = (canvas.view(H, W, 3).cpu().numpy() * 255).astype(np.uint8)

        return canvas


import open3d as o3d

def get_mat_from_dict(pose):
    return get_mat(pose['pos'], rotation_matrix(pose['yaw']))


def debug_viz(poses):
    ego = o3d.geometry.TriangleMesh.create_box(width=4.5, height=2, depth=1.5)
    ego.paint_uniform_color([0., 0., 1.])
    ego.transform(get_mat_from_dict(poses['ego']))

    car_meshes = []
    for car in poses['traffic']:
        car_mesh = o3d.geometry.TriangleMesh.create_box(width=4.5, height=2, depth=1.5)
        car_mesh.paint_uniform_color([1., 0., 0.])
        car_mesh.transform(get_mat_from_dict(poses['traffic'][car]))
        car_meshes.append(car_mesh)

    K = np.array([[FL_X, 0, CX], [0, FL_Y, CY], [0, 0, 1]])

    ego_mat = get_mat_from_dict(poses['ego'])

    cam_mat = np.array([[0, 0, 1, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]])

    camera = o3d.geometry.LineSet.create_camera_visualization(W, H, intrinsic=K, extrinsic=np.linalg.inv(ego_mat @ cam_mat))

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([mesh_frame, *car_meshes, camera])


DATA_DIR = '/home/liqi/data/KITTI'

import torchvision.transforms as T
import cv2

img_transform = T.Compose([T.ToTensor(), T.Resize((128, 128)), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def get_kitti_img(filelist, idx):
    idx = filelist[idx]

    img = cv2.imread(f'{DATA_DIR}/training/nerf/%s_patch.png' % idx)
    img = img_transform(img).unsqueeze(0)

    with open(f'{DATA_DIR}/training/nerf/%s_label.txt' % idx, 'r') as f:
        obj = f.readlines()[0].split()

    _, _, _, _, _, _, _, dx, dy, dz, _ = [float(a) for a in obj]

    return img, dx, dy, dz

def get_img(filelist, env):
    poses = get_car_poses(env)
    imgs = []
    obj_size = []
    for i, pose in enumerate(poses['traffic']):
        img, x, y, z = get_kitti_img(filelist, i)
        imgs.append(img)
        obj_size.append([x, z, y])
    imgs = torch.cat(imgs, 0)
    obj_size = np.array(obj_size)
    return imgs, obj_size


def main():
    env = MetaDriveEnv(dict(
        environment_num=1000,
        start_seed=1010,
        traffic_density=0.5,
    ))

    net = PixelNeRFNet().to(device=device)
    net.load_state_dict(torch.load('output/200.ckpt'))
    renderer = NeRFRenderer().to(device=device)

    filelist = [f[:-10] for f in os.listdir(f'{DATA_DIR}/training/nerf/') if "label" in f ]
    filelist.sort()

    o = env.reset()
    imgs, obj_size = get_img(filelist, env)

    with imageio.get_writer('scene.gif', mode='I', duration=.5) as writer:
        for t in range(600):
            o, r, d, i = env.step([0., .5])
            if d:
                o = env.reset()
                imgs, obj_size = get_img(filelist, env)
            if t % 10 == 0:
                cam_ray, intersect, cam_to_obj_trans, cam_to_obj_rot, obj_size_torch = get_rays(env, obj_size)
                canvas = vis_scene(net, renderer, cam_ray, imgs, intersect, cam_to_obj_trans, cam_to_obj_rot, obj_size_torch)
                canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                writer.append_data(canvas)


if __name__ == '__main__':
    main()
