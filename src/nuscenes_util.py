
import numpy as np

import torch

from pyquaternion import Quaternion
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


def camera2object(pts, cam_to_obj_trans, cam_to_obj_rot, obj_size, pos=False):
    pts_o = np.einsum('BNi,Bi ->BN', Quaternion(cam_to_obj_rot).rotation_matrix[None], pts)
    if pos:
        pts_o += np.array(cam_to_obj_trans)[:3]
    pts_o = pts_o / (np.array(obj_size) / 2 + 1e-9)
    if not pos:
        pts_o = pts_o / np.linalg.norm(pts_o, axis=-1, keepdims=True)

    return pts_o


def object2camera(pts, cam_to_obj_trans, cam_to_obj_rot, obj_size):
    pts_w = pts * (obj_size / 2 + 1e-9)
    pts_w -= cam_to_obj_trans
    pts_w = torch.einsum('BNi,Bki ->BkN', torch.Tensor(Quaternion(cam_to_obj_rot.cpu().numpy()).rotation_matrix.T[None]).cuda(), pts_w)
    return pts_w


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



if __name__ == '__main__':
    pass
