import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2


def load_rgbd_image_pair(filename: str, directory: str):
    color_image = cv2.imread(f"{directory}/{filename}.color.png")
    depth_image = cv2.imread(f"{directory}/{filename}.depth.png", cv2.IMREAD_UNCHANGED)
    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    return color_image, depth_image


def load_rgbd_image(filename: str, directory: str, depth_exp: float = 1.0):
    color_image = cv2.imread(f"{directory}/{filename}.color.png")
    depth_image = cv2.imread(f"{directory}/{filename}.depth.png", cv2.IMREAD_UNCHANGED)
    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)

    if depth_exp != 1.0:
        print(f"Applying depth exponent of {depth_exp}")
        depth_type = depth_image.dtype
        if depth_type == np.uint16:
            depth_norm = depth_image / 65535.0
        else:
            depth_norm = depth_image / 255.0
        depth_image = np.power(depth_norm, depth_exp)
        if depth_type == np.uint16:
            depth_image = (depth_image * 65535).astype(np.uint16)
        else:
            depth_image = (depth_image * 255).astype(np.uint8)

    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    color_raw = o3d.geometry.Image(color_image)
    depth_raw = o3d.geometry.Image(depth_image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, depth_trunc=100000000, convert_rgb_to_intensity=False
    )
    return rgbd_image


def rgbd_to_points_orthographic(
    color_image: np.ndarray,
    depth_image: np.ndarray,
    xy_scale: float = 0.001,
    z_scale: float = 1.0,
):
    assert color_image.shape[:2] == depth_image.shape[:2]
    points = []
    colors = []
    max_val = 255
    if depth_image.dtype == np.uint16:
        max_val = 65535

    depth_values = depth_image.astype(np.float32) / max_val
    depth_values = 1 - depth_values

    # Create a grid of coordinates
    x = np.linspace(
        0,
        depth_values.shape[0],
        depth_values.shape[0],
        endpoint=False,
        dtype=np.float32,
    )
    y = np.linspace(
        0,
        depth_values.shape[1],
        depth_values.shape[1],
        endpoint=False,
        dtype=np.float32,
    )
    x *= xy_scale
    y *= xy_scale
    xx, yy = np.meshgrid(x, y, indexing="ij")
    zz = depth_values * z_scale

    # Stack the coordinates
    points = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)

    # Normalize and reshape color image
    colors = (color_image / 255).reshape(-1, 3)

    manual_pcd = o3d.geometry.PointCloud()

    manual_pcd.points = o3d.utility.Vector3dVector(points)
    manual_pcd.colors = o3d.utility.Vector3dVector(colors)
    return manual_pcd


def rgbd_to_points(
    rgbd_image: o3d.cpu.pybind.geometry.RGBDImage,
    focal_point_scalar=1.0,
    fx_frac=None,
    fy_frac=None,
):
    width, height = np.asarray(rgbd_image.depth).shape
    fx = width * focal_point_scalar if fx_frac is None else fx_frac * width
    fy = height * focal_point_scalar if fy_frac is None else fy_frac * height 
    cx, cy = width / 2, height / 2
    intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
    # flip the point cloud so it's right side up
    point_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return point_cloud


if __name__ == "__main__":
    filename = "kaliane-sunglasses"
    directory = "output"

    # color_image, depth_image = load_rgbd_image_pair(filename, directory)
    rgbd_image = load_rgbd_image(filename, directory, 1.0)
    pcd = rgbd_to_points(rgbd_image, 1.0)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    inlier_cloud = pcd.select_by_index(ind)

    o3d.visualization.draw_geometries([inlier_cloud])
    print("Done!")
