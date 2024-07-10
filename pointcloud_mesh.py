import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2


if __name__ == "__main__":
    filename = "nate"
    directory = "output"

    color_image = cv2.imread(f"{directory}/{filename}.color.png")
    depth_image = cv2.imread(f"{directory}/{filename}.depth.png", cv2.IMREAD_UNCHANGED)
    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    color_raw = o3d.geometry.Image(color_image)
    depth_raw = o3d.geometry.Image(depth_image)

    width, height = depth_image.shape[1], depth_image.shape[0]
    fx, fy = width, height  # Start with these values and adjust
    cx, cy = width / 2, height / 2  # Center of the image
    intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    depth_scale = 10000.0  # Adjust as per your depth scale
    depth_trunc = 10000000.0

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw,
        depth_raw,
        depth_scale=depth_scale,  # Adjust this depending on your depth image format
        depth_trunc=depth_trunc,  # Truncate depth values greater than this value
        convert_rgb_to_intensity=False,
    )

    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    point_cloud = point_cloud.select_by_index(ind)

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            point_cloud, depth=9
        )

    o3d.visualization.draw_geometries([mesh],
                                  zoom=0.664,
                                  front=[-0.4761, -0.4698, -0.7434],
                                  lookat=[1.8900, 3.2596, 0.9284],
                                  up=[0.2304, -0.8825, 0.4101])