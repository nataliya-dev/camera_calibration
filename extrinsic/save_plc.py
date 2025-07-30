import pyrealsense2 as rs
import numpy as np
import open3d as o3d

# Device serial numbers
device_serials = ['913522070103', '943222071556']


def capture_pointcloud():
    pipelines = []
    configs = []

    # Initialize pipelines for each camera
    for serial in device_serials:
        pipeline = rs.pipeline()
        config = rs.config()

        # Enable device by serial number
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 15)

        pipelines.append(pipeline)
        configs.append(config)

        # Start streaming
        pipeline.start(config)
        print(f"Started camera {serial}")

    try:
        # Wait for frames to stabilize
        for _ in range(30):
            for pipeline in pipelines:
                pipeline.wait_for_frames()

        # Capture frames from all cameras
        for i, (pipeline, serial) in enumerate(zip(pipelines, device_serials)):
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                print(f"Failed to get frames from camera {serial}")
                continue

            # Get camera intrinsics
            depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

            # Convert depth frame to numpy array
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Create point cloud
            pc = rs.pointcloud()
            pc.map_to(color_frame)
            points = pc.calculate(depth_frame)

            # Get vertices and texture coordinates
            vertices = np.asanyarray(points.get_vertices()).view(
                np.float32).reshape(-1, 3)
            colors = np.asanyarray(points.get_texture_coordinates()).view(
                np.float32).reshape(-1, 2)

            # Convert color image to RGB and normalize
            color_rgb = color_image[:, :, ::-1]  # BGR to RGB

            # Map texture coordinates to colors
            h, w = color_rgb.shape[:2]
            # Remove points with zero depth
            valid_points = (vertices[:, 2] > 0)

            if np.any(valid_points):
                # Create Open3D point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(vertices[valid_points])

                # Map colors using texture coordinates
                tex_coords = colors[valid_points]
                tex_x = np.clip((tex_coords[:, 0] * w).astype(int), 0, w-1)
                tex_y = np.clip((tex_coords[:, 1] * h).astype(int), 0, h-1)
                point_colors = color_rgb[tex_y, tex_x] / 255.0
                pcd.colors = o3d.utility.Vector3dVector(point_colors)

                # Save point cloud
                filename = f"pointcloud_camera_{serial}.ply"
                o3d.io.write_point_cloud(filename, pcd)
                print(f"Saved point cloud from camera {serial} to {filename}")
                print(f"Point cloud contains {len(pcd.points)} points")
            else:
                print(f"No valid points found for camera {serial}")

    finally:
        # Stop all pipelines
        for pipeline in pipelines:
            pipeline.stop()
        print("All cameras stopped")


if __name__ == "__main__":
    capture_pointcloud()
