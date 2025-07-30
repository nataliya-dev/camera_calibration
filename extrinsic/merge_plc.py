import open3d as o3d
import numpy as np
import json
import copy


def load_pose_data(json_file):
    """Load camera pose data from JSON file"""
    with open(json_file, 'r') as f:
        pose_data = json.load(f)
    return pose_data


def create_transformation_matrix(pose_data):
    """Create 4x4 transformation matrix from pose data"""
    # Extract rotation matrix and position
    rmat = np.array(pose_data['rmat'])
    position = np.array(pose_data['position'])

    # Create 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rmat
    transform[:3, 3] = position

    return transform


def apply_camera_pose_to_pointcloud(pcd, pose_data):
    """Apply camera pose transformation to point cloud"""
    # Create transformation matrix
    transform = create_transformation_matrix(pose_data)
    transform = np.linalg.inv(transform)

    # Apply transformation to point cloud
    pcd_transformed = copy.deepcopy(pcd)
    pcd_transformed.transform(transform)

    return pcd_transformed


def main():
    # Load point clouds
    print("Loading point clouds...")
    pcd1 = o3d.io.read_point_cloud("pointcloud_camera_913522070103.ply")
    pcd2 = o3d.io.read_point_cloud("pointcloud_camera_943222071556.ply")

    # Load pose data
    print("Loading camera poses...")
    pose1 = load_pose_data("pose_data_r2.json")
    # Assuming the second pose file follows similar naming pattern
    pose2 = load_pose_data("pose_data_r1.json")  # Update filename as needed

    # Apply camera poses to transform point clouds to common frame
    print("Transforming point clouds to common frame...")
    pcd1_transformed = apply_camera_pose_to_pointcloud(pcd1, pose1)
    pcd2_transformed = apply_camera_pose_to_pointcloud(pcd2, pose2)

    # # Color the point clouds differently for visualization
    # pcd1_transformed.paint_uniform_color([1, 0, 0])  # Red
    # pcd2_transformed.paint_uniform_color([0, 0, 1])  # Blue

    # Optional: Create coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    # Visualize overlaid point clouds
    print("Visualizing overlaid point clouds...")
    o3d.visualization.draw_geometries([pcd1_transformed, pcd2_transformed, coord_frame],
                                      window_name="Overlaid Point Clouds",
                                      width=1024, height=768)

    # Optional: Save the combined point cloud
    combined_pcd = pcd1_transformed + pcd2_transformed
    o3d.io.write_point_cloud("combined_pointcloud.ply", combined_pcd)
    print("Combined point cloud saved as 'combined_pointcloud.ply'")


def visualize_camera_poses(pose1, pose2):
    """Visualize camera poses as coordinate frames"""
    # Create transformation matrices
    transform1 = create_transformation_matrix(pose1)
    transform2 = create_transformation_matrix(pose2)

    # Create coordinate frames for cameras
    camera1_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    camera2_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)

    # Apply transformations
    camera1_frame.transform(transform1)
    camera2_frame.transform(transform2)

    # World coordinate frame
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    # Visualize camera poses
    o3d.visualization.draw_geometries([camera1_frame, camera2_frame, world_frame],
                                      window_name="Camera Poses")

# Alternative function if you want to see camera poses first


def visualize_poses_only():
    """Visualize only the camera poses to verify alignment"""
    pose1 = load_pose_data("pose_data_r1.json")
    pose2 = load_pose_data("pose_data_r2.json")  # Update filename as needed
    visualize_camera_poses(pose1, pose2)


if __name__ == "__main__":
    main()

    # Uncomment to visualize camera poses only
    # visualize_poses_only()
