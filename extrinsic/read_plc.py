import open3d as o3d

pcd = o3d.io.read_point_cloud("pointcloud_camera_913522070103.ply")
o3d.visualization.draw_geometries([pcd])


pcd = o3d.io.read_point_cloud("pointcloud_camera_943222071556.ply")
o3d.visualization.draw_geometries([pcd])
