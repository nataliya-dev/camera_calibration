import pyrealsense2 as rs
import yaml
import os


def get_camera_intrinsics_yaml(device_serials):
    """
    Get intrinsics parameters from RealSense cameras and format as YAML

    Args:
        device_serials: List of camera serial numbers

    Returns:
        Dictionary containing camera intrinsics in YAML format
    """

    ctx = rs.context()
    devices = ctx.query_devices()

    print("Available devices:")
    for device in devices:
        print(
            f"Device: {device.get_info(rs.camera_info.name)} - Serial: {device.get_info(rs.camera_info.serial_number)}")

    camera_data = {}

    for serial in device_serials:
        try:
            # Create pipeline for specific device
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 1920,
                                 1080, rs.format.bgr8, 30)

            # Start pipeline
            profile = pipeline.start(config)

            # Get color stream intrinsics
            color_stream = profile.get_stream(rs.stream.color)
            color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            print(color_intrinsics)

            # Format intrinsics data
            camera_intrinsics = {
                'fx': float(color_intrinsics.fx),
                'fy': float(color_intrinsics.fy),
                'cx': float(color_intrinsics.ppx),
                'cy': float(color_intrinsics.ppy),
                'has_dist_coeff': 1,
                'dist_k0': float(color_intrinsics.coeffs[0]) if len(color_intrinsics.coeffs) > 0 else 0.0,
                'dist_k1': float(color_intrinsics.coeffs[1]) if len(color_intrinsics.coeffs) > 1 else 0.0,
                'dist_px': float(color_intrinsics.coeffs[2]) if len(color_intrinsics.coeffs) > 2 else 0.0,
                'dist_py': float(color_intrinsics.coeffs[3]) if len(color_intrinsics.coeffs) > 3 else 0.0,
                'dist_k2': float(color_intrinsics.coeffs[4]) if len(color_intrinsics.coeffs) > 4 else 0.0,
                'dist_k3': 0.0,  # Usually not used in RealSense
                'dist_k4': 0.0,  # Usually not used in RealSense
                'dist_k5': 0.0,  # Usually not used in RealSense
                'img_width': color_intrinsics.width,
                'img_height': color_intrinsics.height
            }

            camera_data[f'camera_{serial}'] = camera_intrinsics

            print(f"\nIntrinsics for camera {serial}:")
            print(f"fx: {color_intrinsics.fx}")
            print(f"fy: {color_intrinsics.fy}")
            print(f"cx: {color_intrinsics.ppx}")
            print(f"cy: {color_intrinsics.ppy}")
            print(f"Distortion coefficients: {color_intrinsics.coeffs}")
            print(
                f"Resolution: {color_intrinsics.width}x{color_intrinsics.height}")

            # Stop pipeline
            pipeline.stop()

        except Exception as e:
            print(f"Error accessing camera {serial}: {e}")
            continue

    return camera_data


def create_output_folder(folder_name="intrinsics"):
    """Create output folder if it doesn't exist"""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")
    return folder_name


def save_intrinsics_to_yaml(camera_data, filename="camera_intrinsics.yaml", folder="intrinsics"):
    """Save camera intrinsics to YAML file in specified folder"""
    create_output_folder(folder)
    filepath = os.path.join(folder, filename)
    with open(filepath, 'w') as file:
        yaml.dump(camera_data, file, default_flow_style=False, sort_keys=False)
    print(f"\nIntrinsics saved to {filepath}")


def save_individual_camera_yaml(camera_data, folder="intrinsics"):
    """Save each camera's intrinsics to separate YAML files in specified folder"""
    create_output_folder(folder)
    for camera_name, intrinsics in camera_data.items():
        filename = f"{camera_name}_intrinsics.yaml"
        filepath = os.path.join(folder, filename)
        individual_data = {camera_name: intrinsics}
        with open(filepath, 'w') as file:
            yaml.dump(individual_data, file,
                      default_flow_style=False, sort_keys=False)
        print(f"Individual intrinsics saved to {filepath}")


def print_yaml_format(camera_data):
    """Print intrinsics in YAML format"""
    print("\n" + "="*50)
    print("CAMERA INTRINSICS IN YAML FORMAT:")
    print("="*50)

    for camera_name, intrinsics in camera_data.items():
        print(f"\n{camera_name}:")
        for key, value in intrinsics.items():
            if isinstance(value, float):
                if abs(value) < 1e-6 and value != 0:
                    print(f"  {key}: {value:.2e}")
                else:
                    print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")


# Main execution
if __name__ == "__main__":
    # Specify your camera serial numbers
    device_serials = ['913522070103', '943222071556']

    # Get intrinsics from cameras
    camera_intrinsics = get_camera_intrinsics_yaml(device_serials)

    if camera_intrinsics:
        # Print in YAML format
        print_yaml_format(camera_intrinsics)

        # Save all cameras to one file
        save_intrinsics_to_yaml(
            camera_intrinsics, "all_cameras_intrinsics.yaml")

        # Save each camera to individual files
        save_individual_camera_yaml(camera_intrinsics)

        # Also create a simplified version matching your example format
        print("\n" + "="*50)
        print("SIMPLIFIED FORMAT (like your example):")
        print("="*50)

        for camera_name, data in camera_intrinsics.items():
            print(f"\n# {camera_name}")
            print(data)
            print(f"fx: {data['fx']} fy: {data['fy']} cx: {data['cx']} cy: {data['cy']} "
                  f"has_dist_coeff: {data['has_dist_coeff']} "
                  f"dist_k0: {data['dist_k0']:.0e} dist_k1: {data['dist_k1']:.0e} "
                  f"dist_px: {data['dist_px']:.0e} dist_py: {data['dist_py']:.0e} "
                  f"dist_k2: {data['dist_k2']:.0e} dist_k3: {data['dist_k3']} "
                  f"dist_k4: {data['dist_k4']} dist_k5: {data['dist_k5']} "
                  f"img_width: {data['img_width']} img_height: {data['img_height']}")
    else:
        print("No camera intrinsics retrieved!")
