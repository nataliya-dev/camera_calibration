#!/usr/bin/env python3
"""
Extract camera intrinsics from DepthAI device and save to JSON file.
Uses the factory calibration data stored on the device.
"""

import depthai as dai
import json
import numpy as np
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Camera socket to extract calibration for
# CAM_A (0) = RGB camera (usually 4K)
# CAM_B (1) = Left stereo camera
# CAM_C (2) = Right stereo camera
CAMERA_SOCKET = dai.CameraBoardSocket.CAM_A

# Target resolution (width, height)
# The intrinsics will be scaled to this resolution
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080

# Output file
OUTPUT_FILE = "camera_intrinsics.json"

# ============================================================================
# FUNCTIONS
# ============================================================================


def get_camera_intrinsics(device, camera_socket, target_width, target_height):
    """
    Extract camera intrinsics from device calibration data.

    Args:
        device: DepthAI device
        camera_socket: Camera socket (CAM_A, CAM_B, CAM_C)
        target_width: Target image width
        target_height: Target image height

    Returns:
        Dictionary containing intrinsics data
    """
    # Read calibration from device
    calib = device.readCalibration2()

    # Get EEPROM data
    eeprom_data = calib.eepromToJson()

    # Find camera data for the specified socket
    socket_number = camera_socket.value if hasattr(
        camera_socket, 'value') else int(camera_socket)
    camera_data = None

    for cam in eeprom_data.get('cameraData', []):
        if cam[0] == socket_number:
            camera_data = cam[1]
            break

    if camera_data is None:
        raise ValueError(
            f"No calibration data found for camera socket {socket_number}")

    # Extract native resolution from camera data
    native_width = camera_data['width']
    native_height = camera_data['height']

    # Extract intrinsic matrix
    intrinsic_matrix = camera_data['intrinsicMatrix']
    native_fx = intrinsic_matrix[0][0]
    native_fy = intrinsic_matrix[1][1]
    native_cx = intrinsic_matrix[0][2]
    native_cy = intrinsic_matrix[1][2]

    # Extract distortion coefficients
    dist_coeffs = camera_data['distortionCoeff']

    # Calculate scaling factors
    scale_x = target_width / native_width
    scale_y = target_height / native_height

    # Scale intrinsic parameters
    fx = native_fx * scale_x
    fy = native_fy * scale_y
    cx = native_cx * scale_x
    cy = native_cy * scale_y

    # Create scaled camera matrix
    camera_matrix = [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ]

    # Prepare output data
    intrinsics_data = {
        'camera_socket': socket_number,
        'camera_socket_name': str(camera_socket),
        'native_resolution': {
            'width': int(native_width),
            'height': int(native_height)
        },
        'target_resolution': {
            'width': target_width,
            'height': target_height
        },
        'camera_matrix': camera_matrix,
        'distortion_coefficients': dist_coeffs,
        'intrinsics': {
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy
        },
        'fov_deg': camera_data.get('specHfovDeg'),
        'lens_position': camera_data.get('lensPosition'),
        'camera_type': camera_data.get('cameraType'),
        'device_info': {
            'product_name': eeprom_data.get('productName', 'Unknown'),
            'board_name': eeprom_data.get('boardName', 'Unknown'),
        },
        'timestamp': datetime.now().isoformat(),
        'source': 'factory_calibration'
    }

    return intrinsics_data


def print_intrinsics_summary(data):
    """Print a summary of the intrinsics data."""
    print("\n" + "="*60)
    print("CAMERA INTRINSICS SUMMARY")
    print("="*60)
    print(f"Device: {data['device_info']['product_name']}")
    print(f"Camera Socket: {data['camera_socket_name']}")
    print(
        f"\nNative Resolution: {data['native_resolution']['width']}x{data['native_resolution']['height']}")
    print(
        f"Target Resolution: {data['target_resolution']['width']}x{data['target_resolution']['height']}")
    print(f"\nIntrinsic Parameters:")
    print(f"  fx = {data['intrinsics']['fx']:.2f}")
    print(f"  fy = {data['intrinsics']['fy']:.2f}")
    print(f"  cx = {data['intrinsics']['cx']:.2f}")
    print(f"  cy = {data['intrinsics']['cy']:.2f}")
    if data['fov_deg']:
        print(f"  HFOV = {data['fov_deg']:.2f}°")
    print(
        f"\nDistortion Coefficients: {len(data['distortion_coefficients'])} parameters")
    print(f"  k1 = {data['distortion_coefficients'][0]:.6f}")
    print(f"  k2 = {data['distortion_coefficients'][1]:.6f}")
    print(f"  p1 = {data['distortion_coefficients'][2]:.6f}")
    print(f"  p2 = {data['distortion_coefficients'][3]:.6f}")
    if len(data['distortion_coefficients']) > 4:
        print(f"  k3 = {data['distortion_coefficients'][4]:.6f}")
    print("="*60 + "\n")


def main():
    """Main function to extract and save camera intrinsics."""

    print(f"Connecting to DepthAI device...")

    try:
        # Connect to device
        with dai.Device() as device:
            print(f"Connected to: {device.getDeviceName()}")

            # Check if EEPROM is available
            if not device.isEepromAvailable():
                print("Error: EEPROM not available on this device")
                return

            # Get intrinsics
            intrinsics_data = get_camera_intrinsics(
                device,
                CAMERA_SOCKET,
                TARGET_WIDTH,
                TARGET_HEIGHT
            )

            # Print summary
            print_intrinsics_summary(intrinsics_data)

            # Save to JSON file
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(intrinsics_data, f, indent=2)

            print(f"✓ Intrinsics saved to: {OUTPUT_FILE}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
