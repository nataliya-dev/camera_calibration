#!/usr/bin/env python3
"""
Hand-Eye Calibration Data Collection Script
Collects robot poses and camera images for hand-eye calibration using franky control.
Supports both USB cameras and RealSense cameras.
"""

import os
import json
import time
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.spatial.transform import Rotation as R
from franky import *
# import pyrealsense2 as rs

import depthai as dai
import json
import numpy as np
from datetime import datetime

# =============================================================================
# CONFIGURATION PARAMETERS - MODIFY THESE AS NEEDED
# =============================================================================

# Robot connection
ROBOT_IP = "192.168.0.2"

# Data collection paths
DATA_DIR = "calibration_data"
EE_POSES_FILE = os.path.join(DATA_DIR, "ee_poses.json")

# Robot motion parameters
ROBOT_DYNAMICS_FACTOR = 0.05  # Reduce speed for safety
SETTLE_TIME = 1.0  # Time to wait after reaching each pose (seconds)

# Display configuration
SHOW_PREVIEW_WINDOWS = True  # Set to False for automated data collection
PREVIEW_RESIZE_FACTOR = 0.3   # Scale factor for preview windows (saves memory)

# Camera configuration dictionary
# Each camera can be either 'usb' or 'realsense' type
CAMERA_CONFIG = {
    # "hand_camera": {
    #     "type": "realsense",  # or "realsense"
    #     "id": '913522070103',  # USB camera ID or RealSense serial number
    #     "width": 1920,
    #     "height": 1080,
    #     "save_depth": False,  # Only applicable for RealSense
    #     "directory": "r1"
    # },
    # "external_camera": {
    #     "type": "realsense",  # or "realsense"
    #     "id": '943222071556',  # USB camera ID or RealSense serial number
    #     "width": 1920,
    #     "height": 1080,
    #     "save_depth": False,  # Only applicable for RealSense
    #     "directory": "r2"
    # }
    # Add more cameras as needed:
    # "camera_3": {
    #     "type": "realsense",
    #     "id": "943222071556",  # RealSense serial number
    #     "width": 1920,
    #     "height": 1080,
    #     "save_depth": True,
    #     "directory": "realsense_camera_3"
    # }

    "ext2": {
        "type": "depthai",  # "usb", "realsense", or "depthai"
        "id": dai.CameraBoardSocket.CAM_A,  # or CAM_B, CAM_C
        "width": 1920,
        "height": 1080,
        "directory": "ext2"
    },

}

# =============================================================================
# CAMERA WRAPPER CLASSES
# =============================================================================


class CameraBase:
    """Base class for camera wrappers"""

    def __init__(self, camera_id, width, height):
        self.camera_id = camera_id
        self.width = width
        self.height = height

    def initialize(self):
        raise NotImplementedError

    def capture(self):
        raise NotImplementedError

    def release(self):
        raise NotImplementedError


class DepthAICamera(CameraBase):
    """DepthAI/Luxonis Camera wrapper"""

    def __init__(self, camera_socket, width, height):
        super().__init__(camera_socket, width, height)
        self.camera_socket = camera_socket
        self.pipeline = None
        self.device = None
        self.video_queue = None

    def initialize(self):
        # Create pipeline
        self.pipeline = dai.Pipeline()

        # Create camera node
        cam = self.pipeline.create(dai.node.Camera).build()
        self.video_queue_name = "video"

        # Start pipeline and get queue
        self.device = dai.Device(self.pipeline)
        cam_output = cam.requestOutput((self.width, self.height))
        self.video_queue = cam_output.createOutputQueue()

        self.pipeline.start()
        print(
            f"DepthAI camera {self.camera_socket} - Resolution: {self.width}x{self.height}")

    def capture(self):
        """Returns dict with 'color' key containing BGR image"""
        video_in = self.video_queue.get()
        if not isinstance(video_in, dai.ImgFrame):
            raise RuntimeError(f"Failed to capture from DepthAI camera")

        frame = video_in.getCvFrame()
        return {"color": frame}

    def release(self):
        if self.device:
            self.device.close()


class USBCamera(CameraBase):
    """USB Camera wrapper"""

    def __init__(self, camera_id, width, height):
        super().__init__(camera_id, width, height)
        self.camera = None

    def initialize(self):
        self.camera = cv2.VideoCapture(self.camera_id)
        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open USB camera {self.camera_id}")

        # Configure camera
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.camera.set(cv2.CAP_PROP_FPS, 30)

        # Verify resolution
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(
            f"USB camera {self.camera_id} - Requested: {self.width}x{self.height}, Actual: {actual_width}x{actual_height}")

        if actual_width != self.width or actual_height != self.height:
            print(
                f"⚠ Warning: USB camera {self.camera_id} resolution differs from requested!")

    def capture(self):
        """Returns dict with 'color' key containing BGR image"""
        ret, frame = self.camera.read()
        ret, frame = self.camera.read()  # Double read for stability
        if not ret:
            raise RuntimeError(
                f"Failed to capture from USB camera {self.camera_id}")
        return {"color": frame}

    def release(self):
        if self.camera:
            self.camera.release()


class RealSenseCamera(CameraBase):
    """RealSense Camera wrapper"""

    def __init__(self, serial_number, width, height, save_depth=False):
        super().__init__(serial_number, width, height)
        self.serial_number = str(serial_number)
        self.save_depth = save_depth
        self.pipeline = None

    def initialize(self):

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial_number)
        config.enable_stream(rs.stream.color, self.width,
                             self.height, rs.format.bgr8, 30)

        # if self.save_depth:
        #     config.enable_stream(rs.stream.depth, self.width,
        #                          self.height, rs.format.z16, 30)

        self.pipeline.start(config)
        print(
            f"RealSense camera {self.serial_number} - Resolution: {self.width}x{self.height}, Depth: {self.save_depth}")

    def capture(self):
        """Returns dict with 'color' and optionally 'depth' keys"""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            raise RuntimeError(
                f"Failed to capture color from RealSense {self.serial_number}")

        result = {"color": np.asanyarray(color_frame.get_data())}

        # if self.save_depth:
        #     depth_frame = frames.get_depth_frame()
        #     if depth_frame:
        #         result["depth"] = np.asanyarray(depth_frame.get_data())
        #     else:
        #         print(
        #             f"Warning: Failed to capture depth from RealSense {self.serial_number}")

        return result

    def release(self):
        if self.pipeline:
            self.pipeline.stop()

# =============================================================================
# DATA COLLECTION CLASS
# =============================================================================


class CamRobotDataCollector:
    """Collects calibration data for hand-eye calibration"""

    def __init__(self):
        self.robot = None
        self.cameras = {}
        self.camera_directories = {}
        self.start_pose = None
        self.collected_data = {}

    def initialize_robot(self):
        """Initialize robot connection"""
        print(f"Connecting to robot at {ROBOT_IP}...")
        self.robot = Robot(ROBOT_IP)
        self.robot.recover_from_errors()
        self.robot.relative_dynamics_factor = ROBOT_DYNAMICS_FACTOR

        home = JointMotion([8.90512496e-01, -3.02871668e-01, -2.54106958e-04, -2.59768526e+00,
                            -9.54882244e-01,  1.01583728e+00,  1.67246430e+00])

        self.robot.move(home)

        self.start_pose = self.robot.current_cartesian_state.pose.end_effector_pose
        print(f"Starting pose recorded:")
        print(f"Position: {self.start_pose.translation}")
        print(f"Quaternion: {self.start_pose.quaternion}")

    def initialize_cameras(self):
        """Initialize all configured cameras"""
        print("Initializing cameras...")

        for camera_name, config in CAMERA_CONFIG.items():
            print(f"Setting up {camera_name} ({config['type']})...")

            # Create camera directory
            camera_dir = os.path.join(DATA_DIR, config["directory"])
            os.makedirs(camera_dir, exist_ok=True)
            self.camera_directories[camera_name] = camera_dir

            # Create appropriate camera wrapper
            if config["type"] == "usb":
                camera = USBCamera(
                    config["id"], config["width"], config["height"])
            elif config["type"] == "realsense":
                camera = RealSenseCamera(
                    config["id"],
                    config["width"],
                    config["height"],
                    config.get("save_depth", False)
                )
            elif config["type"] == "depthai":
                camera = DepthAICamera(
                    config["id"],
                    config["width"],
                    config["height"]
                )
            else:
                raise ValueError(f"Unknown camera type: {config['type']}")

            # Initialize camera
            camera.initialize()
            self.cameras[camera_name] = camera

        print(f"Successfully initialized {len(self.cameras)} cameras")

    def generate_poses(self) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        poses = []

        # Ensure start pose is recorded
        if self.start_pose is None:
            self.start_pose = self.robot.current_cartesian_state.pose.end_effector_pose
            print(f"Starting pose recorded:")
            print(f"Position: {self.start_pose.translation}")
            print(f"Quaternion: {self.start_pose.quaternion}")

        print("Generating 20 target poses...")

        # Extract start pose components
        start_position = np.array(self.start_pose.translation)
        start_quaternion = np.array(
            self.start_pose.quaternion)  # [x, y, z, w] format
        start_rotation = R.from_quat(start_quaternion)

        sample_id = 0

        # Z-axis translations
        for z_offset in [0.1, 0.08, 0.05, 0.025, -0.025, -0.05, -0.08, -0.1]:
            pos = start_position + np.array([0, 0, z_offset])
            poses.append((pos, start_quaternion, f"sample_{sample_id:04d}"))
            sample_id += 1

        # X-axis translations
        for x_offset in [0.05, 0.025, -0.025, -0.05]:
            pos = start_position + np.array([x_offset, 0, 0])
            poses.append((pos, start_quaternion, f"sample_{sample_id:04d}"))
            sample_id += 1

        # Y-axis translations
        for y_offset in [00.1, 0.08, 0.05, 0.025, -0.025, -0.05, -0.08, -0.1]:
            pos = start_position + np.array([0, y_offset, 0])
            poses.append((pos, start_quaternion, f"sample_{sample_id:04d}"))
            sample_id += 1

        # Rotation variations
        rotation_variations = [
            # roll, pitch, yaw in degrees
            (5, 0, 0), (-5, 0, 0), (8, 0, 0), (-8, 0, 0),  # roll
            (0, 5, 0), (0, -5, 0), (0, 8, 0), (0, -8, 0),  # pitch
            (0, 0, 5), (0, 0, -5), (0, 0, 8), (0, 0, -8),   # yaw

            (10, 0, 0), (-10, 0, 0), (13, 0, 0), (-13, 0, 0),  # roll
            (0, 10, 0), (0, -10, 0), (0, 13, 0), (0, -13, 0),  # pitch
            (0, 0, 10), (0, 0, -10), (0, 0, 13), (0, 0, -13)   # yaw

        ]

        for roll, pitch, yaw in rotation_variations:
            relative_rotation = R.from_euler(
                'xyz', [roll, pitch, yaw], degrees=True)
            global_rotation = start_rotation * relative_rotation
            global_quaternion = global_rotation.as_quat()
            poses.append((start_position, global_quaternion,
                         f"sample_{sample_id:04d}"))
            sample_id += 1

        # Random poses - define your workspace limits here
        position_limits = {
            'x': (-0.1, 0.2),  # ±15cm from start position
            'y': (-0.15, 0.15),  # ±15cm from start position
            'z': (-0.12, 0.12)   # ±12cm from start position
        }

        rotation_limits = {
            'roll': (-20, 20),   # ±20 degrees
            'pitch': (-20, 20),  # ±20 degrees
            'yaw': (-20, 20)     # ±20 degrees
        }

        num_random_poses = 10  # Adjust as needed

        print(f"Generating {num_random_poses} random poses...")

        # Set random seed for reproducibility (optional)
        np.random.seed(42)

        for i in range(num_random_poses):
            # Random position offset
            x_offset = np.random.uniform(
                position_limits['x'][0], position_limits['x'][1])
            y_offset = np.random.uniform(
                position_limits['y'][0], position_limits['y'][1])
            z_offset = np.random.uniform(
                position_limits['z'][0], position_limits['z'][1])

            random_position = start_position + \
                np.array([x_offset, y_offset, z_offset])

            # Random orientation
            roll = np.random.uniform(
                rotation_limits['roll'][0], rotation_limits['roll'][1])
            pitch = np.random.uniform(
                rotation_limits['pitch'][0], rotation_limits['pitch'][1])
            yaw = np.random.uniform(
                rotation_limits['yaw'][0], rotation_limits['yaw'][1])

            random_rotation = R.from_euler(
                'xyz', [roll, pitch, yaw], degrees=True)
            global_rotation = start_rotation * random_rotation
            random_quaternion = global_rotation.as_quat()

            poses.append((random_position, random_quaternion,
                         f"sample_{sample_id:04d}_random"))
            sample_id += 1

        print(f"Generated exactly {len(poses)} target poses")
        return poses

    def capture_images(self, sample_id: str) -> bool:
        """Capture images from all cameras"""
        try:
            for camera_name, camera in self.cameras.items():
                # Capture from camera
                captured_data = camera.capture()

                # Optional preview (can be disabled for automated collection)
                if SHOW_PREVIEW_WINDOWS:
                    preview_img = captured_data["color"]
                    # Resize for preview to save memory
                    if PREVIEW_RESIZE_FACTOR != 1.0:
                        new_size = (int(preview_img.shape[1] * PREVIEW_RESIZE_FACTOR),
                                    int(preview_img.shape[0] * PREVIEW_RESIZE_FACTOR))
                        preview_img = cv2.resize(preview_img, new_size)

                    cv2.imshow(f"{camera_name}_preview", preview_img)
                    cv2.waitKey(1)  # Non-blocking update

                # Save color image
                color_path = os.path.join(
                    self.camera_directories[camera_name], f"{sample_id}.jpg")
                jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
                success = cv2.imwrite(
                    color_path, captured_data["color"], jpeg_params)

                if not success:
                    print(f"Failed to save color image for {camera_name}")
                    return False

                # # Save depth image if available
                # if "depth" in captured_data:
                #     depth_path = os.path.join(
                #         self.camera_directories[camera_name], f"{sample_id}_depth.png")
                #     success = cv2.imwrite(depth_path, captured_data["depth"])

                #     if not success:
                #         print(f"Failed to save depth image for {camera_name}")
                #         return False

                print(f"Images saved for {camera_name} - {sample_id}")

            return True

        except Exception as e:
            print(f"Error capturing images for {sample_id}: {e}")
            return False

    def move_to_pose(self, position: np.ndarray, quaternion: np.ndarray) -> bool:
        """Move robot to target pose"""
        try:
            transform = Affine(position.tolist(), quaternion.tolist())
            motion = CartesianMotion(transform)

            print(f"Moving to pose: pos={position}, quat={quaternion}")
            self.robot.move(motion)
            time.sleep(SETTLE_TIME)

            # Verify we reached the target pose
            current_state = self.robot.current_cartesian_state
            current_pose = current_state.pose.end_effector_pose
            pos_error = np.linalg.norm(
                np.array(current_pose.translation) - position)

            if pos_error > 0.01:  # 1cm tolerance
                print(
                    f"Warning: Position error {pos_error:.4f}m - waiting additional time")
                time.sleep(1.0)
            else:
                print(f"Pose reached. Position error: {pos_error:.4f}m")

            return True

        except Exception as e:
            print(f"Error moving to pose: {e}")
            return False

    def collect_sample(self, position: np.ndarray, quaternion: np.ndarray, sample_id: str) -> bool:
        """Collect a single calibration sample"""
        print(f"\nCollecting sample {sample_id}...")

        # Move to target pose
        if not self.move_to_pose(position, quaternion):
            return False

        # Get current robot state
        current_state = self.robot.current_cartesian_state
        ee_pose = current_state.pose.end_effector_pose

        # Store pose data
        pose_data = {
            "position": ee_pose.translation.tolist(),
            "orientation": ee_pose.quaternion.tolist()  # [x, y, z, w]
        }

        # Capture images
        if not self.capture_images(sample_id):
            return False

        # Store data
        self.collected_data[sample_id] = pose_data
        print(f"Sample {sample_id} collected successfully")
        return True

    def return_to_start(self):
        """Return robot to starting pose"""
        print("\nReturning to starting pose...")
        try:
            motion = CartesianMotion(self.start_pose)
            self.robot.move(motion)
            time.sleep(SETTLE_TIME)
            print("Returned to starting pose")
        except Exception as e:
            print(f"Error returning to start: {e}")

    def save_data(self):
        """Save collected pose data to JSON file"""
        print(f"\nSaving pose data to {EE_POSES_FILE}...")
        with open(EE_POSES_FILE, 'w') as f:
            json.dump(self.collected_data, f, indent=2)
        print(f"Saved {len(self.collected_data)} samples")

    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up cameras...")
        for camera_name, camera in self.cameras.items():
            camera.release()
            print(f"{camera_name} released")
        cv2.destroyAllWindows()
        print("Camera cleanup complete")

    def create_directories(self):
        """Create necessary directories"""
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"Created data directory: {DATA_DIR}")

    def collect_all_data(self):
        """Main data collection loop"""
        try:
            # Setup
            self.create_directories()
            self.initialize_robot()
            self.initialize_cameras()

            # Generate poses
            poses = self.generate_poses()

            print(f"\nStarting data collection for {len(poses)} poses...")
            input("Press Enter to start data collection...")

            # Collect data
            successful_samples = 0
            failed_samples = 0

            for i, (pos, quat, sample_id) in enumerate(poses):
                print(f"\nProgress: {i+1}/{len(poses)}")

                if self.collect_sample(pos, quat, sample_id):
                    successful_samples += 1
                else:
                    failed_samples += 1
                    print(f"Failed to collect sample {sample_id}")

            # Save data
            if successful_samples > 0:
                self.save_data()

            # Return to start
            self.return_to_start()

            # Summary
            print(f"\n=== DATA COLLECTION SUMMARY ===")
            print(f"Successful samples: {successful_samples}")
            print(f"Failed samples: {failed_samples}")
            print(f"Total poses attempted: {len(poses)}")
            print(f"Success rate: {successful_samples/len(poses)*100:.1f}%")

        except KeyboardInterrupt:
            print("\nData collection interrupted by user")
        except Exception as e:
            print(f"\nError during data collection: {e}")
        finally:
            self.cleanup()

# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """Main function"""
    print("=== Hand-Eye Calibration Data Collection ===")
    print(f"Robot IP: {ROBOT_IP}")
    print(f"Data directory: {DATA_DIR}")

    print("\nConfigured cameras:")
    for camera_name, config in CAMERA_CONFIG.items():
        print(
            f"  {camera_name}: {config['type']} (ID: {config['id']}, {config['width']}x{config['height']})")
        if config['type'] == 'realsense' and config.get('save_depth', False):
            print(f"    - Depth capture enabled")

    # Safety check
    print("\nSAFETY CHECKLIST:")
    print("1. ✓ Robot is powered on and FCI is enabled")
    print("2. ✓ Robot workspace is clear of obstacles")
    print("3. ✓ Chess calibration board is positioned and visible")
    print("4. ✓ All cameras are connected and working")
    print("5. ✓ Emergency stop is accessible")
    print()

    response = input("All safety checks complete? (y/n): ")
    if response.lower() != 'y':
        print("Please complete safety checks before proceeding.")
        return

    # Start data collection
    collector = CamRobotDataCollector()
    collector.collect_all_data()


if __name__ == "__main__":
    main()
