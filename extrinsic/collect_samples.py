#!/usr/bin/env python3
"""
Hand-Eye Calibration Data Collection Script
Collects robot poses and camera images for hand-eye calibration using franky control.
"""

import os
import json
import time
import cv2
import numpy as np
from typing import List, Tuple, Dict
from scipy.spatial.transform import Rotation as R
from franky import *

# =============================================================================
# CONFIGURATION PARAMETERS - MODIFY THESE AS NEEDED
# =============================================================================

# Robot connection
ROBOT_IP = "192.168.0.2"

# Data collection paths
DATA_DIR = "calibration_data"
EXTERNAL_1_IMAGES_DIR = os.path.join(DATA_DIR, "external_camera_1")
EXTERNAL_2_IMAGES_DIR = os.path.join(DATA_DIR, "external_camera_2")
EE_POSES_FILE = os.path.join(DATA_DIR, "ee_poses.json")

# Camera configuration
EXTERNAL_1_CAM_ID = 0      # USB camera ID for hand camera
EXTERNAL_2_CAM_ID = 2  # USB camera ID for external camera
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

# Robot motion parameters
ROBOT_DYNAMICS_FACTOR = 0.05  # Reduce speed for safety
SETTLE_TIME = 1.0  # Time to wait after reaching each pose (seconds)


# =============================================================================
# DATA COLLECTION CLASS
# =============================================================================

class CamRobotDataCollector:
    """Collects calibration data for hand-eye calibration"""

    def __init__(self):
        self.robot = None
        self.external_1_cam = None
        self.external_2_cam = None
        self.start_pose = None
        self.collected_data = {}

    def initialize_robot(self):
        """Initialize robot connection"""
        print(f"Connecting to robot at {ROBOT_IP}...")
        self.robot = Robot(ROBOT_IP)
        self.robot.recover_from_errors()
        self.robot.relative_dynamics_factor = ROBOT_DYNAMICS_FACTOR

        # home = JointMotion([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        home = JointMotion([9.86168202e-01, -1.26735375e+00, -9.55441349e-04, -2.85675190e+00,
                            -1.49717494e+00,  8.84785644e-01,  8.91454801e-01])

        self.robot.move(home)

        time.sleep(1)

        self.start_pose = self.robot.current_cartesian_state.pose.end_effector_pose
        print(f"Starting pose recorded:")
        print(f"Position: {self.start_pose.translation}")
        print(f"Quaternion: {self.start_pose.quaternion}")

    def initialize_cameras(self):
        """Initialize camera connections following calibration script pattern"""
        print("Initializing cameras...")

        # Hand camera (on robot end-effector)
        self.external_1_cam = cv2.VideoCapture(EXTERNAL_1_CAM_ID)
        if not self.external_1_cam.isOpened():
            raise RuntimeError(
                f"Failed to open camera {EXTERNAL_1_CAM_ID}")

        # External camera (static)
        self.external_2_cam = cv2.VideoCapture(EXTERNAL_2_CAM_ID)
        if not self.external_2_cam.isOpened():
            raise RuntimeError(
                f"Failed to open camera {EXTERNAL_2_CAM_ID}")

        # Configure both cameras with same settings
        for camera, camera_name in [(self.external_1_cam, "external_1"), (self.external_2_cam, "external_2")]:
            # Set resolution
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
            camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # disabled
            camera.set(cv2.CAP_PROP_FPS, 10)

            actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(
                f"{camera_name} camera - Requested: {IMAGE_WIDTH}x{IMAGE_HEIGHT}, Actual: {actual_width}x{actual_height}")

            if actual_width != IMAGE_WIDTH or actual_height != IMAGE_HEIGHT:
                print(
                    f"⚠ Warning: {camera_name} camera resolution differs from requested!")

        print("Cameras initialized successfully")

    def generate_poses(self) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """Generate exactly 20 target poses for data collection"""
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

        for z_offset in [0.05, 0.025, -0.025, -0.05]:
            pos = start_position + np.array([0, 0, z_offset])
            poses.append((pos, start_quaternion, f"sample_{sample_id:04d}"))
            sample_id += 1

        for x_offset in [0.05, 0.025, -0.025, -0.05]:
            pos = start_position + np.array([x_offset, 0, 0])
            poses.append((pos, start_quaternion, f"sample_{sample_id:04d}"))
            sample_id += 1

        for y_offset in [0.05, 0.025, -0.025, -0.05]:
            pos = start_position + np.array([0, y_offset, 0])
            poses.append((pos, start_quaternion, f"sample_{sample_id:04d}"))
            sample_id += 1

        rotation_variations = [
            # roll
            (5, 0, 0),
            (-5, 0, 0),
            (8, 0, 0),
            (-8, 0, 0),

            # pitch
            (0, 5, 0),
            (0, -5, 0),
            (0, 8, 0),
            (0, -8, 0),

            # yaw
            (0, 0, 5),
            (0, 0, -5),
            (0, 0, 8),
            (0, 0, -8)
        ]

        for roll, pitch, yaw in rotation_variations:
            relative_rotation = R.from_euler(
                'xyz', [roll, pitch, yaw], degrees=True)
            global_rotation = start_rotation * relative_rotation
            global_quaternion = global_rotation.as_quat()
            poses.append((start_position, global_quaternion,
                         f"sample_{sample_id:04d}"))
            sample_id += 1

        random_poses = [
            # Format: (x, y, z, roll, pitch, yaw)
            (0.03, 0.02, 0.03, 8, -5, 4),
            (-0.02, 0.04, -0.03, -9, 7, -4),
            (0.04, -0.03, 0.03, 5, 3, -5),
            (-0.03, -0.02, 0.03, -8, -8, 5),
            (0.0, -0.02, -0.01, -1, 5, 1),
            (0.0, -0.02, -0.01, -2, 1, 2),
            (0.01, -0.02, 0.01, -3, 2, -4),
            (0.01, -0.02, 0.01, -4, 0, -1),
        ]

        for x, y, z, roll, pitch, yaw in random_poses:
            pos = start_position + np.array([x, y, z])
            relative_rotation = R.from_euler(
                'xyz', [roll, pitch, yaw], degrees=True)

            global_rotation = start_rotation * relative_rotation
            global_quaternion = global_rotation.as_quat()

            poses.append((pos, global_quaternion, f"sample_{sample_id:04d}"))
            sample_id += 1

        print(f"Generated exactly {len(poses)} target poses")
        return poses

    def capture_images(self, sample_id: str) -> bool:
        """Capture images from both cameras with multiple attempts for stability"""
        try:
            ret, img1 = self.external_1_cam.read()
            ret, img1 = self.external_1_cam.read()
            cv2.imshow("Cam1", img1)  # Added window name
            cv2.waitKey(1)
            if not ret:
                print(f"Failed to capture image for {sample_id}")

            ret, img2 = self.external_2_cam.read()
            ret, img2 = self.external_2_cam.read()
            cv2.imshow("Cam2", img2)
            cv2.waitKey(1)
            if not ret:
                print(
                    f"Failed to capture image for {sample_id}")

            external_1_path = os.path.join(
                EXTERNAL_1_IMAGES_DIR, f"{sample_id}.jpg")
            external_2_path = os.path.join(
                EXTERNAL_2_IMAGES_DIR, f"{sample_id}.jpg")

            jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 95]

            success_1 = cv2.imwrite(
                external_1_path, img1, jpeg_params)
            success_2 = cv2.imwrite(
                external_2_path, img2, jpeg_params)

            if not success_1 or not success_2:
                print(f"Failed to save images for {sample_id}")
                return False

            print(
                f"Images saved for {sample_id} (img1: {img1.shape}, img2: {img2.shape})")
            return True

        except Exception as e:
            print(f"Error capturing images for {sample_id}: {e}")
            return False

    def move_to_pose(self, position: np.ndarray, quaternion: np.ndarray) -> bool:
        """Move robot to target pose"""
        try:
            transform = Affine(position.tolist(), quaternion.tolist())
            motion = CartesianMotion(transform)

            # Execute motion
            print(f"Moving to pose: pos={position}, quat={quaternion}")
            self.robot.move(motion)
            # Wait for robot to settle
            time.sleep(SETTLE_TIME)

            # Verify we reached the target pose
            current_state = self.robot.current_cartesian_state
            current_pose = current_state.pose.end_effector_pose

            # Check position accuracy
            pos_error = np.linalg.norm(
                np.array(current_pose.translation) - position)

            if pos_error > 0.01:  # 1cm tolerance
                print(
                    f"Warning: Position error {pos_error:.4f}m - waiting additional time")
                time.sleep(1.0)  # Additional settling time
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
            # Create motion to starting pose
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
        """Clean up resources following calibration script pattern"""
        print("Cleaning up cameras...")
        if self.external_1_cam:
            self.external_1_cam.release()
            print("External 2 camera released")
        if self.external_2_cam:
            self.external_2_cam.release()
            print("External 1 camera released")
        cv2.destroyAllWindows()
        print("Camera cleanup complete")

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

    def create_directories(self):
        """Create necessary directories"""
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(EXTERNAL_1_IMAGES_DIR, exist_ok=True)
        os.makedirs(EXTERNAL_2_IMAGES_DIR, exist_ok=True)
        print(f"Created directories: {DATA_DIR}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """Main function"""
    print("=== Hand-Eye Calibration Data Collection ===")
    print(f"Robot IP: {ROBOT_IP}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Hand camera: ID {EXTERNAL_1_CAM_ID}")
    print(f"External camera: ID {EXTERNAL_2_CAM_ID}")
    print(f"Image resolution: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")

    # Safety check
    print("SAFETY CHECKLIST:")
    print("1. ✓ Robot is powered on and FCI is enabled")
    print("2. ✓ Robot workspace is clear of obstacles")
    print("3. ✓ Chess calibration board is positioned and visible")
    print("4. ✓ Both cameras are connected and working")
    print("6. ✓ Emergency stop is accessible")
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
