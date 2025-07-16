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
import itertools

# =============================================================================
# CONFIGURATION PARAMETERS - MODIFY THESE AS NEEDED
# =============================================================================

# Robot connection
ROBOT_IP = "192.168.0.2"

# Data collection paths
DATA_DIR = "calibration_data"
HAND_IMAGES_DIR = os.path.join(DATA_DIR, "hand_camera")
EXTERNAL_IMAGES_DIR = os.path.join(DATA_DIR, "external_camera")
EE_POSES_FILE = os.path.join(DATA_DIR, "ee_poses.json")

# Camera configuration
HAND_CAMERA_ID = 0      # USB camera ID for hand camera
EXTERNAL_CAMERA_ID = 2  # USB camera ID for external camera
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Camera settings - following calibration script pattern
USE_AUTOFOCUS = "disabled"  # Options: "enabled", "disabled", "fixed"

# Robot motion parameters
ROBOT_DYNAMICS_FACTOR = 0.05  # Reduce speed for safety
SETTLE_TIME = 1.0  # Time to wait after reaching each pose (seconds)


# =============================================================================
# DATA COLLECTION CLASS
# =============================================================================

class HandEyeDataCollector:
    """Collects calibration data for hand-eye calibration"""

    def __init__(self):
        self.robot = None
        self.hand_camera = None
        self.external_camera = None
        self.start_pose = None
        self.collected_data = {}

    def initialize_robot(self):
        """Initialize robot connection"""
        print(f"Connecting to robot at {ROBOT_IP}...")
        self.robot = Robot(ROBOT_IP)
        self.robot.recover_from_errors()
        self.robot.relative_dynamics_factor = ROBOT_DYNAMICS_FACTOR

        home = JointMotion([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.robot.move(home)

        motion = CartesianMotion(
            Affine([0.0, 0.35, -0.13]), ReferenceType.Relative)
        self.robot.move(motion)

        # Store starting pose
        self.start_pose = self.robot.current_cartesian_state.pose.end_effector_pose
        print(f"Starting pose recorded:")
        print(f"Position: {self.start_pose.translation}")
        print(f"Quaternion:\n{self.start_pose.quaternion}")

    def initialize_cameras(self):
        """Initialize camera connections following calibration script pattern"""
        print("Initializing cameras...")

        # Hand camera (on robot end-effector)
        self.hand_camera = cv2.VideoCapture(HAND_CAMERA_ID)
        if not self.hand_camera.isOpened():
            raise RuntimeError(f"Failed to open hand camera {HAND_CAMERA_ID}")

        # External camera (static)
        self.external_camera = cv2.VideoCapture(EXTERNAL_CAMERA_ID)
        if not self.external_camera.isOpened():
            raise RuntimeError(
                f"Failed to open external camera {EXTERNAL_CAMERA_ID}")

        # Configure both cameras with same settings
        for camera, camera_name in [(self.hand_camera, "hand"), (self.external_camera, "external")]:
            # Set resolution
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

            # Handle autofocus based on setting
            if USE_AUTOFOCUS == "enabled":
                camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                print(f"{camera_name} camera autofocus: Enabled throughout capture")
            elif USE_AUTOFOCUS == "disabled":
                camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                print(f"{camera_name} camera autofocus: Disabled (manual focus)")
            elif USE_AUTOFOCUS == "fixed":
                # Enable autofocus initially, then disable after focusing
                camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                print(
                    f"{camera_name} camera autofocus: Enabled for initial focus, then will be fixed")

            # Get actual resolution (may differ from requested)
            actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(
                f"{camera_name} camera - Requested: {IMAGE_WIDTH}x{IMAGE_HEIGHT}, Actual: {actual_width}x{actual_height}")

            if actual_width != IMAGE_WIDTH or actual_height != IMAGE_HEIGHT:
                print(
                    f"⚠ Warning: {camera_name} camera resolution differs from requested!")

        # For "fixed" autofocus, let cameras focus then disable
        if USE_AUTOFOCUS == "fixed":
            print("Allowing cameras to focus...")
            time.sleep(3)  # Give time to focus
            self.hand_camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.external_camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            print("Autofocus now disabled - focus is fixed for both cameras")

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

        # # 1. Original pose (starting position)
        # poses.append((start_position, start_quaternion,
        #              f"sample_{sample_id:04d}"))
        # sample_id += 1

        # 2-3. Z up and down (±5cm)
        for z_offset in [0.05, -0.05]:
            pos = start_position + np.array([0, 0, z_offset])
            poses.append((pos, start_quaternion, f"sample_{sample_id:04d}"))
            sample_id += 1

        # 4-7. X positive and negative, Y positive and negative (±5cm)
        for offset in [(0.05, 0, 0), (-0.05, 0, 0), (0, 0.05, 0), (0, -0.05, 0)]:
            pos = start_position + np.array(offset)
            poses.append((pos, start_quaternion, f"sample_{sample_id:04d}"))
            sample_id += 1

        # 8-13. Original position with small rotations (±10 degrees)
        rotation_variations = [
            (8, 0, 0),   # Roll +10°
            (-8, 0, 0),  # Roll -10°
            (0, 8, 0),   # Pitch +10°
            (0, -8, 0),  # Pitch -10°
            (0, 0, 8),   # Yaw +10°
            (0, 0, -8)   # Yaw -10°
        ]

        for roll, pitch, yaw in rotation_variations:
            relative_rotation = R.from_euler(
                'xyz', [roll, pitch, yaw], degrees=True)
            global_rotation = start_rotation * relative_rotation
            global_quaternion = global_rotation.as_quat()
            poses.append((start_position, global_quaternion,
                         f"sample_{sample_id:04d}"))
            sample_id += 1

        # 14-20. Random poses with position and rotation variations
        random_poses = [
            # Format: (x, y, z, roll, pitch, yaw)
            (0.03, 0.02, 0.03, 8, -5, 12),
            (-0.02, 0.04, -0.03, -12, 7, -8),
            (0.04, -0.03, 0.03, 5, 3, -10),
            (-0.03, -0.02, 0.03, -8, -12, 6),
            (0.02, 0.05, -0.03, 8, -3, 9),
            (-0.04, 0.01, 0.03, -6, 8, -11),
            (0.01, -0.04, -0.02, 8, -7, 4)
        ]

        for x, y, z, roll, pitch, yaw in random_poses:
            # Position
            pos = start_position + np.array([x, y, z])

            # Rotation
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

            # Capture hand camera image
            ret_hand, hand_image = self.hand_camera.read()
            # cv2.imshow("Hand Camera", hand_image)  # Added window name
            # cv2.waitKey(1)
            if not ret_hand:
                print(f"Failed to capture hand camera image for {sample_id}")

            # Capture external camera image
            ret_external, external_image = self.external_camera.read()
            # cv2.imshow("External Camera", external_image)
            # cv2.waitKey(1)
            if not ret_external:
                print(
                    f"Failed to capture external camera image for {sample_id}")

            # Validate image quality (basic checks)
            if hand_image is None or external_image is None:
                print(f"Captured images are None for {sample_id}")
                return False

            if hand_image.size == 0 or external_image.size == 0:
                print(f"Captured images are empty for {sample_id}")
                return False

            # Save images with high quality
            hand_image_path = os.path.join(HAND_IMAGES_DIR, f"{sample_id}.jpg")
            external_image_path = os.path.join(
                EXTERNAL_IMAGES_DIR, f"{sample_id}.jpg")

            # Use high quality JPEG settings
            jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 95]

            success_hand = cv2.imwrite(
                hand_image_path, hand_image, jpeg_params)
            success_external = cv2.imwrite(
                external_image_path, external_image, jpeg_params)

            if not success_hand or not success_external:
                print(f"Failed to save images for {sample_id}")
                return False

            print(
                f"Images saved for {sample_id} (hand: {hand_image.shape}, external: {external_image.shape})")
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
        if self.hand_camera:
            self.hand_camera.release()
            print("Hand camera released")
        if self.external_camera:
            self.external_camera.release()
            print("External camera released")
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

                # Optional: Add user confirmation for each sample
                # response = input("Continue to next sample? (y/n): ")
                # if response.lower() != 'y':
                #     break

            # Return to start
            self.return_to_start()

            # Save data
            if successful_samples > 0:
                self.save_data()

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
        os.makedirs(HAND_IMAGES_DIR, exist_ok=True)
        os.makedirs(EXTERNAL_IMAGES_DIR, exist_ok=True)
        print(f"Created directories: {DATA_DIR}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """Main function"""
    print("=== Hand-Eye Calibration Data Collection ===")
    print(f"Robot IP: {ROBOT_IP}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Hand camera: ID {HAND_CAMERA_ID}")
    print(f"External camera: ID {EXTERNAL_CAMERA_ID}")
    print(f"Image resolution: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"Autofocus mode: {USE_AUTOFOCUS}")

    # Safety check
    print("SAFETY CHECKLIST:")
    print("1. ✓ Robot is powered on and FCI is enabled")
    print("2. ✓ Robot workspace is clear of obstacles")
    print("3. ✓ ChArUco calibration board is positioned and visible")
    print("4. ✓ Both cameras are connected and working")
    print("5. ✓ Camera intrinsic parameters are correctly set above")
    print("6. ✓ Emergency stop is accessible")
    print()

    response = input("All safety checks complete? (y/n): ")
    if response.lower() != 'y':
        print("Please complete safety checks before proceeding.")
        return

    # Start data collection
    collector = HandEyeDataCollector()
    collector.collect_all_data()


if __name__ == "__main__":
    main()
