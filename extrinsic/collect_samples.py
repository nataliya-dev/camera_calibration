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
from franky import Robot, Affine, CartesianMotion, ReferenceType
import itertools

# =============================================================================
# CONFIGURATION PARAMETERS - MODIFY THESE AS NEEDED
# =============================================================================

# Robot connection
ROBOT_IP = "172.16.0.2"

# Data collection paths
DATA_DIR = "calibration_data"
HAND_IMAGES_DIR = os.path.join(DATA_DIR, "hand_camera")
EXTERNAL_IMAGES_DIR = os.path.join(DATA_DIR, "external_camera")
EE_POSES_FILE = os.path.join(DATA_DIR, "ee_poses.json")

# Camera configuration
HAND_CAMERA_ID = 0      # USB camera ID for hand camera
EXTERNAL_CAMERA_ID = 1  # USB camera ID for external camera
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Camera settings - following calibration script pattern
USE_AUTOFOCUS = "fixed"  # Options: "enabled", "disabled", "fixed"
CAPTURE_WARMUP_FRAMES = 10  # Number of frames to skip for camera warmup

# Camera intrinsic parameters (set these from your camera calibration)
# Hand camera intrinsics
HAND_CAMERA_MATRIX = np.array([
    [800.0, 0.0, 320.0],
    [0.0, 800.0, 240.0],
    [0.0, 0.0, 1.0]
])
HAND_DIST_COEFFS = np.array([0.1, -0.2, 0.0, 0.0, 0.0])

# External camera intrinsics  
EXTERNAL_CAMERA_MATRIX = np.array([
    [1000.0, 0.0, 640.0],
    [0.0, 1000.0, 480.0],
    [0.0, 0.0, 1.0]
])
EXTERNAL_DIST_COEFFS = np.array([0.05, -0.1, 0.0, 0.0, 0.0])

# Robot motion parameters
ROBOT_DYNAMICS_FACTOR = 0.05  # Reduce speed for safety
SETTLE_TIME = 2.0  # Time to wait after reaching each pose (seconds)

# 3D Grid definition (relative to current robot position)
# Position grid (meters relative to current position)
X_POSITIONS = [-0.1, 0.0, 0.1]  # X offsets
Y_POSITIONS = [-0.1, 0.0, 0.1]  # Y offsets  
Z_POSITIONS = [-0.05, 0.0, 0.05]  # Z offsets

# Orientation variations (degrees)
ROLL_ANGLES = [-15, 0, 15]   # Roll variations in degrees
PITCH_ANGLES = [-15, 0, 15]  # Pitch variations in degrees
YAW_ANGLES = [-15, 0, 15]    # Yaw variations in degrees

# Safety limits
MAX_TRANSLATION_FROM_START = 0.2  # Maximum distance from starting position (meters)
MIN_SAMPLES = 10  # Minimum number of samples required

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
        self.robot.relative_dynamics_factor = ROBOT_DYNAMICS_FACTOR
        
        # Store starting pose
        self.start_pose = self.robot.current_cartesian_state.pose.end_effector_pose
        print(f"Starting pose recorded:")
        print(f"Position: {self.start_pose.translation}")
        print(f"Rotation matrix:\n{self.start_pose.linear}")
        
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
            raise RuntimeError(f"Failed to open external camera {EXTERNAL_CAMERA_ID}")
        
        # Configure both cameras with same settings
        for camera, camera_name in [(self.hand_camera, "hand"), (self.external_camera, "external")]:
            # Set resolution
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
                print(f"{camera_name} camera autofocus: Enabled for initial focus, then will be fixed")
            
            # Get actual resolution (may differ from requested)
            actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"{camera_name} camera - Requested: {IMAGE_WIDTH}x{IMAGE_HEIGHT}, Actual: {actual_width}x{actual_height}")
            
            if actual_width != IMAGE_WIDTH or actual_height != IMAGE_HEIGHT:
                print(f"⚠ Warning: {camera_name} camera resolution differs from requested!")
        
        # For "fixed" autofocus, let cameras focus then disable
        if USE_AUTOFOCUS == "fixed":
            print("Allowing cameras to focus...")
            time.sleep(3)  # Give time to focus
            self.hand_camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.external_camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            print("Autofocus now disabled - focus is fixed for both cameras")
        
        # Warm up cameras with more frames for stability
        print(f"Warming up cameras with {CAPTURE_WARMUP_FRAMES} frames...")
        for i in range(CAPTURE_WARMUP_FRAMES):
            ret_hand, _ = self.hand_camera.read()
            ret_external, _ = self.external_camera.read()
            if not ret_hand or not ret_external:
                print(f"Warning: Camera read failed during warmup frame {i+1}")
            time.sleep(0.1)  # Small delay between warmup frames
        
        print("Cameras initialized successfully")
        
    def generate_poses(self) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """Generate all target poses for data collection"""
        poses = []
        sample_id = 0
        
        print("Generating target poses...")
        
        for x, y, z in itertools.product(X_POSITIONS, Y_POSITIONS, Z_POSITIONS):
            for roll, pitch, yaw in itertools.product(ROLL_ANGLES, PITCH_ANGLES, YAW_ANGLES):
                # Create relative position
                relative_position = np.array([x, y, z])
                
                # Check if within safety limits
                if np.linalg.norm(relative_position) > MAX_TRANSLATION_FROM_START:
                    continue
                
                # Create relative rotation (in degrees, convert to radians)
                relative_rotation = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
                
                # Create relative transform
                relative_transform = Affine(relative_position.tolist(), 
                                          relative_rotation.as_quat())
                
                # Generate sample ID
                sample_id_str = f"sample_{sample_id:04d}"
                
                poses.append((relative_position, relative_rotation.as_quat(), sample_id_str))
                sample_id += 1
        
        print(f"Generated {len(poses)} target poses")
        return poses
        
    def capture_images(self, sample_id: str) -> bool:
        """Capture images from both cameras with multiple attempts for stability"""
        try:
            # Take multiple frames to ensure good capture (following calibration script pattern)
            max_attempts = 3
            
            for attempt in range(max_attempts):
                # Capture hand camera image
                ret_hand, hand_image = self.hand_camera.read()
                if not ret_hand:
                    print(f"Failed to capture hand camera image for {sample_id}, attempt {attempt+1}")
                    if attempt < max_attempts - 1:
                        time.sleep(0.1)
                        continue
                    else:
                        return False
                
                # Capture external camera image  
                ret_external, external_image = self.external_camera.read()
                if not ret_external:
                    print(f"Failed to capture external camera image for {sample_id}, attempt {attempt+1}")
                    if attempt < max_attempts - 1:
                        time.sleep(0.1)
                        continue
                    else:
                        return False
                
                # If we got here, both captures succeeded
                break
            
            # Validate image quality (basic checks)
            if hand_image is None or external_image is None:
                print(f"Captured images are None for {sample_id}")
                return False
            
            if hand_image.size == 0 or external_image.size == 0:
                print(f"Captured images are empty for {sample_id}")
                return False
            
            # Save images with high quality
            hand_image_path = os.path.join(HAND_IMAGES_DIR, f"{sample_id}.jpg")
            external_image_path = os.path.join(EXTERNAL_IMAGES_DIR, f"{sample_id}.jpg")
            
            # Use high quality JPEG settings
            jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
            
            success_hand = cv2.imwrite(hand_image_path, hand_image, jpeg_params)
            success_external = cv2.imwrite(external_image_path, external_image, jpeg_params)
            
            if not success_hand or not success_external:
                print(f"Failed to save images for {sample_id}")
                return False
            
            print(f"Images saved for {sample_id} (hand: {hand_image.shape}, external: {external_image.shape})")
            return True
            
        except Exception as e:
            print(f"Error capturing images for {sample_id}: {e}")
            return False
    
    def move_to_pose(self, relative_position: np.ndarray, relative_quaternion: np.ndarray) -> bool:
        """Move robot to target pose"""
        try:
            # Create relative motion
            relative_transform = Affine(relative_position.tolist(), relative_quaternion)
            motion = CartesianMotion(relative_transform, ReferenceType.Relative)
            
            # Execute motion
            print(f"Moving to pose: pos={relative_position}, quat={relative_quaternion}")
            self.robot.move(motion)
            
            # Wait for robot to settle
            time.sleep(SETTLE_TIME)
            return True
            
        except Exception as e:
            print(f"Error moving to pose: {e}")
            return False
    
    def collect_sample(self, relative_position: np.ndarray, relative_quaternion: np.ndarray, sample_id: str) -> bool:
        """Collect a single calibration sample"""
        print(f"\nCollecting sample {sample_id}...")
        
        # Move to target pose
        if not self.move_to_pose(relative_position, relative_quaternion):
            return False
        
        # Get current robot state
        current_state = self.robot.current_cartesian_state
        ee_pose = current_state.pose.end_effector_pose
        
        # Store pose data
        pose_data = {
            "position": ee_pose.translation.tolist(),
            "orientation": R.from_matrix(ee_pose.linear).as_quat().tolist()  # [x, y, z, w]
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
            
            if len(poses) < MIN_SAMPLES:
                print(f"Warning: Only {len(poses)} poses generated, minimum is {MIN_SAMPLES}")
            
            print(f"\nStarting data collection for {len(poses)} poses...")
            print("Make sure the ChArUco calibration board is visible to both cameras!")
            input("Press Enter to start data collection...")
            
            # Collect data
            successful_samples = 0
            failed_samples = 0
            
            for i, (rel_pos, rel_quat, sample_id) in enumerate(poses):
                print(f"\nProgress: {i+1}/{len(poses)}")
                
                if self.collect_sample(rel_pos, rel_quat, sample_id):
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
            
            if successful_samples >= MIN_SAMPLES:
                print(f"✓ Sufficient data collected for calibration!")
            else:
                print(f"⚠ Warning: Only {successful_samples} samples collected, recommend at least {MIN_SAMPLES}")
            
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
    print(f"Position grid: {len(X_POSITIONS)}x{len(Y_POSITIONS)}x{len(Z_POSITIONS)}")
    print(f"Orientation grid: {len(ROLL_ANGLES)}x{len(PITCH_ANGLES)}x{len(YAW_ANGLES)}")
    print(f"Expected samples: ~{len(X_POSITIONS)*len(Y_POSITIONS)*len(Z_POSITIONS)*len(ROLL_ANGLES)*len(PITCH_ANGLES)*len(YAW_ANGLES)}")
    
    print(f"\nCamera Intrinsics Configured:")
    print(f"Hand camera matrix:\n{HAND_CAMERA_MATRIX}")
    print(f"Hand distortion: {HAND_DIST_COEFFS}")
    print(f"External camera matrix:\n{EXTERNAL_CAMERA_MATRIX}")
    print(f"External distortion: {EXTERNAL_DIST_COEFFS}")
    print()
    
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