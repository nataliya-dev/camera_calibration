# Example data structure and usage

# 1. Directory structure expected:
"""
calibration_data/
├── ee_poses.json
├── hand_camera/
│   ├── sample_001.jpg
│   ├── sample_002.jpg
│   └── ...
└── external_camera/
    ├── sample_001.jpg
    ├── sample_002.jpg
    └── ...
"""

# 2. Expected format for ee_poses.json:
example_ee_poses = {
    "sample_001": {
        "position": [0.5, 0.2, 0.3],  # [x, y, z] in meters
        "orientation": [0.0, 0.0, 0.0, 1.0]  # quaternion [x, y, z, w]
    },
    "sample_002": {
        "position": [0.4, 0.3, 0.35],
        "orientation": [0.1, 0.0, 0.0, 0.995]
    },
    # ... more samples
}

# 3. Usage example with your robot API:
"""
import json
import cv2
import numpy as np

# Assuming you have a robot API like this:
# robot = YourRobotAPI()

def collect_calibration_data(robot, num_samples=20):
    '''Collect calibration data by moving robot to different positions'''
    
    ee_poses = {}
    
    # Create directories
    os.makedirs("calibration_data/hand_camera", exist_ok=True)
    os.makedirs("calibration_data/external_camera", exist_ok=True)
    
    for i in range(num_samples):
        sample_id = f"sample_{i:03d}"
        
        # Move robot to a new position (you define this logic)
        # For example, sample different positions in workspace
        if i == 0:
            # First position
            joint_angles = [0.0, -1.57, 1.57, 0.0, 1.57, 0.0]  # example
        else:
            # Generate random valid joint angles or predefined positions
            joint_angles = generate_random_joint_angles()  # your function
        
        # Move robot
        robot.move_to_joint_angles(joint_angles)
        
        # Get current end-effector pose
        ee_position, ee_orientation = robot.get_ee_pose()  # your API
        
        # Store pose data
        ee_poses[sample_id] = {
            "position": ee_position.tolist(),
            "orientation": ee_orientation.tolist()  # ensure quaternion [x,y,z,w]
        }
        
        # Capture images from both cameras
        hand_image = robot.get_hand_camera_image()  # your API
        external_image = get_external_camera_image()  # your API
        
        # Save images
        cv2.imwrite(f"calibration_data/hand_camera/{sample_id}.jpg", hand_image)
        cv2.imwrite(f"calibration_data/external_camera/{sample_id}.jpg", external_image)
        
        print(f"Collected sample {i+1}/{num_samples}")
    
    # Save end-effector poses
    with open("calibration_data/ee_poses.json", "w") as f:
        json.dump(ee_poses, f, indent=2)
    
    print("Data collection complete!")

def generate_random_joint_angles():
    '''Generate random valid joint angles within robot limits'''
    # Define joint limits for your robot
    joint_limits = [
        (-3.14, 3.14),  # Joint 1
        (-2.0, 2.0),    # Joint 2
        (-2.0, 2.0),    # Joint 3
        (-3.14, 3.14),  # Joint 4
        (-1.57, 1.57),  # Joint 5
        (-3.14, 3.14),  # Joint 6
    ]
    
    joint_angles = []
    for min_angle, max_angle in joint_limits:
        angle = np.random.uniform(min_angle, max_angle)
        joint_angles.append(angle)
    
    return joint_angles
"""

# 4. Camera calibration helper (run this first to get camera intrinsics):
"""
import cv2
import numpy as np
import glob

def calibrate_camera(images_path, checkerboard_size=(9, 6), square_size=0.025):
    '''Calibrate camera using checkerboard images'''
    
    # Prepare object points
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    images = glob.glob(images_path)
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                      cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            imgpoints.append(corners2)
    
    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    
    return camera_matrix, dist_coeffs

# Usage:
# hand_camera_matrix, hand_dist_coeffs = calibrate_camera("hand_camera_calib/*.jpg")
# external_camera_matrix, external_dist_coeffs = calibrate_camera("external_camera_calib/*.jpg")
"""

# 5. ChArUco board generation (optional - if you need to print the board):
"""
import cv2
import numpy as np

def generate_charuco_board(squares_x=7, squares_y=5, square_length=40, marker_length=20, 
                          dictionary=cv2.aruco.DICT_6X6_250, dpi=300):
    '''Generate ChArUco board for printing'''
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)
    
    # Calculate image size in pixels
    board_width_mm = squares_x * square_length
    board_height_mm = squares_y * square_length
    
    # Convert to pixels (assuming 300 DPI)
    mm_to_inch = 1 / 25.4
    img_width = int(board_width_mm * mm_to_inch * dpi)
    img_height = int(board_height_mm * mm_to_inch * dpi)
    
    # Generate board image
    board_image = board.generateImage((img_width, img_height))
    
    # Save board
    cv2.imwrite("charuco_board.png", board_image)
    print(f"ChArUco board saved as charuco_board.png")
    print(f"Board size: {squares_x}x{squares_y} squares")
    print(f"Square size: {square_length}mm")
    print(f"Marker size: {marker_length}mm")
    print(f"Print at {dpi} DPI for correct scale")
    
    return board_image

# Generate the board
generate_charuco_board()
"""

# 6. Advanced usage with better initial guess:
"""
def get_better_initial_guess(samples):
    '''Get better initial guess using basic geometric relationships'''
    
    # This is a simplified approach - you can implement more sophisticated methods
    # like using PnP solutions or basic geometric constraints
    
    # For hand-eye: assume camera is roughly at end-effector with some offset
    T_hand_eye = np.eye(4)
    T_hand_eye[:3, 3] = [0.05, 0.0, 0.1]  # 5cm in x, 10cm in z
    
    # For base-external: assume external camera is roughly above the workspace
    T_base_external = np.eye(4)
    T_base_external[:3, 3] = [0.5, 0.5, 1.0]  # 50cm x, 50cm y, 1m z
    
    return T_hand_eye, T_base_external

# Use in calibration:
# initial_guess = get_better_initial_guess(samples)
# result = calibrator.calibrate(initial_guess=initial_guess)
"""

# 7. Complete usage example:
"""
if __name__ == "__main__":
    # Step 1: Collect calibration data (run once)
    # collect_calibration_data(robot, num_samples=20)
    
    # Step 2: Get camera intrinsics (run once per camera)
    # hand_camera_matrix, hand_dist_coeffs = calibrate_camera("hand_camera_calib/*.jpg")
    # external_camera_matrix, external_dist_coeffs = calibrate_camera("external_camera_calib/*.jpg")
    
    # Step 3: Run calibration (main script above)
    # main()
"""