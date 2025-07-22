import cv2
import numpy as np
import os
import glob
import json
import pickle
from datetime import datetime

# Configuration parameters
INPUT_DIR = "calibration_images"  # Directory containing calibration images
OUTPUT_DIR = "calibration_results"  # Directory to save results
IMAGE_PATTERN = "calib_*.jpg"  # Pattern to match calibration images

# Chessboard configuration (must match capture script)
CHESSBOARD_SIZE = (7, 6)  # Internal corners (width, height)
SQUARE_SIZE = 25.0  # Size of chessboard square in mm

# Debug settings
DEBUG_MODE = True  # Show detailed debug information
SAVE_DEBUG_IMAGES = True  # Save images with detected corners


def create_output_directory(directory):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def generate_object_points(pattern_size, square_size):
    """
    Generate 3D object points for the chessboard pattern.

    Args:
        pattern_size: Tuple of (width, height) internal corners
        square_size: Size of each square in mm

    Returns:
        3D points array
    """
    # Create a grid of 3D points
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0],
                           0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    return objp


def load_and_process_images(input_dir, pattern, pattern_size):
    """
    Load calibration images and detect chessboard corners.

    Args:
        input_dir: Directory containing images
        pattern: File pattern to match
        pattern_size: Chessboard pattern size

    Returns:
        (object_points, image_points, image_size, valid_images)
    """
    # Get list of calibration images
    image_files = glob.glob(os.path.join(input_dir, pattern))
    image_files.sort()

    if not image_files:
        raise ValueError(
            f"No images found in {input_dir} matching pattern {pattern}")

    print(f"Found {len(image_files)} calibration images")

    # Generate 3D object points
    objp = generate_object_points(pattern_size, SQUARE_SIZE)

    # Arrays to store object points and image points
    object_points = []  # 3D points in real world space
    image_points = []   # 2D points in image plane
    valid_images = []   # List of successfully processed images

    image_size = None

    for i, image_file in enumerate(image_files):
        print(
            f"Processing image {i+1}/{len(image_files)}: {os.path.basename(image_file)}")

        # Read image
        img = cv2.imread(image_file)
        if img is None:
            print(f"  ⚠ Could not load image: {image_file}")
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Store image size (from first valid image)
        if image_size is None:
            image_size = gray.shape[::-1]  # (width, height)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS +
                        cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)

            # Store the points
            object_points.append(objp)
            image_points.append(corners)
            valid_images.append(image_file)

            print(f"  ✓ Chessboard detected successfully")

            # Save debug image if requested
            if SAVE_DEBUG_IMAGES and DEBUG_MODE:
                debug_img = img.copy()
                cv2.drawChessboardCorners(
                    debug_img, pattern_size, corners, ret)

                debug_filename = f"debug_{i:03d}_{os.path.basename(image_file)}"
                debug_path = os.path.join(OUTPUT_DIR, debug_filename)
                cv2.imwrite(debug_path, debug_img)
        else:
            print(f"  ✗ Chessboard not found in image")

    print(
        f"\nSuccessfully processed {len(valid_images)}/{len(image_files)} images")

    if len(valid_images) < 10:
        print("⚠ Warning: Less than 10 valid images. Calibration quality may be poor.")

    return object_points, image_points, image_size, valid_images


def perform_calibration(object_points, image_points, image_size):
    """
    Perform camera calibration using detected points.

    Args:
        object_points: List of 3D object points
        image_points: List of 2D image points
        image_size: Image size (width, height)

    Returns:
        (ret, camera_matrix, dist_coeffs, rvecs, tvecs)
    """
    print("\nPerforming camera calibration...")

    # Initial camera matrix guess
    camera_matrix = np.eye(3, dtype=np.float32)
    camera_matrix[0, 0] = image_size[0]  # fx
    camera_matrix[1, 1] = image_size[1]  # fy
    camera_matrix[0, 2] = image_size[0] / 2  # cx
    camera_matrix[1, 2] = image_size[1] / 2  # cy

    # Initialize distortion coefficients
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    # Perform calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size,
        camera_matrix, dist_coeffs
    )

    return ret, camera_matrix, dist_coeffs, rvecs, tvecs


def calculate_reprojection_error(object_points, image_points, camera_matrix, dist_coeffs, rvecs, tvecs):
    """
    Calculate reprojection error for calibration quality assessment.

    Args:
        object_points: List of 3D object points
        image_points: List of 2D image points
        camera_matrix: Camera matrix
        dist_coeffs: Distortion coefficients
        rvecs: Rotation vectors
        tvecs: Translation vectors

    Returns:
        (mean_error, per_image_errors)
    """
    total_error = 0
    per_image_errors = []

    for i in range(len(object_points)):
        # Project 3D points to image plane
        projected_points, _ = cv2.projectPoints(
            object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )

        # Calculate error
        error = cv2.norm(
            image_points[i], projected_points, cv2.NORM_L2) / len(projected_points)
        per_image_errors.append(error)
        total_error += error

    mean_error = total_error / len(object_points)
    return mean_error, per_image_errors


def save_reprojection_visualization(object_points, image_points, camera_matrix, dist_coeffs, rvecs, tvecs, valid_images, output_dir):
    """
    Save images showing detected corners vs reprojected corners for visual validation.
    """
    for i, image_file in enumerate(valid_images):
        # Load original image
        img = cv2.imread(image_file)
        if img is None:
            continue

        # Project 3D points to image plane
        projected_points, _ = cv2.projectPoints(
            object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )

        # Draw detected corners in green
        cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, image_points[i], True)

        # Draw reprojected corners in red
        for point in projected_points:
            cv2.circle(img, tuple(point[0].astype(int)), 3, (0, 0, 255), -1)

        # Add legend
        cv2.putText(img, "Green: Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, "Red: Reprojected", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Save visualization
        reproj_filename = f"reprojection_{i:03d}_{os.path.basename(image_file)}"
        reproj_path = os.path.join(output_dir, reproj_filename)
        cv2.imwrite(reproj_path, img)

    print(f"Reprojection visualizations saved to {output_dir}")


def save_calibration_results(camera_matrix, dist_coeffs, image_size, reprojection_error,
                             per_image_errors, valid_images, output_dir):
    """
    Save calibration results to files.

    Args:
        camera_matrix: Camera matrix
        dist_coeffs: Distortion coefficients
        image_size: Image size
        reprojection_error: Mean reprojection error
        per_image_errors: Per-image reprojection errors
        valid_images: List of valid image files
        output_dir: Output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as NumPy arrays
    np.savez(os.path.join(output_dir, f"camera_params_{timestamp}.npz"),
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             image_size=image_size,
             reprojection_error=reprojection_error)

    # Save as pickle (for easy Python loading)
    calibration_data = {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'image_size': image_size,
        'reprojection_error': reprojection_error,
        'per_image_errors': per_image_errors,
        'valid_images': valid_images,
        'timestamp': timestamp
    }

    with open(os.path.join(output_dir, f"calibration_data_{timestamp}.pkl"), 'wb') as f:
        pickle.dump(calibration_data, f)

    # Save as JSON (human-readable)
    json_data = {
        'camera_matrix': camera_matrix.tolist(),
        'dist_coeffs': dist_coeffs.tolist(),
        'image_size': image_size,
        'reprojection_error': float(reprojection_error),
        'per_image_errors': [float(e) for e in per_image_errors],
        'valid_images': valid_images,
        'timestamp': timestamp,
        'calibration_settings': {
            'chessboard_size': CHESSBOARD_SIZE,
            'square_size_mm': SQUARE_SIZE
        }
    }

    with open(os.path.join(output_dir, f"calibration_results_{timestamp}.json"), 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"Calibration results saved to {output_dir}")


def print_calibration_summary(camera_matrix, dist_coeffs, image_size, reprojection_error,
                              per_image_errors, valid_images):
    """Print detailed calibration summary."""
    print("\n" + "="*60)
    print("CAMERA CALIBRATION RESULTS")
    print("="*60)

    print(f"Number of images used: {len(valid_images)}")
    print(f"Image size: {image_size[0]} x {image_size[1]} pixels")
    print(
        f"Chessboard pattern: {CHESSBOARD_SIZE[0]} x {CHESSBOARD_SIZE[1]} internal corners")
    print(f"Square size: {SQUARE_SIZE} mm")

    print(f"\nCamera Matrix:")
    print(f"  fx = {camera_matrix[0, 0]:.2f}")
    print(f"  fy = {camera_matrix[1, 1]:.2f}")
    print(f"  cx = {camera_matrix[0, 2]:.2f}")
    print(f"  cy = {camera_matrix[1, 2]:.2f}")

    print(f"\nDistortion Coefficients:")
    print(f"  k1 = {dist_coeffs[0, 0]:.6f}")
    print(f"  k2 = {dist_coeffs[1, 0]:.6f}")
    print(f"  p1 = {dist_coeffs[2, 0]:.6f}")
    print(f"  p2 = {dist_coeffs[3, 0]:.6f}")
    if len(dist_coeffs) > 4:
        print(f"  k3 = {dist_coeffs[4, 0]:.6f}")

    print(f"\nReprojection Error:")
    print(f"  Mean error: {reprojection_error:.4f} pixels")
    print(f"  Max error: {max(per_image_errors):.4f} pixels")
    print(f"  Min error: {min(per_image_errors):.4f} pixels")
    print(f"  Std deviation: {np.std(per_image_errors):.4f} pixels")

    # Quality assessment
    print(f"\nCalibration Quality Assessment:")
    if reprojection_error < 0.5:
        print("  ✓ Excellent calibration (error < 0.5 pixels)")
    elif reprojection_error < 1.0:
        print("  ✓ Good calibration (error < 1.0 pixels)")
    elif reprojection_error < 2.0:
        print("  ⚠ Acceptable calibration (error < 2.0 pixels)")
    else:
        print("  ✗ Poor calibration (error > 2.0 pixels)")
        print("    Consider recapturing images with better lighting and focus")

    print("="*60)


def main():
    """Main function to perform camera calibration."""

    # Create output directory
    create_output_directory(OUTPUT_DIR)

    try:
        # Load and process images
        object_points, image_points, image_size, valid_images = load_and_process_images(
            INPUT_DIR, IMAGE_PATTERN, CHESSBOARD_SIZE
        )

        if len(valid_images) < 3:
            print("Error: Need at least 3 valid images for calibration")
            return

        # Perform calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = perform_calibration(
            object_points, image_points, image_size
        )

        if not ret:
            print("Error: Calibration failed")
            return

        # Calculate reprojection error
        reprojection_error, per_image_errors = calculate_reprojection_error(
            object_points, image_points, camera_matrix, dist_coeffs, rvecs, tvecs
        )

        save_reprojection_visualization(
            object_points, image_points, camera_matrix, dist_coeffs, rvecs, tvecs, valid_images, OUTPUT_DIR)

        # Print results
        if DEBUG_MODE:
            print_calibration_summary(
                camera_matrix, dist_coeffs, image_size, reprojection_error,
                per_image_errors, valid_images
            )

        # Save results
        save_calibration_results(
            camera_matrix, dist_coeffs, image_size, reprojection_error,
            per_image_errors, valid_images, OUTPUT_DIR
        )

        print(f"\n✓ Calibration completed successfully!")
        print(f"Results saved to: {OUTPUT_DIR}")

    except Exception as e:
        print(f"Error during calibration: {str(e)}")
        return


if __name__ == "__main__":
    main()
