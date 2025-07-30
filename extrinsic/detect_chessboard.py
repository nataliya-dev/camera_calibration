import cv2
import numpy as np
import os
import glob

# Parameters - modify these as needed
CHESSBOARD_SIZE = (7, 6)  # Internal corners (width, height)
INPUT_DIR = "calibration_data/r1"  # Directory containing input images
OUTPUT_DIR = "debug_output"  # Directory for debug images with overlays
IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png",
                    "*.bmp", "*.tiff"]  # Supported formats

# Detection parameters
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS +
                   cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def create_output_directory(output_dir):
    """Create output directory if it doesn't exist"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")


def get_image_files(input_dir, extensions):
    """Get all image files from input directory"""
    image_files = []
    for ext in extensions:
        pattern = os.path.join(input_dir, ext)
        image_files.extend(glob.glob(pattern))
    return image_files


def detect_and_draw_chessboard(image_path, chessboard_size, output_dir):
    """Detect chessboard in image and save with overlay"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return False

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # Create output image (copy of original)
    output_img = img.copy()

    if ret:
        print(f"✓ Chessboard detected in {os.path.basename(image_path)}")

        # Refine corner positions for better accuracy
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)

        # Draw chessboard corners and grid
        cv2.drawChessboardCorners(
            output_img, chessboard_size, corners_refined, ret)

        # Add text indicating successful detection
        cv2.putText(output_img, "CHESSBOARD DETECTED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Add corner count info
        corner_count = f"Corners: {len(corners_refined)}"
        cv2.putText(output_img, corner_count, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        print(f"✗ No chessboard detected in {os.path.basename(image_path)}")

        # Add text indicating failed detection
        cv2.putText(output_img, "NO CHESSBOARD DETECTED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Save output image
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    output_filename = f"{name}_debug{ext}"
    output_path = os.path.join(output_dir, output_filename)

    success = cv2.imwrite(output_path, output_img)
    if success:
        print(f"  Saved debug image: {output_filename}")
    else:
        print(f"  Error saving: {output_filename}")

    return ret


def main():
    """Main function to process all images"""
    print(f"Chessboard Detection Script")
    print(f"Chessboard size: {CHESSBOARD_SIZE}")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)

    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' does not exist!")
        print(f"Please create the directory and add images to process.")
        return

    # Create output directory
    create_output_directory(OUTPUT_DIR)

    # Get all image files
    image_files = get_image_files(INPUT_DIR, IMAGE_EXTENSIONS)

    if not image_files:
        print(f"No image files found in '{INPUT_DIR}'")
        print(f"Supported formats: {', '.join(IMAGE_EXTENSIONS)}")
        return

    print(f"Found {len(image_files)} image(s) to process")
    print("-" * 50)

    # Process each image
    successful_detections = 0
    total_images = len(image_files)

    for image_path in image_files:
        if detect_and_draw_chessboard(image_path, CHESSBOARD_SIZE, OUTPUT_DIR):
            successful_detections += 1

    # Summary
    print("-" * 50)
    print(f"Processing complete!")
    print(f"Total images processed: {total_images}")
    print(f"Successful detections: {successful_detections}")
    print(f"Failed detections: {total_images - successful_detections}")
    print(f"Debug images saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
