import depthai as dai
import cv2
import os
import time
import numpy as np

# ============================================================================
# CONFIGURATION - Edit these parameters as needed
# ============================================================================

# Camera settings
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

# Capture settings
CAPTURE_INTERVAL = 2.0  # Seconds between automatic captures
MIN_CAPTURES = 30  # Minimum number of images needed
MAX_CAPTURES = 40  # Maximum number of images to capture
OUTPUT_DIR = "calibration_images"
IMAGE_PREFIX = "calib_"

# Chessboard configuration
CHESSBOARD_SIZE = (7, 6)  # Internal corners (width, height)
SQUARE_SIZE = 25.0  # Size of chessboard square in mm

# Quality control settings
ENABLE_BLUR_DETECTION = True
BLUR_THRESHOLD = 50.0  # Laplacian variance threshold
MIN_SHARPNESS_SCORE = 40.0  # Minimum sharpness for chessboard region

# Display settings
DEBUG_MODE = True
WINDOW_NAME = "DepthAI Calibration - Press 'q' to quit, 'c' to capture"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def create_output_directory(directory):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def detect_chessboard(image, pattern_size):
    """
    Detect chessboard corners in the image.

    Returns:
        (success, corners, gray_image)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        # Refine corner positions
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    return ret, corners, gray


def calculate_blur_score(image):
    """
    Calculate blur score using Laplacian variance.
    Higher scores indicate sharper images.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = laplacian.var()

    return blur_score


def calculate_region_sharpness(image, corners):
    """Calculate sharpness specifically in the chessboard region."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Get bounding box of chessboard
    x_coords = corners[:, 0, 0]
    y_coords = corners[:, 0, 1]

    x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
    y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))

    # Add padding
    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(gray.shape[1], x_max + padding)
    y_max = min(gray.shape[0], y_max + padding)

    # Extract chessboard region
    roi = gray[y_min:y_max, x_min:x_max]

    # Calculate sharpness using gradient magnitude
    sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    sharpness_score = np.mean(gradient_magnitude)

    return sharpness_score


def is_image_acceptable(image, corners):
    """
    Check if image quality is acceptable for calibration.

    Returns:
        (is_acceptable, blur_score, sharpness_score, reason)
    """
    if not ENABLE_BLUR_DETECTION:
        return True, 0, 0, "Blur detection disabled"

    blur_score = calculate_blur_score(image)
    sharpness_score = calculate_region_sharpness(image, corners)

    if blur_score < BLUR_THRESHOLD:
        return False, blur_score, sharpness_score, f"Too blurry ({blur_score:.1f} < {BLUR_THRESHOLD})"

    if sharpness_score < MIN_SHARPNESS_SCORE:
        return False, blur_score, sharpness_score, f"Not sharp enough ({sharpness_score:.1f} < {MIN_SHARPNESS_SCORE})"

    return True, blur_score, sharpness_score, "Good quality"


def draw_debug_info(image, corners_found, capture_count, blur_score=0,
                    sharpness_score=0, quality_ok=True):
    """Draw debug information on the image."""
    height, width = image.shape[:2]

    # Status
    status_text = f"Captures: {capture_count}/{MAX_CAPTURES}"
    cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0) if corners_found else (0, 0, 255), 2)

    # Detection status
    detection_text = "Chessboard DETECTED" if corners_found else "Chessboard NOT FOUND"
    color = (0, 255, 0) if corners_found else (0, 0, 255)
    cv2.putText(image, detection_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2)

    # Quality information
    if ENABLE_BLUR_DETECTION and corners_found:
        quality_text = f"Quality: {'GOOD' if quality_ok else 'POOR'}"
        quality_color = (0, 255, 0) if quality_ok else (0, 165, 255)
        cv2.putText(image, quality_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, quality_color, 2)

        cv2.putText(image, f"Blur: {blur_score:.1f}", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, f"Sharp: {sharpness_score:.1f}", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Instructions
    instructions = [
        "Press 'q' to quit",
        "Press 'c' to capture manually",
        "Auto-capture when good quality"
    ]

    for i, instruction in enumerate(instructions):
        cv2.putText(image, instruction, (10, height - 90 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to capture calibration images using DepthAI v3."""

    # Create output directory
    create_output_directory(OUTPUT_DIR)

    print(f"DepthAI v3 Camera Calibration Capture")
    print(f"======================================")
    print(f"Resolution: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"Chessboard pattern: {CHESSBOARD_SIZE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Target: {MIN_CAPTURES}-{MAX_CAPTURES} images")
    print(
        f"Blur detection: {'Enabled' if ENABLE_BLUR_DETECTION else 'Disabled'}")
    print()

    capture_count = 0
    last_capture_time = 0
    stats = {"total_frames": 0, "detected_frames": 0, "quality_rejected": 0}

    try:
        # Create pipeline
        with dai.Pipeline() as pipeline:
            # Create camera node and configure output
            cam = pipeline.create(dai.node.Camera).build()
            video_queue = cam.requestOutput(
                (IMAGE_WIDTH, IMAGE_HEIGHT)).createOutputQueue()

            # Start pipeline
            pipeline.start()
            print("Pipeline started. Position chessboard in view...\n")

            while pipeline.isRunning() and capture_count < MAX_CAPTURES:
                # Get frame
                video_in = video_queue.get()
                assert isinstance(video_in, dai.ImgFrame)
                frame = video_in.getCvFrame()

                stats["total_frames"] += 1

                # Detect chessboard
                corners_found, corners, gray = detect_chessboard(
                    frame, CHESSBOARD_SIZE)

                # Initialize quality variables
                blur_score = 0
                sharpness_score = 0
                quality_ok = True
                quality_reason = ""

                if corners_found:
                    stats["detected_frames"] += 1

                    # Check image quality
                    quality_ok, blur_score, sharpness_score, quality_reason = \
                        is_image_acceptable(frame, corners)

                    if not quality_ok:
                        stats["quality_rejected"] += 1

                    # Draw corners
                    display_frame = frame.copy()
                    cv2.drawChessboardCorners(display_frame, CHESSBOARD_SIZE,
                                              corners, corners_found)

                    # Auto-capture if enough time passed and quality is good
                    current_time = time.time()
                    if (current_time - last_capture_time > CAPTURE_INTERVAL and
                            (quality_ok or not ENABLE_BLUR_DETECTION)):

                        # Save image
                        filename = f"{IMAGE_PREFIX}{capture_count:03d}.jpg"
                        filepath = os.path.join(OUTPUT_DIR, filename)
                        cv2.imwrite(filepath, frame)

                        capture_count += 1
                        last_capture_time = current_time

                        quality_info = f" ({quality_reason})" if ENABLE_BLUR_DETECTION else ""
                        print(
                            f"Captured {capture_count}/{MAX_CAPTURES}: {filename}{quality_info}")

                        if capture_count >= MIN_CAPTURES:
                            print(f"✓ Minimum {MIN_CAPTURES} captures reached. "
                                  f"Continue for better accuracy or press 'q'")

                    elif not quality_ok and DEBUG_MODE:
                        print(f"Rejected: {quality_reason}")
                else:
                    display_frame = frame.copy()

                # Draw debug info
                if DEBUG_MODE:
                    draw_debug_info(display_frame, corners_found, capture_count,
                                    blur_score, sharpness_score, quality_ok)

                # Display frame
                cv2.imshow(WINDOW_NAME, display_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('c') and corners_found:
                    # Manual capture with quality check
                    if ENABLE_BLUR_DETECTION:
                        quality_ok, blur_score, sharpness_score, quality_reason = \
                            is_image_acceptable(frame, corners)
                        if not quality_ok:
                            print(f"Manual capture rejected: {quality_reason}")
                            continue

                    filename = f"{IMAGE_PREFIX}{capture_count:03d}.jpg"
                    filepath = os.path.join(OUTPUT_DIR, filename)
                    cv2.imwrite(filepath, frame)

                    capture_count += 1
                    quality_info = f" ({quality_reason})" if ENABLE_BLUR_DETECTION else ""
                    print(
                        f"Manual capture {capture_count}/{MAX_CAPTURES}: {filename}{quality_info}")

                elif key == ord('c') and not corners_found:
                    print("Cannot capture: No chessboard detected")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        cv2.destroyAllWindows()

        # Print summary
        print(f"\n{'='*50}")
        print(f"CAPTURE SUMMARY")
        print(f"{'='*50}")
        print(f"Total images captured: {capture_count}")
        print(f"Images saved to: {OUTPUT_DIR}")
        print(f"Detection rate: {stats['detected_frames']}/{stats['total_frames']} "
              f"({stats['detected_frames']/max(stats['total_frames'], 1)*100:.1f}%)")

        if ENABLE_BLUR_DETECTION:
            print(f"Quality rejections: {stats['quality_rejected']}")
            quality_rate = (stats['detected_frames'] - stats['quality_rejected']) / \
                max(stats['detected_frames'], 1) * 100
            print(f"Good quality rate: {quality_rate:.1f}% of detected frames")

        if capture_count >= MIN_CAPTURES:
            print(f"\n✓ SUCCESS: Sufficient images for calibration!")
        else:
            print(f"\n⚠ WARNING: Only {capture_count} images. "
                  f"Recommended minimum: {MIN_CAPTURES}")

        print(f"\nNext step: Run calibration script to compute camera parameters")


if __name__ == "__main__":
    main()
