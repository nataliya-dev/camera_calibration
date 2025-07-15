import cv2
import os
import time
import numpy as np

# Configuration parameters
CAMERA_INDEX = 0  # Default camera (0 for built-in, 1+ for external)
CAPTURE_INTERVAL = 2.0  # Seconds between automatic captures (increased for stability)
MIN_CAPTURES = 20  # Minimum number of images needed for calibration
MAX_CAPTURES = 30  # Maximum number of images to capture
OUTPUT_DIR = "calibration_images"  # Directory to save captured images
IMAGE_PREFIX = "calib_"  # Prefix for saved image files

# Camera settings - IMPORTANT: Use the same resolution you'll use in your application!
TARGET_WIDTH = 640  # Set to your target application width
TARGET_HEIGHT = 480  # Set to your target application height
USE_AUTOFOCUS = "disabled"  # Options: "enabled", "disabled", "fixed"
# "enabled" - autofocus throughout (may hurt calibration accuracy)
# "disabled" - manual focus (may be blurry if not set right)  
# "fixed" - autofocus initially, then fix (recommended)

# Blur detection settings
ENABLE_BLUR_DETECTION = True  # Reject blurry images automatically
BLUR_THRESHOLD = 80.0  # Laplacian variance threshold (lower = more strict)
MIN_SHARPNESS_SCORE = 40.0  # Minimum sharpness for chessboard region

# Chessboard configuration
CHESSBOARD_SIZE = (7, 6)  # Internal corners (width, height)
SQUARE_SIZE = 25.0  # Size of chessboard square in mm (adjust to your board)

# Display settings
WINDOW_NAME = "Camera Calibration - Press 'q' to quit, 'c' to capture manually"
DEBUG_MODE = True  # Show detection results and statistics

def create_output_directory(directory):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def detect_chessboard(image, pattern_size):
    """
    Detect chessboard corners in the image.
    
    Args:
        image: Input image
        pattern_size: Tuple of (width, height) internal corners
    
    Returns:
        (success, corners, gray_image)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    
    if ret:
        # Refine corner positions for better accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
    return ret, corners, gray

def calculate_blur_score(image):
    """
    Calculate blur score using Laplacian variance method.
    Higher scores indicate sharper images.
    
    Args:
        image: Input image (grayscale or color)
    
    Returns:
        blur_score: Higher = sharper, Lower = blurrier
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate Laplacian variance (edge detection)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = laplacian.var()
    
    return blur_score

def calculate_region_sharpness(image, corners, pattern_size):
    """
    Calculate sharpness specifically in the chessboard region.
    
    Args:
        image: Input image
        corners: Detected chessboard corners
        pattern_size: Chessboard pattern size
    
    Returns:
        sharpness_score: Sharpness measure for the chessboard region
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Get bounding box of chessboard
    x_coords = corners[:, 0, 0]
    y_coords = corners[:, 0, 1]
    
    x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
    y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
    
    # Add some padding
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

def is_image_acceptable(image, corners, pattern_size):
    """
    Check if image quality is acceptable for calibration.
    
    Args:
        image: Input image
        corners: Detected chessboard corners  
        pattern_size: Chessboard pattern size
    
    Returns:
        (is_acceptable, blur_score, sharpness_score, reason)
    """
    if not ENABLE_BLUR_DETECTION:
        return True, 0, 0, "Blur detection disabled"
    
    # Calculate overall blur score
    blur_score = calculate_blur_score(image)
    
    # Calculate region-specific sharpness
    sharpness_score = calculate_region_sharpness(image, corners, pattern_size)
    
    # Check thresholds
    if blur_score < BLUR_THRESHOLD:
        return False, blur_score, sharpness_score, f"Too blurry (score: {blur_score:.1f} < {BLUR_THRESHOLD})"
    
    if sharpness_score < MIN_SHARPNESS_SCORE:
        return False, blur_score, sharpness_score, f"Chessboard region not sharp enough (score: {sharpness_score:.1f} < {MIN_SHARPNESS_SCORE})"
    
    return True, blur_score, sharpness_score, "Acceptable quality"


def draw_debug_info(image, corners_found, capture_count, max_captures, blur_score=0, sharpness_score=0, quality_ok=True):
    """Draw debug information on the image."""
    height, width = image.shape[:2]
    
    # Status text
    status_text = f"Captures: {capture_count}/{max_captures}"
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
        quality_color = (0, 255, 0) if quality_ok else (0, 165, 255)  # Green or Orange
        cv2.putText(image, quality_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, quality_color, 2)
        
        blur_text = f"Blur: {blur_score:.1f}"
        cv2.putText(image, blur_text, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1)
        
        sharp_text = f"Sharp: {sharpness_score:.1f}"
        cv2.putText(image, sharp_text, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1)
    
    # Instructions
    instructions = [
        "Press 'q' to quit",
        "Press 'c' to capture manually",
        "Auto-capture when good quality detected" if ENABLE_BLUR_DETECTION else "Auto-capture when detected"
    ]
    
    for i, instruction in enumerate(instructions):
        cv2.putText(image, instruction, (10, height - 90 + i * 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def main():
    """Main function to capture calibration images."""
    
    # Create output directory
    create_output_directory(OUTPUT_DIR)
    
    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_INDEX}")
        return
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Handle autofocus based on setting
    if USE_AUTOFOCUS == "enabled":
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        print("Autofocus: Enabled throughout capture")
    elif USE_AUTOFOCUS == "disabled":
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        print("Autofocus: Disabled (manual focus)")
    elif USE_AUTOFOCUS == "fixed":
        # Enable autofocus initially, then disable after focusing
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        print("Autofocus: Enabled for initial focus, then will be fixed")
        print("Position chessboard at typical distance and wait for focus...")
        time.sleep(3)  # Give time to focus
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        print("Autofocus: Now disabled - focus is fixed")
    
    # Get actual resolution (may differ from requested)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Camera initialized.")
    print(f"Requested resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    print(f"Actual resolution: {actual_width}x{actual_height}")
    if actual_width != TARGET_WIDTH or actual_height != TARGET_HEIGHT:
        print("⚠ Warning: Camera resolution differs from requested!")
    print(f"Capturing images for chessboard pattern: {CHESSBOARD_SIZE}")
    print(f"Images will be saved to: {OUTPUT_DIR}")
    print(f"Target: {MIN_CAPTURES}-{MAX_CAPTURES} images")
    
    capture_count = 0
    last_capture_time = 0
    detection_stats = {"total_frames": 0, "detected_frames": 0, "quality_rejected": 0}
    
    try:
        while capture_count < MAX_CAPTURES:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            detection_stats["total_frames"] += 1
            
            # Detect chessboard
            corners_found, corners, gray = detect_chessboard(frame, CHESSBOARD_SIZE)
            
            # Initialize quality variables
            blur_score = 0
            sharpness_score = 0
            quality_ok = True
            quality_reason = ""
            
            if corners_found:
                detection_stats["detected_frames"] += 1
                
                # Check image quality
                quality_ok, blur_score, sharpness_score, quality_reason = is_image_acceptable(
                    frame, corners, CHESSBOARD_SIZE
                )
                
                if not quality_ok:
                    detection_stats["quality_rejected"] += 1
                
                # Draw corners on the image
                original_frame = frame.copy()
                cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners, corners_found)
                
                # Auto-capture if enough time has passed and quality is good
                current_time = time.time()
                if (current_time - last_capture_time > CAPTURE_INTERVAL and 
                    (quality_ok or not ENABLE_BLUR_DETECTION)):
                    
                    # Save image
                    filename = f"{IMAGE_PREFIX}{capture_count:03d}.jpg"
                    filepath = os.path.join(OUTPUT_DIR, filename)
                    cv2.imwrite(filepath, original_frame)
                    
                    capture_count += 1
                    last_capture_time = current_time
                    
                    quality_info = f" (Quality: {quality_reason})" if ENABLE_BLUR_DETECTION else ""
                    print(f"Captured image {capture_count}/{MAX_CAPTURES}: {filename}{quality_info}")
                    
                    if capture_count >= MIN_CAPTURES:
                        print(f"Minimum captures ({MIN_CAPTURES}) reached. Continue for better accuracy or press 'q' to stop.")
                
                elif not quality_ok and DEBUG_MODE:
                    print(f"Rejected frame: {quality_reason}")
            
            # Draw debug information
            if DEBUG_MODE:
                draw_debug_info(frame, corners_found, capture_count, MAX_CAPTURES, 
                              blur_score, sharpness_score, quality_ok)
            
            # Display frame
            cv2.imshow(WINDOW_NAME, frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and corners_found:
                # Manual capture - check quality first
                if ENABLE_BLUR_DETECTION:
                    quality_ok, blur_score, sharpness_score, quality_reason = is_image_acceptable(
                        frame, corners, CHESSBOARD_SIZE
                    )
                    if not quality_ok:
                        print(f"Manual capture rejected: {quality_reason}")
                        print("Try better lighting, steadier hands, or move closer/farther")
                        continue
                
                # Manual capture
                filename = f"{IMAGE_PREFIX}{capture_count:03d}.jpg"
                filepath = os.path.join(OUTPUT_DIR, filename)
                cv2.imwrite(filepath, frame)
                
                capture_count += 1
                quality_info = f" (Quality: {quality_reason})" if ENABLE_BLUR_DETECTION else ""
                print(f"Manually captured image {capture_count}/{MAX_CAPTURES}: {filename}{quality_info}")
            elif key == ord('c') and not corners_found:
                print("Cannot capture: No chessboard detected")
    
    except KeyboardInterrupt:
        print("\nCapture interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        print(f"\nCapture Summary:")
        print(f"Total images captured: {capture_count}")
        print(f"Images saved to: {OUTPUT_DIR}")
        print(f"Detection rate: {detection_stats['detected_frames']}/{detection_stats['total_frames']} frames ({detection_stats['detected_frames']/max(detection_stats['total_frames'], 1)*100:.1f}%)")
        
        if ENABLE_BLUR_DETECTION:
            print(f"Quality rejections: {detection_stats['quality_rejected']} frames")
            quality_rate = (detection_stats['detected_frames'] - detection_stats['quality_rejected']) / max(detection_stats['detected_frames'], 1) * 100
            print(f"Good quality rate: {quality_rate:.1f}% of detected frames")
        
        if capture_count >= MIN_CAPTURES:
            print(f"✓ Sufficient images captured for calibration!")
        else:
            print(f"⚠ Warning: Only {capture_count} images captured. Recommended minimum: {MIN_CAPTURES}")
        
        print(f"Next step: Run the calibration script to compute camera parameters.")

if __name__ == "__main__":
    main()