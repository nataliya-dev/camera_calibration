import cv2
import numpy as np


def main():
    # Chessboard configuration (from your reference script)
    CHESSBOARD_SIZE = (7, 6)  # Internal corners (width, height)

    # Initialize the camera
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print("Chessboard Detection - Press 'q' to quit")
    print(f"Looking for chessboard with {CHESSBOARD_SIZE} internal corners")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break

        # Convert to grayscale for chessboard detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret_chess, corners = cv2.findChessboardCorners(
            gray, CHESSBOARD_SIZE, None)

        # If chessboard is found, draw the corners
        if ret_chess:
            # Refine corner positions for better accuracy
            criteria = (cv2.TERM_CRITERIA_EPS +
                        cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)

            # Draw the corners on the frame
            cv2.drawChessboardCorners(
                frame, CHESSBOARD_SIZE, corners, ret_chess)

            # Add detection status text
            cv2.putText(frame, "Chessboard DETECTED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # Add detection status text
            cv2.putText(frame, "Chessboard NOT FOUND", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow('Chessboard Detection', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
