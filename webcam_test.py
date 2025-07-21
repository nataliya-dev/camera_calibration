import cv2


def main():
    # Initialize the camera (0 is usually the default camera)
    cap = cv2.VideoCapture(2)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Set camera properties (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Press 'q' to quit the camera stream")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting...")
            break

        # Display the frame
        cv2.imshow('Webcam Stream', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
