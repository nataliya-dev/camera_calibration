import pyrealsense2 as rs
import numpy as np
import cv2


def main():
    # Chessboard configuration
    CHESSBOARD_SIZE = (7, 6)  # Internal corners (width, height)

    # Configure depth and color streams
    ctx = rs.context()
    devices = ctx.query_devices()

    print("Available devices:")
    for device in devices:
        print(
            f"Device: {device.get_info(rs.camera_info.name)} - Serial: {device.get_info(rs.camera_info.serial_number)}")

    device_serials = ['913522070103', '943222071556']
    # device_serials = ['838212073725', '913522070103', '943222071556']

    def start_camera(serial):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

        try:
            pipeline.start(config)
            return pipeline
        except Exception as e:
            print(f"Failed to start camera {serial}: {e}")
            return None

    # Start all available cameras
    pipelines = []
    active_serials = []

    for serial in device_serials:
        pipeline = start_camera(serial)
        if pipeline is not None:
            pipelines.append(pipeline)
            active_serials.append(serial)
            print(f"Successfully started camera: {serial}")

    if not pipelines:
        print("Error: No cameras could be started")
        return

    print(f"\nChessboard Detection - Press 'q' to quit")
    print(f"Looking for chessboard with {CHESSBOARD_SIZE} internal corners")
    print(f"Active cameras: {len(pipelines)}")

    try:
        while True:
            frames_data = []

            # Capture frames from all cameras
            for i, pipeline in enumerate(pipelines):
                try:
                    # Wait for a color frame
                    frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()

                    if not color_frame:
                        continue

                    # Convert image to numpy array
                    color_image = np.asanyarray(color_frame.get_data())

                    frames_data.append({
                        'color': color_image,
                        'serial': active_serials[i]
                    })

                except Exception as e:
                    print(
                        f"Error capturing from camera {active_serials[i]}: {e}")
                    continue

            if not frames_data:
                continue

            # Process each camera's frame
            display_frames = []

            for frame_data in frames_data:
                color_image = frame_data['color']
                serial = frame_data['serial']

                # Convert to grayscale for chessboard detection
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

                # Find chessboard corners
                ret_chess, corners = cv2.findChessboardCorners(
                    gray, CHESSBOARD_SIZE, None)

                # Create a copy for display
                display_frame = color_image.copy()

                # If chessboard is found, draw the corners
                if ret_chess:
                    # Refine corner positions for better accuracy
                    criteria = (cv2.TERM_CRITERIA_EPS +
                                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), criteria)

                    # Draw the corners on the frame
                    cv2.drawChessboardCorners(
                        display_frame, CHESSBOARD_SIZE, corners, ret_chess)

                    # Add detection status text
                    cv2.putText(display_frame, "Chessboard DETECTED", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Add detection status text
                    cv2.putText(display_frame, "Chessboard NOT FOUND", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Add camera serial number
                cv2.putText(display_frame, f"Camera: {serial}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                display_frames.append(display_frame)

            # Display all camera feeds
            for i, display_frame in enumerate(display_frames):
                window_name = f'RealSense Chessboard Detection - Camera {i+1}'
                cv2.imshow(window_name, display_frame)

            # Break the loop when 'q' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        # Stop streaming from all pipelines
        for pipeline in pipelines:
            try:
                pipeline.stop()
            except:
                pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
