import cv2
import numpy as np
import pyrealsense2 as rs
import json


class CheckerboardPoseEstimator:
    def __init__(self, chessboard_size=(7, 6), square_size=0.025):
        """
        Initialize the pose estimator

        Args:
            chessboard_size: (width, height) - number of internal corners
            square_size: Size of each square in meters (default: 2.5cm)
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size

        # Prepare object points for the checkerboard
        self.object_points = self._prepare_object_points()

    def _prepare_object_points(self):
        """Prepare 3D points of checkerboard corners in object coordinate system"""
        objp = np.zeros(
            (self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0],
                               0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size  # Scale by actual square size
        return objp

    def get_camera_intrinsics_from_realsense(self, pipeline):
        """Extract camera intrinsics from RealSense camera"""
        # Get camera intrinsics
        profile = pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(
            profile.get_stream(rs.stream.color))
        color_intrinsics = color_profile.get_intrinsics()

        # Convert to OpenCV format
        camera_matrix = np.array([
            [color_intrinsics.fx, 0, color_intrinsics.ppx],
            [0, color_intrinsics.fy, color_intrinsics.ppy],
            [0, 0, 1]
        ], dtype=np.float32)

        # Distortion coefficients
        dist_coeffs = np.array([
            color_intrinsics.coeffs[0],  # k1
            color_intrinsics.coeffs[1],  # k2
            color_intrinsics.coeffs[2],  # p1
            color_intrinsics.coeffs[3],  # p2
            color_intrinsics.coeffs[4]   # k3
        ], dtype=np.float32)

        return camera_matrix, dist_coeffs

    def detect_checkerboard(self, image):
        """Detect checkerboard corners in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(
            gray,
            self.chessboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            # Refine corner positions
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS +
                          cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )

        return ret, corners

    def estimate_pose(self, image_points, camera_matrix, dist_coeffs):
        """Estimate pose of checkerboard relative to camera"""
        success, rvec, tvec = cv2.solvePnP(
            self.object_points,
            image_points,
            camera_matrix,
            dist_coeffs
        )

        if success:
            # Convert rotation vector to rotation matrix
            rmat, _ = cv2.Rodrigues(rvec)
            return True, rvec, tvec, rmat

        return False, None, None, None

    def get_checkerboard_position(self, image, camera_matrix, dist_coeffs):
        """
        Get the 3D position of the checkerboard center in camera coordinates

        Returns:
            success: bool - whether detection was successful
            position: tuple (x, y, z) - position in meters relative to camera
            orientation: tuple - rotation angles in degrees (rx, ry, rz)
        """
        # Detect checkerboard
        ret, corners = self.detect_checkerboard(image)

        if not ret:
            return False, None, None

        # Estimate pose
        success, rvec, tvec, rmat = self.estimate_pose(
            corners, camera_matrix, dist_coeffs)

        if not success:
            return False, None, None

        # Extract position (translation vector gives position of object origin relative to camera)
        position = (float(tvec[0][0]), float(tvec[1][0]), float(tvec[2][0]))

        # # Convert rotation vector to Euler angles (in degrees)
        # rotation_angles = cv2.Rodrigues(rvec)[0]
        # rx = np.degrees(np.arctan2(
        #     rotation_angles[2][1], rotation_angles[2][2]))
        # ry = np.degrees(np.arctan2(-rotation_angles[2][0],
        #                            np.sqrt(rotation_angles[2][1]**2 + rotation_angles[2][2]**2)))
        # rz = np.degrees(np.arctan2(
        #     rotation_angles[1][0], rotation_angles[0][0]))

        # orientation = (rx, ry, rz)

        return True, position, rvec, rmat

    def draw_coordinate_system(self, image, rvec, tvec, camera_matrix, dist_coeffs, axis_length=0.1):
        """Draw coordinate system on the checkerboard"""
        # Define 3D points for coordinate axes
        axis_points = np.float32([
            [0, 0, 0],                    # Origin
            [axis_length, 0, 0],          # X-axis (red)
            [0, axis_length, 0],          # Y-axis (green)
            [0, 0, -axis_length]          # Z-axis (blue)
        ]).reshape(-1, 3)

        # Project 3D points to image plane
        imgpts, _ = cv2.projectPoints(
            axis_points, rvec, tvec, camera_matrix, dist_coeffs)
        imgpts = np.int32(imgpts).reshape(-1, 2)

        # Draw coordinate axes
        origin = tuple(imgpts[0].ravel())
        image = cv2.line(image, origin, tuple(
            imgpts[1].ravel()), (0, 0, 255), 5)  # X-axis: Red
        image = cv2.line(image, origin, tuple(
            imgpts[2].ravel()), (0, 255, 0), 5)  # Y-axis: Green
        image = cv2.line(image, origin, tuple(
            imgpts[3].ravel()), (255, 0, 0), 5)  # Z-axis: Blue

        return image


def main():
    """Example usage with RealSense camera"""
    # Initialize the pose estimator
    # Adjust square_size to match your actual checkerboard (in meters)
    estimator = CheckerboardPoseEstimator(
        chessboard_size=(7, 6),
        square_size=0.025  # 2.5cm squares
    )

    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device('943222071556')
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        # Get camera intrinsics
        camera_matrix, dist_coeffs = estimator.get_camera_intrinsics_from_realsense(
            pipeline)
        print("Camera Matrix:")
        print(camera_matrix)
        print("\nDistortion Coefficients:")
        print(dist_coeffs)

        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # Convert to numpy array
            color_image = np.asanyarray(color_frame.get_data())

            # Get checkerboard position
            success, position, rvec, rmat = estimator.get_checkerboard_position(
                color_image, camera_matrix, dist_coeffs
            )

            if success:

                data = {
                    "position": position,  # This is already a tuple
                    "rvec": rvec.flatten().tolist(),  # Convert to list
                    "rmat": rmat.tolist()  # Convert to list
                }

                with open('pose_data_r1.json', 'w') as f:
                    json.dump(data, f, indent=2)

                # Draw coordinate system on image
                ret, corners = estimator.detect_checkerboard(color_image)
                if ret:
                    success, rvec, tvec, _ = estimator.estimate_pose(
                        corners, camera_matrix, dist_coeffs)
                    if success:
                        color_image = estimator.draw_coordinate_system(
                            color_image, rvec, tvec, camera_matrix, dist_coeffs
                        )
                        # Draw checkerboard corners
                        cv2.drawChessboardCorners(
                            color_image, estimator.chessboard_size, corners, ret)

            # Display the image
            cv2.imshow('Checkerboard Pose Estimation', color_image)

            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
