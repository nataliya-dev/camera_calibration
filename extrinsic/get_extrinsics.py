#!/usr/bin/env python3
"""
Robot-Camera Extrinsic Calibration Script

This script performs extrinsic calibration to find:
1. Transform from end-effector to chessboard (T_ee_to_cb)
2. Transform from camera to robot base (T_cam_to_base)

The optimization minimizes reprojection error of chessboard corners.
"""

import numpy as np
import cv2
import json
import glob
import os
import yaml
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ExtrinsicCalibrator:
    def __init__(self, ee_poses_file: str, intrinsics_file: str, camera_serial: str,
                 chessboard_size: Tuple[int, int] = (7, 6),
                 square_size: float = 0.025):
        """
        Initialize the calibrator.

        Args:
            ee_poses_file: Path to JSON file with end-effector poses
            intrinsics_file: Path to YAML file with camera intrinsics
            camera_serial: Serial number of the camera to use
            chessboard_size: (width, height) number of inner corners
            square_size: Size of chessboard squares in meters
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.camera_serial = camera_serial

        # Load end-effector poses
        with open(ee_poses_file, 'r') as f:
            self.ee_poses = json.load(f)

        # Load camera intrinsics
        camera_params = self._load_camera_intrinsics(
            intrinsics_file, camera_serial)

        # Camera intrinsics
        self.camera_matrix = np.array([
            [camera_params['fx'], 0, camera_params['cx']],
            [0, camera_params['fy'], camera_params['cy']],
            [0, 0, 1]
        ])

        # Distortion coefficients
        if camera_params.get('has_dist_coeff', 0):
            self.dist_coeffs = np.array([
                camera_params.get('dist_k0', 0.0),
                camera_params.get('dist_k1', 0.0),
                camera_params.get('dist_px', 0.0),
                camera_params.get('dist_py', 0.0),
                camera_params.get('dist_k2', 0.0)
            ])
        else:
            self.dist_coeffs = np.zeros(5)

        # Store image dimensions
        self.img_width = camera_params.get('img_width', 1920)
        self.img_height = camera_params.get('img_height', 1080)

        # Storage for detected corners and corresponding poses
        self.image_points = []
        self.ee_poses_valid = []
        self.sample_names = []

        # Generate 3D chessboard points
        self.object_points_3d = self._generate_chessboard_3d()

        print(f"Initialized calibrator for camera {camera_serial}")
        print(f"Chessboard size: {chessboard_size} corners")
        print(f"Square size: {square_size} m")
        print(f"Camera matrix:\n{self.camera_matrix}")
        print(f"Distortion coefficients: {self.dist_coeffs}")

    def _load_camera_intrinsics(self, intrinsics_file: str, camera_serial: str) -> Dict:
        """Load camera intrinsics from YAML file based on serial number."""
        with open(intrinsics_file, 'r') as f:
            all_cameras = yaml.safe_load(f)

        # Find camera by serial number
        camera_key = f"camera_{camera_serial}"
        if camera_key not in all_cameras:
            raise ValueError(f"Camera {camera_serial} not found in {intrinsics_file}. "
                             f"Available cameras: {list(all_cameras.keys())}")

        camera_params = all_cameras[camera_key]
        print(f"Loaded intrinsics for {camera_key}")
        return camera_params

    def _generate_chessboard_3d(self) -> np.ndarray:
        """Generate 3D coordinates of chessboard corners in chessboard frame."""
        objp = np.zeros(
            (self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0],
                               0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp

    def detect_chessboard_corners(self, images_dir: str, camera_name: str = "r1") -> int:
        """
        Detect chessboard corners in all images.

        Args:
            images_dir: Directory containing calibration images
            camera_name: Subdirectory name for camera images

        Returns:
            Number of successfully processed images
        """
        # Find all image files
        image_pattern = os.path.join(images_dir, camera_name, "sample_*.jpg")
        image_files = sorted(glob.glob(image_pattern))

        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        successful_detections = 0

        for img_file in image_files:
            # Extract sample name from filename
            sample_name = os.path.splitext(os.path.basename(img_file))[0]

            # Check if we have corresponding EE pose
            if sample_name not in self.ee_poses:
                print(f"Warning: No EE pose found for {sample_name}")
                continue

            # Load and process image
            img = cv2.imread(img_file)
            if img is None:
                print(f"Warning: Could not load image {img_file}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(
                gray, self.chessboard_size, None)

            if ret:
                # Refine corner positions
                corners2 = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria)

                # Store results
                self.image_points.append(corners2.reshape(-1, 2))
                self.ee_poses_valid.append(self.ee_poses[sample_name])
                self.sample_names.append(sample_name)
                successful_detections += 1

                print(f"✓ Detected corners in {sample_name}")
            else:
                print(f"✗ Could not detect corners in {sample_name}")

        print(
            f"\nSuccessfully detected corners in {successful_detections}/{len(image_files)} images")
        return successful_detections

    def quaternion_to_matrix(self, quat: List[float]) -> np.ndarray:
        """Convert quaternion [x, y, z, w] to rotation matrix."""
        x, y, z, w = quat
        return R.from_quat([x, y, z, w]).as_matrix()

    def rpy_to_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert roll, pitch, yaw to rotation matrix."""
        return R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()

    def pose_to_transform(self, position: List[float], orientation: List[float]) -> np.ndarray:
        """Convert position and quaternion to 4x4 transformation matrix."""
        T = np.eye(4)
        T[:3, :3] = self.quaternion_to_matrix(orientation)
        T[:3, 3] = position
        return T

    def rpy_pos_to_transform(self, params: np.ndarray) -> np.ndarray:
        """Convert [roll, pitch, yaw, x, y, z] to 4x4 transformation matrix."""
        roll, pitch, yaw, x, y, z = params
        T = np.eye(4)
        T[:3, :3] = self.rpy_to_matrix(roll, pitch, yaw)
        T[:3, 3] = [x, y, z]
        return T

    def project_points(self, points_3d: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Project 3D points to image plane."""
        projected, _ = cv2.projectPoints(
            points_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        return projected.reshape(-1, 2)

    def objective_function(self, params: np.ndarray) -> float:
        """
        Objective function for optimization.

        Args:
            params: [ee_to_cb_rpy(3), ee_to_cb_pos(3), cam_to_base_rpy(3), cam_to_base_pos(3)]

        Returns:
            Total reprojection error
        """
        try:
            # Split parameters
            ee_to_cb_params = params[:6]  # [roll, pitch, yaw, x, y, z]
            cam_to_base_params = params[6:12]  # [roll, pitch, yaw, x, y, z]

            # Build transformation matrices
            T_ee_to_cb = self.rpy_pos_to_transform(ee_to_cb_params)
            T_cam_to_base = self.rpy_pos_to_transform(cam_to_base_params)
            T_base_to_cam = np.linalg.inv(T_cam_to_base)

            total_error = 0.0

            for i, (img_points, ee_pose) in enumerate(zip(self.image_points, self.ee_poses_valid)):
                # Transform from robot base to end-effector
                T_base_to_ee = self.pose_to_transform(
                    ee_pose['position'], ee_pose['orientation'])

                # Complete transformation: camera -> base -> ee -> chessboard
                T_cam_to_cb = T_base_to_cam @ T_base_to_ee @ T_ee_to_cb

                # Extract rotation and translation for OpenCV
                R_cam_to_cb = T_cam_to_cb[:3, :3]
                t_cam_to_cb = T_cam_to_cb[:3, 3]

                # Convert to rodrigues vector
                rvec, _ = cv2.Rodrigues(R_cam_to_cb)
                tvec = t_cam_to_cb.reshape(3, 1)

                # Project chessboard points to image
                projected_points = self.project_points(
                    self.object_points_3d, rvec, tvec)

                # Calculate reprojection error
                error = np.sum((projected_points - img_points) ** 2)
                total_error += error

            return total_error

        except Exception as e:
            # Return large error for invalid parameters
            return 1e10

    def generate_random_initial_guess(self, seed: Optional[int] = None) -> np.ndarray:
        """Generate a random initial guess for the optimization parameters."""
        if seed is not None:
            np.random.seed(seed)

        # Generate random parameters within reasonable bounds
        initial_guess = np.array([
            # ee_to_cb: chessboard orientation and position relative to EE
            np.random.uniform(-0.3, 0.3),  # roll
            np.random.uniform(-0.3, 0.3),  # pitch
            np.random.uniform(-0.3, 0.3),  # yaw
            np.random.uniform(-0.1, 0.1),  # x
            np.random.uniform(-0.1, 0.1),  # y
            np.random.uniform(0.05, 0.2),  # z (chessboard in front of EE)

            # cam_to_base: camera position and orientation relative to robot base
            np.random.uniform(-0.5, 0.5),  # roll
            np.random.uniform(-0.5, 0.5),  # pitch
            np.random.uniform(-0.5, 0.5),  # yaw
            np.random.uniform(-1.0, 1.0),  # x
            np.random.uniform(-1.0, 1.0),  # y
            np.random.uniform(0.2, 1.5),   # z (camera above base)
        ])

        return initial_guess

    def get_optimization_bounds(self) -> List[Tuple]:
        """Get optimization bounds for parameters."""
        bounds = [
            (-np.pi/2, np.pi/2),   # ee_to_cb roll
            (-np.pi/2, np.pi/2),   # ee_to_cb pitch
            (-np.pi/2, np.pi/2),   # ee_to_cb yaw
            (-0.2, 0.2),           # ee_to_cb x
            (-0.2, 0.2),           # ee_to_cb y
            (0.0, 0.3),            # ee_to_cb z
            (-np.pi/2, np.pi/2),   # cam_to_base roll
            (-np.pi/2, np.pi/2),   # cam_to_base pitch
            (-np.pi/2, np.pi/2),   # cam_to_base yaw
            (-2.0, 2.0),           # cam_to_base x
            (-2.0, 2.0),           # cam_to_base y
            (0.0, 2.0)             # cam_to_base z
        ]
        return bounds

    def multi_start_calibrate(self, num_starts: int = 20, use_default_start: bool = True,
                              use_differential_evolution: bool = True) -> Dict:
        """
        Perform multi-start optimization to find global minimum.

        Args:
            num_starts: Number of random starting points
            use_default_start: Whether to include the default initial guess
            use_differential_evolution: Whether to use differential evolution first

        Returns:
            Dictionary with best calibration results and all attempts
        """
        if len(self.image_points) == 0:
            raise ValueError(
                "No valid image points detected. Run detect_chessboard_corners first.")

        bounds = self.get_optimization_bounds()
        all_results = []
        best_result = None
        best_error = float('inf')

        # Use differential evolution first for global optimization
        if use_differential_evolution:
            print("Running differential evolution for global optimization...")
            try:
                de_result = differential_evolution(
                    self.objective_function,
                    bounds,
                    maxiter=300,
                    popsize=15,
                    seed=42,
                    disp=False
                )

                if de_result.success:
                    de_refined = self._single_optimization(de_result.x, bounds)
                    all_results.append(
                        {**de_refined, 'method': 'differential_evolution'})

                    if de_refined['success'] and de_refined['final_error'] < best_error:
                        best_error = de_refined['final_error']
                        best_result = de_refined
                        print(f"  ✓ DE best error: {best_error:.2f}")

            except Exception as e:
                print(f"  ✗ Differential evolution failed: {e}")

        # Include default initial guess if requested
        start_count = 0
        if use_default_start:
            default_guess = np.array([
                0.0, 0.0, 0.0,      # ee_to_cb rpy
                0.0, 0.0, 0.1,      # ee_to_cb pos
                0.0, 0.0, 0.0,      # cam_to_base rpy
                0.5, 0.0, 0.5       # cam_to_base pos
            ])

            print(
                f"Attempt {start_count + 1}/{num_starts + 1}: Default initial guess")
            result = self._single_optimization(default_guess, bounds)
            result['method'] = 'default_start'
            all_results.append(result)

            if result['success'] and result['final_error'] < best_error:
                best_error = result['final_error']
                best_result = result
                print(f"  ✓ New best error: {best_error:.2f}")
            else:
                print(
                    f"  ✗ Error: {result['final_error']:.2f}, Success: {result['success']}")

            start_count = 1

        # Random starts
        for i in range(num_starts):
            print(
                f"Attempt {start_count + i + 1}/{num_starts + (1 if use_default_start else 0)}: Random start {i + 1}")

            initial_guess = self.generate_random_initial_guess(seed=i)
            result = self._single_optimization(initial_guess, bounds)
            result['method'] = 'random_start'
            all_results.append(result)

            if result['success'] and result['final_error'] < best_error:
                best_error = result['final_error']
                best_result = result
                print(f"  ✓ New best error: {best_error:.2f}")
            else:
                print(
                    f"  ✗ Error: {result['final_error']:.2f}, Success: {result['success']}")

        # Analyze results
        successful_results = [r for r in all_results if r['success']]
        print(f"\nOptimization Summary:")
        print(f"Total attempts: {len(all_results)}")
        print(f"Successful attempts: {len(successful_results)}")

        if successful_results:
            errors = [r['final_error'] for r in successful_results]
            print(f"Best error: {min(errors):.2f}")
            print(f"Worst error: {max(errors):.2f}")
            print(f"Mean error: {np.mean(errors):.2f}")
            print(f"Std error: {np.std(errors):.2f}")

        if best_result is None:
            print("ERROR: No successful optimization found!")
            best_result = min(all_results, key=lambda x: x['final_error'])

        # Add summary statistics
        best_result['all_results'] = all_results
        best_result['num_successful'] = len(successful_results)
        best_result['convergence_rate'] = len(
            successful_results) / len(all_results)

        return best_result

    def _single_optimization(self, initial_guess: np.ndarray, bounds: List[Tuple]) -> Dict:
        """Perform single optimization run."""
        try:
            result = minimize(
                self.objective_function,
                initial_guess,
                method='L-BFGS-B',
                bounds=bounds,
                options={'disp': False, 'maxiter': 1000, 'ftol': 1e-9}
            )

            if result.success:
                # Build transformation matrices
                optimal_params = result.x
                ee_to_cb_params = optimal_params[:6]
                cam_to_base_params = optimal_params[6:12]

                T_ee_to_cb = self.rpy_pos_to_transform(ee_to_cb_params)
                T_cam_to_base = self.rpy_pos_to_transform(cam_to_base_params)

                return {
                    'success': True,
                    'final_error': result.fun,
                    'num_iterations': result.nit,
                    'T_ee_to_cb': T_ee_to_cb,
                    'T_cam_to_base': T_cam_to_base,
                    'ee_to_cb_rpy': ee_to_cb_params[:3],
                    'ee_to_cb_pos': ee_to_cb_params[3:6],
                    'cam_to_base_rpy': cam_to_base_params[:3],
                    'cam_to_base_pos': cam_to_base_params[3:6],
                    'reprojection_error_per_image': self._calculate_per_image_errors(optimal_params),
                    'initial_guess': initial_guess.copy()
                }
            else:
                return {
                    'success': False,
                    'final_error': result.fun,
                    'num_iterations': result.nit,
                    'message': result.message,
                    'initial_guess': initial_guess.copy()
                }

        except Exception as e:
            return {
                'success': False,
                'final_error': 1e10,
                'num_iterations': 0,
                'message': str(e),
                'initial_guess': initial_guess.copy()
            }

    def _calculate_per_image_errors(self, params: np.ndarray) -> List[float]:
        """Calculate reprojection error for each image."""
        errors = []

        ee_to_cb_params = params[:6]
        cam_to_base_params = params[6:12]

        T_ee_to_cb = self.rpy_pos_to_transform(ee_to_cb_params)
        T_cam_to_base = self.rpy_pos_to_transform(cam_to_base_params)
        T_base_to_cam = np.linalg.inv(T_cam_to_base)

        for img_points, ee_pose in zip(self.image_points, self.ee_poses_valid):
            T_base_to_ee = self.pose_to_transform(
                ee_pose['position'], ee_pose['orientation'])
            T_cam_to_cb = T_base_to_cam @ T_base_to_ee @ T_ee_to_cb

            R_cam_to_cb = T_cam_to_cb[:3, :3]
            t_cam_to_cb = T_cam_to_cb[:3, 3]

            rvec, _ = cv2.Rodrigues(R_cam_to_cb)
            tvec = t_cam_to_cb.reshape(3, 1)

            projected_points = self.project_points(
                self.object_points_3d, rvec, tvec)
            error = np.sqrt(np.mean((projected_points - img_points) ** 2))
            errors.append(error)

        return errors

    def validate_calibration(self, results: Dict) -> Dict:
        """Validate calibration results with additional metrics."""
        if not results['success']:
            return {'valid': False, 'reason': 'Optimization failed'}

        validation = {'valid': True, 'warnings': [], 'metrics': {}}

        # Check reprojection errors
        errors = results['reprojection_error_per_image']
        avg_error = np.mean(errors)
        max_error = np.max(errors)
        std_error = np.std(errors)

        validation['metrics']['avg_reprojection_error'] = avg_error
        validation['metrics']['max_reprojection_error'] = max_error
        validation['metrics']['std_reprojection_error'] = std_error

        # Validation criteria
        if avg_error > 2.0:
            validation['warnings'].append(
                f"High average reprojection error: {avg_error:.2f} pixels")
        if max_error > 5.0:
            validation['warnings'].append(
                f"Very high maximum error: {max_error:.2f} pixels")
        if std_error > 1.0:
            validation['warnings'].append(
                f"High error variance: {std_error:.2f} pixels")

        # Check transform reasonableness
        ee_pos = results['ee_to_cb_pos']
        cam_pos = results['cam_to_base_pos']

        if np.linalg.norm(ee_pos) > 0.5:
            validation['warnings'].append(
                f"Large EE-to-chessboard distance: {np.linalg.norm(ee_pos):.3f}m")
        if np.linalg.norm(cam_pos) > 3.0:
            validation['warnings'].append(
                f"Large camera-to-base distance: {np.linalg.norm(cam_pos):.3f}m")

        # Check for degenerate configurations
        if len(self.image_points) < 10:
            validation['warnings'].append(
                f"Few calibration images: {len(self.image_points)}")

        validation['overall_quality'] = 'good' if len(
            validation['warnings']) == 0 else 'acceptable' if avg_error < 3.0 else 'poor'

        return validation

    def visualize_results(self, results: Dict, save_path: Optional[str] = None):
        """Visualize calibration results."""
        all_results = results.get('all_results', [])

        if all_results:
            # Multi-start visualization
            self.visualize_multi_start_results(results, save_path)
        else:
            # Single result visualization
            self._visualize_single_result(results, save_path)

    def _visualize_single_result(self, results: Dict, save_path: Optional[str] = None):
        """Visualize single calibration result."""
        errors = results['reprojection_error_per_image']

        plt.figure(figsize=(12, 8))

        # Plot 1: Reprojection errors
        plt.subplot(2, 2, 1)
        plt.plot(errors, 'bo-')
        plt.xlabel('Image Index')
        plt.ylabel('RMS Reprojection Error (pixels)')
        plt.title('Per-Image Reprojection Error')
        plt.grid(True)

        # Plot 2: Error histogram
        plt.subplot(2, 2, 2)
        plt.hist(errors, bins=20, alpha=0.7)
        plt.xlabel('RMS Reprojection Error (pixels)')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.grid(True)

        # Plot 3: EE to Chessboard transform
        plt.subplot(2, 2, 3)
        rpy = np.degrees(results['ee_to_cb_rpy'])
        pos = results['ee_to_cb_pos']
        labels = ['Roll', 'Pitch', 'Yaw', 'X', 'Y', 'Z']
        values = np.concatenate([rpy, pos])
        colors = ['red'] * 3 + ['blue'] * 3
        bars = plt.bar(labels, values, color=colors, alpha=0.7)
        plt.ylabel('Value')
        plt.title('EE to Chessboard Transform')
        plt.xticks(rotation=45)

        # Plot 4: Camera to Base transform
        plt.subplot(2, 2, 4)
        rpy = np.degrees(results['cam_to_base_rpy'])
        pos = results['cam_to_base_pos']
        values = np.concatenate([rpy, pos])
        bars = plt.bar(labels, values, color=colors, alpha=0.7)
        plt.ylabel('Value')
        plt.title('Camera to Base Transform')
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")

        plt.show()

    def visualize_multi_start_results(self, results: Dict, save_path: Optional[str] = None):
        """Visualize results from multi-start optimization."""
        all_results = results.get('all_results', [])
        if not all_results:
            print("No multi-start results to visualize")
            return

        successful_results = [r for r in all_results if r['success']]
        failed_results = [r for r in all_results if not r['success']]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Plot 1: Error convergence
        ax = axes[0, 0]
        if successful_results:
            errors = [r['final_error'] for r in successful_results]
            methods = [r.get('method', 'unknown') for r in successful_results]
            colors = ['green' if m == 'differential_evolution' else 'blue' if m == 'default_start' else 'orange'
                      for m in methods]
            ax.scatter(range(len(errors)), errors, c=colors, alpha=0.7)
        if failed_results:
            errors = [r['final_error'] for r in failed_results]
            ax.scatter(range(len(successful_results), len(all_results)), errors,
                       c='red', alpha=0.7, marker='x')
        ax.set_xlabel('Attempt Number')
        ax.set_ylabel('Final Error')
        ax.set_title('Optimization Results by Attempt')
        ax.grid(True)

        # Add legend
        legend_elements = []
        if any(r.get('method') == 'differential_evolution' for r in successful_results):
            legend_elements.append(plt.Line2D(
                [0], [0], marker='o', color='w', markerfacecolor='green', label='DE'))
        if any(r.get('method') == 'default_start' for r in successful_results):
            legend_elements.append(plt.Line2D(
                [0], [0], marker='o', color='w', markerfacecolor='blue', label='Default'))
        if any(r.get('method') == 'random_start' for r in successful_results):
            legend_elements.append(plt.Line2D(
                [0], [0], marker='o', color='w', markerfacecolor='orange', label='Random'))
        if failed_results:
            legend_elements.append(plt.Line2D(
                [0], [0], marker='x', color='red', label='Failed'))
        if legend_elements:
            ax.legend(handles=legend_elements)

        # Plot 2: Error histogram
        ax = axes[0, 1]
        if successful_results:
            errors = [r['final_error'] for r in successful_results]
            ax.hist(errors, bins=min(10, len(errors)),
                    alpha=0.7, color='green')
        ax.set_xlabel('Final Error')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Distribution (Successful Only)')
        ax.grid(True)

        # Plot 3: Best solution reprojection errors
        ax = axes[0, 2]
        if 'reprojection_error_per_image' in results:
            errors = results['reprojection_error_per_image']
            ax.plot(errors, 'bo-')
            ax.set_xlabel('Image Index')
            ax.set_ylabel('RMS Error (pixels)')
            ax.set_title('Best Solution: Per-Image Error')
            ax.grid(True)

        # Plot 4: EE to Chessboard transform
        ax = axes[1, 0]
        if 'ee_to_cb_rpy' in results:
            rpy = np.degrees(results['ee_to_cb_rpy'])
            pos = results['ee_to_cb_pos']
            labels = ['Roll', 'Pitch', 'Yaw', 'X', 'Y', 'Z']
            values = np.concatenate([rpy, pos])
            colors = ['red'] * 3 + ['blue'] * 3
            bars = ax.bar(labels, values, color=colors, alpha=0.7)
            ax.set_ylabel('Value')
            ax.set_title('EE to Chessboard Transform')
            plt.setp(ax.get_xticklabels(), rotation=45)

        # Plot 5: Camera to Base transform
        ax = axes[1, 1]
        if 'cam_to_base_rpy' in results:
            rpy = np.degrees(results['cam_to_base_rpy'])
            pos = results['cam_to_base_pos']
            values = np.concatenate([rpy, pos])
            bars = ax.bar(labels, values, color=colors, alpha=0.7)
            ax.set_ylabel('Value')
            ax.set_title('Camera to Base Transform')
            plt.setp(ax.get_xticklabels(), rotation=45)

        # Plot 6: Convergence statistics
        ax = axes[1, 2]
        stats_labels = ['Total\nAttempts',
                        'Successful\nAttempts', 'Convergence\nRate (%)']
        stats_values = [
            len(all_results),
            results.get('num_successful', 0),
            results.get('convergence_rate', 0) * 100
        ]
        bars = ax.bar(stats_labels, stats_values, color=[
                      'blue', 'green', 'orange'], alpha=0.7)
        ax.set_ylabel('Count / Percentage')
        ax.set_title('Optimization Statistics')

        # Add value labels on bars
        for bar, value in zip(bars, stats_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(stats_values),
                    f'{value:.1f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Multi-start visualization saved to {save_path}")

        plt.show()

    def print_results(self, results: Dict):
        """Print calibration results in a formatted way."""
        print("\n" + "="*60)
        print("EXTRINSIC CALIBRATION RESULTS")
        print("="*60)

        print(f"Optimization Success: {results['success']}")
        print(
            f"Final Reprojection Error: {results['final_error']:.2f} pixels²")
        print(
            f"RMS Reprojection Error: {np.sqrt(results['final_error'] / len(self.image_points)):.2f} pixels")
        print(f"Number of Iterations: {results['num_iterations']}")
        print(f"Images Used: {len(self.image_points)}")

        # Print convergence statistics if available
        if 'convergence_rate' in results:
            print(f"Convergence Rate: {results['convergence_rate']*100:.1f}%")
            print(f"Successful Attempts: {results['num_successful']}")

        print("\n" + "-"*40)
        print("END-EFFECTOR TO CHESSBOARD TRANSFORM")
        print("-"*40)
        rpy_deg = np.degrees(results['ee_to_cb_rpy'])
        print(
            f"Roll:  {rpy_deg[0]:8.3f}° ({results['ee_to_cb_rpy'][0]:8.6f} rad)")
        print(
            f"Pitch: {rpy_deg[1]:8.3f}° ({results['ee_to_cb_rpy'][1]:8.6f} rad)")
        print(
            f"Yaw:   {rpy_deg[2]:8.3f}° ({results['ee_to_cb_rpy'][2]:8.6f} rad)")
        print(f"X:     {results['ee_to_cb_pos'][0]:8.6f} m")
        print(f"Y:     {results['ee_to_cb_pos'][1]:8.6f} m")
        print(f"Z:     {results['ee_to_cb_pos'][2]:8.6f} m")

        print("\n" + "-"*40)
        print("CAMERA TO ROBOT BASE TRANSFORM")
        print("-"*40)
        rpy_deg = np.degrees(results['cam_to_base_rpy'])
        print(
            f"Roll:  {rpy_deg[0]:8.3f}° ({results['cam_to_base_rpy'][0]:8.6f} rad)")
        print(
            f"Pitch: {rpy_deg[1]:8.3f}° ({results['cam_to_base_rpy'][1]:8.6f} rad)")
        print(
            f"Yaw:   {rpy_deg[2]:8.3f}° ({results['cam_to_base_rpy'][2]:8.6f} rad)")
        print(f"X:     {results['cam_to_base_pos'][0]:8.6f} m")
        print(f"Y:     {results['cam_to_base_pos'][1]:8.6f} m")
        print(f"Z:     {results['cam_to_base_pos'][2]:8.6f} m")

        print("\n" + "-"*40)
        print("TRANSFORMATION MATRICES")
        print("-"*40)
        print("T_ee_to_chessboard:")
        print(results['T_ee_to_cb'])
        print("\nT_camera_to_base:")
        print(results['T_cam_to_base'])

        avg_error = np.mean(results['reprojection_error_per_image'])
        max_error = np.max(results['reprojection_error_per_image'])
        print(f"\nAverage RMS Error per Image: {avg_error:.3f} pixels")
        print(f"Maximum RMS Error per Image: {max_error:.3f} pixels")

        # Print validation results if available
        validation = self.validate_calibration(results)
        print("\n" + "-"*40)
        print("CALIBRATION VALIDATION")
        print("-"*40)
        print(f"Overall Quality: {validation['overall_quality'].upper()}")
        if validation['warnings']:
            print("Warnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")
        else:
            print("No validation warnings")

    def save_results(self, results: Dict, output_file: str):
        """Save calibration results to JSON file."""
        results_to_save = {
            'camera_serial': self.camera_serial,
            'chessboard_size': self.chessboard_size,
            'square_size': self.square_size,
            'T_ee_to_chessboard': results['T_ee_to_cb'].tolist(),
            'T_camera_to_base': results['T_cam_to_base'].tolist(),
            'ee_to_cb_rpy_degrees': np.degrees(results['ee_to_cb_rpy']).tolist(),
            'ee_to_cb_position': results['ee_to_cb_pos'].tolist(),
            'cam_to_base_rpy_degrees': np.degrees(results['cam_to_base_rpy']).tolist(),
            'cam_to_base_position': results['cam_to_base_pos'].tolist(),
            'final_reprojection_error': float(results['final_error']),
            'rms_error_per_image': results['reprojection_error_per_image'],
            'images_used': len(self.image_points),
            'sample_names': self.sample_names,
            'convergence_rate': results.get('convergence_rate', 0.0),
            'num_successful': results.get('num_successful', 1),
            'validation': self.validate_calibration(results)
        }

        with open(output_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)

        print(f"Results saved to '{output_file}'")

    def run_full_calibration(self, images_dir: str, camera_name: str = "r2",
                             output_dir: str = "calibration_results") -> Dict:
        """
        Run complete calibration pipeline.

        Args:
            images_dir: Directory containing calibration images
            camera_name: Subdirectory name for camera images
            output_dir: Directory to save results

        Returns:
            Calibration results dictionary
        """
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Detect chessboard corners
        print("="*60)
        print("STEP 1: DETECTING CHESSBOARD CORNERS")
        print("="*60)
        num_detected = self.detect_chessboard_corners(images_dir, camera_name)

        if num_detected < 5:
            raise ValueError(
                f"Only {num_detected} images with detected corners. Need at least 5.")

        # Step 2: Run multi-start calibration
        print("\n" + "="*60)
        print("STEP 2: RUNNING MULTI-START CALIBRATION")
        print("="*60)
        results = self.multi_start_calibrate(
            num_starts=30, use_default_start=True)

        # Step 3: Display and save results
        print("\n" + "="*60)
        print("STEP 3: RESULTS AND VALIDATION")
        print("="*60)
        self.print_results(results)

        # Step 4: Save results and visualizations
        print("\n" + "="*60)
        print("STEP 4: SAVING RESULTS")
        print("="*60)

        # Save JSON results
        results_file = os.path.join(
            output_dir, f"calibration_results_{self.camera_serial}.json")
        self.save_results(results, results_file)

        # Save visualizations
        viz_file = os.path.join(
            output_dir, f"calibration_visualization_{self.camera_serial}.png")
        self.visualize_results(results, save_path=viz_file)

        print("Calibration complete!")
        return results


def main():
    """Main calibration routine."""
    # Configuration
    ee_poses_file = "calibration_data/ee_poses.json"
    intrinsics_file = "intrinsics/all_cameras_intrinsics.yaml"
    images_dir = "calibration_data"
    camera_name = "r2"
    camera_serial = "943222071556"

    # Chessboard configuration
    CHESSBOARD_SIZE = (7, 6)
    square_size = 0.025

    try:
        # Initialize calibrator
        calibrator = ExtrinsicCalibrator(
            ee_poses_file=ee_poses_file,
            intrinsics_file=intrinsics_file,
            camera_serial=camera_serial,
            chessboard_size=CHESSBOARD_SIZE,
            square_size=square_size
        )

        # Run full calibration pipeline
        results = calibrator.run_full_calibration(
            images_dir=images_dir,
            camera_name=camera_name,
            output_dir="calibration_results"
        )

        # Additional validation check
        validation = calibrator.validate_calibration(results)
        if validation['overall_quality'] == 'poor':
            print("\n⚠️  WARNING: Calibration quality is poor. Consider:")
            print("   - Taking more calibration images")
            print("   - Ensuring better chessboard visibility")
            print("   - Checking camera intrinsics")
            print("   - Verifying robot pose accuracy")
        elif validation['overall_quality'] == 'good':
            print("\n✅ Calibration quality is good!")

    except Exception as e:
        print(f"Calibration failed: {e}")
        raise


if __name__ == "__main__":
    main()
