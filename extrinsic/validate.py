#!/usr/bin/env python3
"""
Hand-Eye Calibration Validation Script
Validates calibration results and provides diagnostic plots and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os
from scipy.spatial.transform import Rotation as R
from typing import Dict, List, Tuple
import seaborn as sns

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Data paths
DATA_DIR = "calibration_data"
CALIBRATION_RESULTS_FILE = "calibration_results.json"
T_HAND_EYE_FILE = "T_hand_eye.npy"
T_BASE_EXTERNAL_FILE = "T_base_external.npy"

# Plot settings
FIGURE_SIZE = (12, 8)
DPI = 100
SAVE_PLOTS = True
PLOT_DIR = "validation_plots"

# Analysis thresholds
GOOD_ROTATION_ERROR_DEG = 5.0
GOOD_TRANSLATION_ERROR_M = 0.02
ACCEPTABLE_ROTATION_ERROR_DEG = 15.0
ACCEPTABLE_TRANSLATION_ERROR_M = 0.05

# =============================================================================
# VALIDATION CLASS
# =============================================================================


class CalibrationValidator:
    """Validates hand-eye calibration results"""

    def __init__(self):
        self.results = None
        self.T_hand_eye = None
        self.T_base_external = None
        self.samples_data = None

    def load_data(self):
        """Load calibration results and transformation matrices"""
        print("Loading calibration data...")

        # Load calibration results
        with open(CALIBRATION_RESULTS_FILE, 'r') as f:
            self.results = json.load(f)

        # Load transformation matrices
        self.T_hand_eye = np.load(T_HAND_EYE_FILE)
        self.T_base_external = np.load(T_BASE_EXTERNAL_FILE)

        # Load end-effector poses
        ee_poses_file = os.path.join(DATA_DIR, "ee_poses.json")
        with open(ee_poses_file, 'r') as f:
            self.samples_data = json.load(f)

        print(
            f"Loaded {len(self.results['evaluation']['individual_errors'])} samples")

    def create_plots_directory(self):
        """Create directory for saving plots"""
        if SAVE_PLOTS:
            os.makedirs(PLOT_DIR, exist_ok=True)

    def plot_error_distribution(self):
        """Plot error distribution histograms"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE)

        errors = self.results['evaluation']['individual_errors']
        rotation_errors = [e['rotation_error_deg'] for e in errors]
        translation_errors = [e['translation_error_m'] for e in errors]

        # Rotation errors
        ax1.hist(rotation_errors, bins=10, alpha=0.7,
                 color='skyblue', edgecolor='black')
        ax1.axvline(GOOD_ROTATION_ERROR_DEG, color='green', linestyle='--',
                    label=f'Good: {GOOD_ROTATION_ERROR_DEG}°')
        ax1.axvline(ACCEPTABLE_ROTATION_ERROR_DEG, color='orange', linestyle='--',
                    label=f'Acceptable: {ACCEPTABLE_ROTATION_ERROR_DEG}°')
        ax1.axvline(np.mean(rotation_errors), color='red', linestyle='-',
                    label=f'Mean: {np.mean(rotation_errors):.1f}°')
        ax1.set_xlabel('Rotation Error (degrees)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Rotation Error Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Translation errors
        ax2.hist(translation_errors, bins=10, alpha=0.7,
                 color='lightcoral', edgecolor='black')
        ax2.axvline(GOOD_TRANSLATION_ERROR_M, color='green', linestyle='--',
                    label=f'Good: {GOOD_TRANSLATION_ERROR_M}m')
        ax2.axvline(ACCEPTABLE_TRANSLATION_ERROR_M, color='orange', linestyle='--',
                    label=f'Acceptable: {ACCEPTABLE_TRANSLATION_ERROR_M}m')
        ax2.axvline(np.mean(translation_errors), color='red', linestyle='-',
                    label=f'Mean: {np.mean(translation_errors):.3f}m')
        ax2.set_xlabel('Translation Error (meters)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Translation Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if SAVE_PLOTS:
            plt.savefig(os.path.join(
                PLOT_DIR, 'error_distribution.png'), dpi=DPI)
        plt.show()

    def plot_error_per_sample(self):
        """Plot errors per sample to identify problematic samples"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGURE_SIZE)

        errors = self.results['evaluation']['individual_errors']
        sample_ids = [e['sample_id'] for e in errors]
        rotation_errors = [e['rotation_error_deg'] for e in errors]
        translation_errors = [e['translation_error_m'] for e in errors]

        # Rotation errors per sample
        bars1 = ax1.bar(range(len(sample_ids)), rotation_errors,
                        alpha=0.7, color='skyblue')
        ax1.axhline(GOOD_ROTATION_ERROR_DEG, color='green',
                    linestyle='--', label='Good')
        ax1.axhline(ACCEPTABLE_ROTATION_ERROR_DEG, color='orange',
                    linestyle='--', label='Acceptable')
        ax1.set_ylabel('Rotation Error (degrees)')
        ax1.set_title('Rotation Error per Sample')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Highlight problematic samples
        for i, (bar, error) in enumerate(zip(bars1, rotation_errors)):
            if error > ACCEPTABLE_ROTATION_ERROR_DEG:
                bar.set_color('red')
                bar.set_alpha(0.8)

        # Translation errors per sample
        bars2 = ax2.bar(range(len(sample_ids)),
                        translation_errors, alpha=0.7, color='lightcoral')
        ax2.axhline(GOOD_TRANSLATION_ERROR_M, color='green',
                    linestyle='--', label='Good')
        ax2.axhline(ACCEPTABLE_TRANSLATION_ERROR_M, color='orange',
                    linestyle='--', label='Acceptable')
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Translation Error (meters)')
        ax2.set_title('Translation Error per Sample')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Highlight problematic samples
        for i, (bar, error) in enumerate(zip(bars2, translation_errors)):
            if error > ACCEPTABLE_TRANSLATION_ERROR_M:
                bar.set_color('red')
                bar.set_alpha(0.8)

        plt.tight_layout()
        if SAVE_PLOTS:
            plt.savefig(os.path.join(
                PLOT_DIR, 'error_per_sample.png'), dpi=DPI)
        plt.show()

    def plot_3d_poses(self):
        """Plot 3D visualization of robot poses and camera positions"""
        fig = plt.figure(figsize=(15, 10))

        # Extract end-effector positions
        ee_positions = []
        for sample_id, pose_data in self.samples_data.items():
            ee_positions.append(pose_data["position"])
        ee_positions = np.array(ee_positions)

        # Calculate camera positions
        hand_cam_positions = []
        for pos, quat in zip(ee_positions, [self.samples_data[sid]["orientation"] for sid in self.samples_data]):
            T_base_ee = self.ee_pose_to_matrix(pos, quat)
            T_base_hand_cam = T_base_ee @ self.T_hand_eye
            hand_cam_positions.append(T_base_hand_cam[:3, 3])
        hand_cam_positions = np.array(hand_cam_positions)

        # External camera position
        ext_cam_pos = self.T_base_external[:3, 3]

        # Create 3D plot
        ax = fig.add_subplot(111, projection='3d')

        # Plot end-effector positions
        ax.scatter(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2],
                   c='blue', s=50, alpha=0.7, label='End-Effector Poses')

        # Plot hand camera positions
        ax.scatter(hand_cam_positions[:, 0], hand_cam_positions[:, 1], hand_cam_positions[:, 2],
                   c='red', s=50, alpha=0.7, label='Hand Camera Poses')

        # Plot external camera position
        ax.scatter(ext_cam_pos[0], ext_cam_pos[1], ext_cam_pos[2],
                   c='green', s=200, marker='^', label='External Camera')

        # Connect EE to hand camera for each sample
        for i in range(len(ee_positions)):
            ax.plot([ee_positions[i, 0], hand_cam_positions[i, 0]],
                    [ee_positions[i, 1], hand_cam_positions[i, 1]],
                    [ee_positions[i, 2], hand_cam_positions[i, 2]],
                    'k-', alpha=0.3, linewidth=0.5)

        # Add coordinate frame at origin
        ax.plot([0, 0.1], [0, 0], [0, 0], 'r-', linewidth=3, label='X-axis')
        ax.plot([0, 0], [0, 0.1], [0, 0], 'g-', linewidth=3, label='Y-axis')
        ax.plot([0, 0], [0, 0], [0, 0.1], 'b-', linewidth=3, label='Z-axis')

        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title('3D Visualization of Calibration Setup')
        ax.legend()

        # Equal aspect ratio
        max_range = np.array([ee_positions.max()-ee_positions.min(),
                             hand_cam_positions.max()-hand_cam_positions.min()]).max() / 2.0
        mid_x = (ee_positions[:, 0].max() + ee_positions[:, 0].min()) * 0.5
        mid_y = (ee_positions[:, 1].max() + ee_positions[:, 1].min()) * 0.5
        mid_z = (ee_positions[:, 2].max() + ee_positions[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.tight_layout()
        if SAVE_PLOTS:
            plt.savefig(os.path.join(PLOT_DIR, '3d_poses.png'), dpi=DPI)
        plt.show()

    def plot_transformation_analysis(self):
        """Analyze transformation matrices"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Hand-eye transformation
        R_he = self.T_hand_eye[:3, :3]
        t_he = self.T_hand_eye[:3, 3]

        # Base-external transformation
        R_be = self.T_base_external[:3, :3]
        t_be = self.T_base_external[:3, 3]

        # Hand-eye rotation matrix heatmap
        im1 = ax1.imshow(R_he, cmap='RdBu', vmin=-1, vmax=1)
        ax1.set_title('Hand-Eye Rotation Matrix')
        ax1.set_xlabel('Column')
        ax1.set_ylabel('Row')
        for i in range(3):
            for j in range(3):
                ax1.text(j, i, f'{R_he[i,j]:.3f}', ha='center', va='center')
        plt.colorbar(im1, ax=ax1)

        # Base-external rotation matrix heatmap
        im2 = ax2.imshow(R_be, cmap='RdBu', vmin=-1, vmax=1)
        ax2.set_title('Base-External Rotation Matrix')
        ax2.set_xlabel('Column')
        ax2.set_ylabel('Row')
        for i in range(3):
            for j in range(3):
                ax2.text(j, i, f'{R_be[i,j]:.3f}', ha='center', va='center')
        plt.colorbar(im2, ax=ax2)

        # Translation vectors
        ax3.bar(['X', 'Y', 'Z'], t_he, alpha=0.7, color='skyblue')
        ax3.set_title('Hand-Eye Translation Vector')
        ax3.set_ylabel('Translation (meters)')
        ax3.grid(True, alpha=0.3)
        for i, v in enumerate(t_he):
            ax3.text(i, v + 0.01*np.sign(v),
                     f'{v:.3f}', ha='center', va='bottom')

        ax4.bar(['X', 'Y', 'Z'], t_be, alpha=0.7, color='lightcoral')
        ax4.set_title('Base-External Translation Vector')
        ax4.set_ylabel('Translation (meters)')
        ax4.grid(True, alpha=0.3)
        for i, v in enumerate(t_be):
            ax4.text(i, v + 0.01*np.sign(v),
                     f'{v:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        if SAVE_PLOTS:
            plt.savefig(os.path.join(
                PLOT_DIR, 'transformation_analysis.png'), dpi=DPI)
        plt.show()

    def ee_pose_to_matrix(self, position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        """Convert end-effector pose to transformation matrix"""
        T = np.eye(4)
        T[:3, :3] = R.from_quat(quaternion).as_matrix()
        T[:3, 3] = position
        return T

    def analyze_calibration_quality(self):
        """Provide detailed analysis of calibration quality"""
        print("\n" + "="*60)
        print("CALIBRATION QUALITY ANALYSIS")
        print("="*60)

        eval_data = self.results['evaluation']

        # Overall assessment
        mean_rot_error = eval_data['mean_rotation_error_deg']
        mean_trans_error = eval_data['mean_translation_error_m']

        print(f"\nOverall Assessment:")
        if mean_rot_error < GOOD_ROTATION_ERROR_DEG and mean_trans_error < GOOD_TRANSLATION_ERROR_M:
            print("✓ EXCELLENT calibration quality")
        elif mean_rot_error < ACCEPTABLE_ROTATION_ERROR_DEG and mean_trans_error < ACCEPTABLE_TRANSLATION_ERROR_M:
            print("⚠ ACCEPTABLE calibration quality")
        else:
            print("✗ POOR calibration quality - needs improvement")

        # Detailed statistics
        print(f"\nDetailed Statistics:")
        print(
            f"Rotation Error: {mean_rot_error:.1f}° ± {eval_data['std_rotation_error_deg']:.1f}° (max: {eval_data['max_rotation_error_deg']:.1f}°)")
        print(
            f"Translation Error: {mean_trans_error:.3f}m ± {eval_data['std_translation_error_m']:.3f}m (max: {eval_data['max_translation_error_m']:.3f}m)")

        # Problematic samples
        errors = eval_data['individual_errors']
        bad_samples = [e for e in errors if e['rotation_error_deg'] > ACCEPTABLE_ROTATION_ERROR_DEG or
                       e['translation_error_m'] > ACCEPTABLE_TRANSLATION_ERROR_M]

        if bad_samples:
            print(f"\nProblematic Samples ({len(bad_samples)}):")
            for sample in bad_samples:
                print(
                    f"- {sample['sample_id']}: {sample['rotation_error_deg']:.1f}° rotation, {sample['translation_error_m']:.3f}m translation")

        # Recommendations
        print(f"\nRecommendations:")
        if mean_rot_error > ACCEPTABLE_ROTATION_ERROR_DEG:
            print("- High rotation errors: Check camera intrinsic calibration")
            print("- Consider more diverse robot orientations during data collection")

        if mean_trans_error > ACCEPTABLE_TRANSLATION_ERROR_M:
            print("- High translation errors: Check measurement accuracy of chessboard")
            print("- Ensure stable camera mounting and good lighting conditions")

        if len(bad_samples) > len(errors) * 0.3:
            print("- Many problematic samples: Consider re-collecting data")
            print("- Check for consistent chessboard detection across all images")

        print(
            f"\nCalibration Success Rate: {(len(errors) - len(bad_samples))/len(errors)*100:.1f}%")

    def run_validation(self):
        """Run complete validation suite"""
        print("="*60)
        print("HAND-EYE CALIBRATION VALIDATION")
        print("="*60)

        self.load_data()
        self.create_plots_directory()

        # Generate all plots
        print("\nGenerating validation plots...")
        self.plot_error_distribution()
        self.plot_error_per_sample()
        self.plot_3d_poses()
        self.plot_transformation_analysis()

        # Analyze calibration quality
        self.analyze_calibration_quality()

        if SAVE_PLOTS:
            print(f"\nPlots saved to: {PLOT_DIR}/")

# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """Main validation pipeline"""
    validator = CalibrationValidator()
    validator.run_validation()


if __name__ == "__main__":
    main()
