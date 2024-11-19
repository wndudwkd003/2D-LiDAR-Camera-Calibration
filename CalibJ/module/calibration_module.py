import numpy as np
import cv2

def calibration_2dlidar_camera(lidar_features, apriltag_features, camera_matrix, dist_coeffs):
    """
    Perform calibration between 2D LiDAR and camera using corresponding points.

    Args:
        lidar_features (list of tuples): LiDAR feature points in world coordinates (2D).
        apriltag_features (list of tuples): AprilTag feature points in image coordinates.
        camera_matrix (np.ndarray): Camera intrinsic matrix (3x3).
        dist_coeffs (np.ndarray): Camera distortion coefficients (1D array).

    Returns:
        tuple:
            success (bool): Whether the calibration succeeded.
            rvec (np.ndarray): Rotation vector (3x1).
            tvec (np.ndarray): Translation vector (3x1).
            extrinsic (np.ndarray): Extrinsic matrix (4x4).
    """
    # Ensure the inputs are numpy arrays
    lidar_features = np.array(lidar_features, dtype=np.float32)
    apriltag_features = np.array(apriltag_features, dtype=np.float32)

    # Convert 2D LiDAR points to 3D by setting Z = 0
    lidar_features_3d = np.hstack((lidar_features, np.zeros((lidar_features.shape[0], 1))))  # Add Z = 0

    # Check if the number of points matches
    if lidar_features_3d.shape[0] != apriltag_features.shape[0]:
        raise ValueError("The number of LiDAR features and AprilTag features must be the same.")

    # SolvePnP to estimate the rotation and translation vectors
    success, rvec, tvec = cv2.solvePnP(
        lidar_features_3d,  # 3D world points (LiDAR)
        apriltag_features,  # 2D image points (AprilTag)
        camera_matrix,  # Camera intrinsic matrix
        dist_coeffs,  # Camera distortion coefficients
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return False, None, None, None

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # Create the extrinsic matrix (4x4)
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation_matrix
    extrinsic[:3, 3] = tvec.flatten()

    return success, rvec, tvec, extrinsic

