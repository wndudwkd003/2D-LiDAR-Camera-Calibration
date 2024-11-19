#######
# KJY #
#######

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import json
from sensor_msgs.msg import LaserScan
from CalibJ.utils.config_loader import load_config, load_json, load_calibration_ex_json

class VisualizationNode(Node):
    def __init__(self):
        super().__init__('viscam_node')
        self.get_logger().info("Visualization Node Initialized.")

        self.config = load_config()
        self.get_logger().info(f"Loaded configuration: {self.config}")

        self.camera_params = load_json(self.config.cam_calib_result_json)
        self.ex_params = load_calibration_ex_json(self.config.ex_calib_result_json)

        # Parse calibration data
        self.camera_matrix = self.camera_params.camera_matrix
        self.dist_coeffs = self.camera_params.dist_coeffs
        self.extrinsic_matrix = self.ex_params.extrinsic_matrix

        # Camera initialization
        self.capture = cv2.VideoCapture(self.config.camera_number)  # Default camera device
        if not self.capture.isOpened():
            self.get_logger().error("Failed to open camera. Exiting...")
            rclpy.shutdown()

        # LiDAR data
        self.lidar_points = []
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Timer for processing and visualization
        self.timer = self.create_timer(0.1, self.visualize)


    def scan_callback(self, msg):
        """Callback for processing LiDAR scan data."""
        self.lidar_points = self.convert_scan_to_points(msg)

    def convert_scan_to_points(self, scan):
        """Convert LaserScan data to 2D points."""
        angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
        points = [
            (r * np.cos(angle), r * np.sin(angle))
            for r, angle in zip(scan.ranges, angles) if not np.isinf(r)
        ]
        return np.array(points, dtype=np.float32)

    def project_lidar_to_image(self, lidar_points):
        """Project LiDAR points to the camera frame."""
        if lidar_points.shape[1] == 2:
            lidar_points = np.hstack((lidar_points, np.zeros((lidar_points.shape[0], 1))))  # Z = 0 추가

        # Homogeneous transformation
        lidar_homogeneous = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))
        camera_coords = lidar_homogeneous @ self.extrinsic_matrix.T  # Apply extrinsic matrix

        # Normalize to image plane
        image_points = camera_coords[:, :3] / camera_coords[:, 2:3]
        pixel_coords = (image_points @ self.camera_matrix.T)[:, :2]

        # Undistort points
        pixel_coords = cv2.undistortPoints(
            pixel_coords.reshape(-1, 1, 2), self.camera_matrix, self.dist_coeffs
        ).reshape(-1, 2)
        return pixel_coords

    def visualize(self):
        """Capture camera frame and visualize LiDAR data."""
        ret, frame = self.capture.read()
        if not ret:
            self.get_logger().error("Failed to capture frame from camera.")
            return

        # Project LiDAR points onto the camera frame
        if len(self.lidar_points) > 0:
            projected_points = self.project_lidar_to_image(self.lidar_points)
            for point in projected_points:
                x, y = int(point[0]), int(point[1])
                if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)  # Draw green points

        # Display the frame
        cv2.imshow("Camera with LiDAR", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key
            self.destroy_node()
            rclpy.shutdown()

    def destroy_node(self):
        """Cleanup resources on shutdown."""
        self.capture.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
