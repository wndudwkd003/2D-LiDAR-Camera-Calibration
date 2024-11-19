#######
# KJY #
#######

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan, PointCloud2
import cv2  
from threading import Thread, Lock
import time  
import numpy as np
from datetime import datetime
from aprilgrid import Detector
from CalibJ.utils.config_loader import load_config, load_json
from CalibJ.utils.clustering_vis import points_to_pointcloud2
from CalibJ.module.clustering_scan import save_execution_statistics, dbscan_clustering, display_clusters, world_to_pixel # , optics_clustering
from CalibJ.module.calibration_module import calibration_2dlidar_camera
from CalibJ.module.apriltag_detect import detect_apriltag
from CalibJ.evaluate.clustering_eval import evaluate_clustering, record_evaluation_result 
from CalibJ.module.tracking import calculate_cluster_centers, ClusterTracker

class CalibrationNode(Node):
    def __init__(self):
        super().__init__('calibration_node')
        self.config = load_config()
        self.get_logger().info(f"Loaded configuration: {self.config}")

        self.camera_params = load_json(self.config.cam_calib_result_json)

        # ROS 2 Subscribers and Publishers
        self.scan_sub = self.create_subscription(LaserScan, self.config.scan_topic, self.scan_callback, 10)
        self.cluster_pub = self.create_publisher(PointCloud2, self.config.cluster_topic, 10)

        # Data Management
        self.latest_scan = None
        self.latest_camera_frame = None
        self.scan_lock = Lock()
        self.running = True
        self.lidar_features = []

        # AprilTag Detector and Tracker
        self.detector = Detector("t36h11")
        self.apriltag_features = []
        self.cluster_tracker = ClusterTracker()

        # Mouse Interaction
        self.click_x = None
        self.click_y = None
        cv2.namedWindow("Camera and Clusters")
        cv2.setMouseCallback("Camera and Clusters", self.on_mouse_click)

        # Camera Initialization
        self.capture = cv2.VideoCapture(self.config.camera_number)
        if not self.capture.isOpened():
            self.get_logger().error("Failed to open the camera!")
            raise RuntimeError("Camera initialization failed")

        # ROS 2 Timers for main thread processing
        self.create_timer(0.1, self.camera_loop)  # 100ms interval
        self.create_timer(0.1, self.process_data_loop)  # 100ms interval
        self.create_timer(0.1, self.show_visualization)  # 100ms interval for GUI

        self.get_logger().info("All modules are ready.")

    def scan_callback(self, msg):
        """Callback for LaserScan topic."""
        with self.scan_lock:
            self.latest_scan = {'timestamp': time.time(), 'data': msg}

    def camera_loop(self):
        """Periodic camera frame capture."""
        ret, frame = self.capture.read()
        if ret:
            self.latest_camera_frame = {'timestamp': time.time(), 'frame': frame}
        else:
            self.get_logger().error("Failed to capture frame from the camera!")

    def process_data_loop(self):
        """Synchronize and process data."""
        with self.scan_lock:
            if self.latest_scan and self.latest_camera_frame:
                scan_time = self.latest_scan['timestamp']
                camera_time = self.latest_camera_frame['timestamp']

                if abs(scan_time - camera_time) < 0.3:  # Synchronization threshold
                    scan_data = self.latest_scan['data']
                    camera_frame = self.latest_camera_frame['frame']
                    self.process_synchronized_data(scan_data, camera_frame)

    def process_synchronized_data(self, scan_data, camera_frame):
        """Process synchronized scan and camera data."""
        labels, cluster_points, _ = dbscan_clustering(
            scan_data, epsilon=self.config.epsilon, min_samples=self.config.min_samples
        )

        # rviz 시각화용
        # cluster_msg = points_to_pointcloud2(labels, cluster_points, frame_id=self.config.cluster_frame)
        # self.cluster_pub.publish(cluster_msg)

        pix_labels, pix_cluster_points = world_to_pixel(labels, cluster_points, max_distance=self.config.max_distance,)

        # update cluster center positions
        self.cluster_tracker.update_clusters(pix_labels, pix_cluster_points)


        # 마우스 클릭으로 중심 좌표 추적 시작
        if self.click_x is not None and self.click_y is not None:
            selected_center = self.cluster_tracker.select_tracked_center(self.click_x, self.click_y, selection_radius=100)
            if selected_center:
                pass
                # self.get_logger().info(f"Tracking cluster center at: {selected_center}")
            self.click_x = None
            self.click_y = None

        # 칼만 필터 추적
        if self.cluster_tracker.is_tracking:
            predicted_position = self.cluster_tracker.track()
            if predicted_position is not None:
                pass
                # self.get_logger().info(f"Tracking predicted position: {predicted_position}")


        # Visualization preparation
        self.cluster_canvas = display_clusters(
            pix_labels,
            pix_cluster_points,
            canvas_size=800,
            tracked_center=self.cluster_tracker.tracked_center,  # 중심 좌표 전달
            only_clustering=False
        )

        self.camera_frame = camera_frame

    def show_visualization(self):
        """Main thread OpenCV GUI rendering."""
        if hasattr(self, 'cluster_canvas') and self.camera_frame is not None:
            try:
                # Resize and combine frames for display
                detect_frame = self.camera_frame

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Quit
                    self.destroy_node()
                    rclpy.shutdown()

                elif key == ord('a'):  # Detect AprilTag
                    self.get_logger().info("Key 'a' pressed: Detecting AprilTag")
                    _ = detect_apriltag(self.detector, detect_frame)

                elif key == ord('y'):  # Process both LiDAR and AprilTag coordinates
                    self.get_logger().info("Key 'y' pressed: Processing LiDAR and AprilTag coordinates")
                    result = self.process_combined_coordinates(detect_frame, divide_num=10)
                    if not result:
                        print("AprilTag의 좌표가 맞지 않습니다. 6개가 나와야함.")

                elif key == ord('c'):  # 개수 세기용
                    self.get_logger().info(f"Key 'c' pressed: lidar_features -> {len(set(self.lidar_features))}, apriltag_features -> {len(set(self.apriltag_features))}.")

                elif key == ord('z'): # 캘리브레이션 진행
                    lidar_features = list(set(self.lidar_features))
                    apriltag_features = list(set(self.apriltag_features))
                    print(f"lidar_features: {len(lidar_features)}, apriltag_features: {len(apriltag_features)}")

                    # 캘리브레이션 외부행렬 추정
                    success, rvec, tvec, extrinsic = calibration_2dlidar_camera(lidar_features, apriltag_features, self.camera_params.camera_matrix, self.camera_params.dist_coeffs)

                camera_frame_resized = cv2.resize(detect_frame, (self.cluster_canvas.shape[1], self.cluster_canvas.shape[0]))
                combined_canvas = np.hstack((camera_frame_resized, self.cluster_canvas))

                # Show combined canvas
                cv2.imshow("Camera and Clusters", combined_canvas)

            except Exception as e:
                self.get_logger().error(f"Failed to display combined image: {e}")

    def process_combined_coordinates(self, frame, divide_num=10):
        """
        LiDAR와 AprilTag 좌표를 처리하여 각각 직선을 추정하고 일정 간격으로 나눕니다.

        Args:
            frame (np.ndarray): 카메라 프레임.
            divide_num (int): 직선을 나눌 구간 수.
        """

        # 1. AprilTag 데이터 처리
        apriltag_positions = detect_apriltag(self.detector, frame)
        if len(apriltag_positions) == 6:
            apriltag_line = self.estimate_line(apriltag_positions)
            apriltag_divided_points = self.divide_line(apriltag_line, divide_num)
            self.get_logger().info(f"AprilTag line divided into {len(apriltag_divided_points)} points.")
            self.apriltag_features.extend(apriltag_divided_points)
        else:
            return False


        # 2. LiDAR 데이터 처리
        if self.cluster_tracker.tracked_center is None:
            self.get_logger().warn("No cluster is being tracked!")
        else:
            tracked_points = list(self.cluster_tracker.cluster_centers.values())
            if len(tracked_points) > 1:
                lidar_line = self.estimate_line(tracked_points)
                lidar_divided_points = self.divide_line(lidar_line, divide_num)
                self.get_logger().info(f"LiDAR line divided into {len(lidar_divided_points)} points.")
                self.lidar_features.extend(lidar_divided_points)

        return True
        


    def on_mouse_click(self, event, x, y, flags, param):
        """Handle mouse click events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 왼쪽 카메라 창 너비 계산
            camera_width = self.cluster_canvas.shape[1]

            if x < camera_width:
                # 카메라 창에서 클릭
                self.get_logger().info(f"Mouse clicked in Camera window at ({x}, {y})")
                self.click_x = x
                self.click_y = y
                # 필요한 경우 카메라 관련 추가 로직 작성 가능
            else:
                # LiDAR 창에서 클릭 (좌표 변환)
                lidar_x = x - camera_width
                self.get_logger().info(f"Mouse clicked in LiDAR window at ({lidar_x}, {y})")
                self.click_x = lidar_x
                self.click_y = y

    def destroy_node(self):
        """Cleanup on shutdown."""
        self.running = False
        self.capture.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    main()