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


from CalibJ.utils.config_loader import load_config
from CalibJ.utils.clustering_vis import points_to_pointcloud2
from CalibJ.module.clustering_scan import dbscan_clustering, display_clusters


class CalibrationNode(Node):
    def __init__(self):
        super().__init__('calibration_node')  # 노드 이름 설정

        self.config = load_config()
        self.get_logger().info(f"Loaded configuration: {self.config}")

        # 토픽 구독
        self.scan_sub = self.create_subscription(LaserScan, self.config.scan_topic, self.scan_callback, 10)


        # 토픽 발행
        self.cluster_pub = self.create_publisher(PointCloud2, self.config.cluster_topic, 10)

        # 스캔 데이터 및 카메라 데이터 관리
        self.latest_scan = None
        self.latest_camera_frame = None
        self.scan_lock = Lock()
        self.running = True

        # 카메라 초기화
        self.capture = cv2.VideoCapture(self.config.camera_number)
        if not self.capture.isOpened():
            self.get_logger().error("Failed to open the camera!")
            raise RuntimeError("Camera initialization failed")

        # 카메라 데이터 수집 스레드 시작
        self.camera_thread = Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()

        # 데이터 동기화 및 처리 스레드
        self.processing_thread = Thread(target=self.process_data_loop, daemon=True)
        self.processing_thread.start()

        self.get_logger().info("All modules are ready.")  # 시작 시 로그 출력

    def scan_callback(self, msg):
        """스캔 데이터 콜백 함수"""
        with self.scan_lock:
            self.latest_scan = {'timestamp': time.time(), 'data': msg}
            # self.get_logger().info(f"Scan received: {len(msg.ranges)} ranges")

    def camera_loop(self):
        """카메라 데이터를 주기적으로 가져오는 루프"""
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                self.get_logger().error("Failed to capture frame from the camera!")
                break

            # 카메라 데이터 저장
            self.latest_camera_frame = {'timestamp': time.time(), 'frame': frame}

    def process_data_loop(self):
        """스캔 데이터와 카메라 데이터 동기화 및 처리"""
        while self.running:
            with self.scan_lock:
                if self.latest_scan and self.latest_camera_frame:
                    scan_time = self.latest_scan['timestamp']
                    camera_time = self.latest_camera_frame['timestamp']

                    # 시간 동기화: 300ms 이내인 경우 처리
                    if abs(scan_time - camera_time) < 0.3:
                        scan_data = self.latest_scan['data']
                        camera_frame = self.latest_camera_frame['frame']

                        # 동기화된 데이터 처리
                        self.process_synchronized_data(scan_data, camera_frame)

            time.sleep(0.05)  # 처리 루프 주기

    def process_synchronized_data(self, scan_data, camera_frame):
        """동기화된 데이터를 처리"""
        self.get_logger().info("Processing synchronized data...")

        # 스캔 데이터 클러스터링
        labels, cluster_points = dbscan_clustering(
            scan_data, epsilon=self.config.epsilon, min_samples=self.config.min_samples
        )

        # 클러스터링된 데이터를 PointCloud2 메시지로 변환 및 퍼블리시
        cluster_msg = points_to_pointcloud2(labels, cluster_points, frame_id=self.config.cluster_frame)
        self.cluster_pub.publish(cluster_msg)

        # 클러스터링 결과를 시각화 이미지로 생성
        cluster_canvas = display_clusters(labels, cluster_points, max_distance=self.config.max_distance)

        # 두 이미지를 병합 (카메라 프레임 왼쪽, 클러스터 결과 오른쪽)
        try:
            # 카메라 프레임 크기와 클러스터 캔버스 크기 맞추기
            camera_frame_resized = cv2.resize(camera_frame, (cluster_canvas.shape[1], cluster_canvas.shape[0]))

            # 두 이미지를 수평으로 병합
            combined_canvas = np.hstack((camera_frame_resized, cluster_canvas))

            # OpenCV 윈도우로 표시
            cv2.imshow('Camera and Clusters', combined_canvas)
            cv2.waitKey(1)  # 1ms 대기
        except Exception as e:
            self.get_logger().error(f"Failed to display combined image: {e}")

        # 가장 가까운 장애물 거리 출력
        closest_range = min(scan_data.ranges)
        self.get_logger().info(f"Closest object: {closest_range} meters")


    def destroy_node(self):
        """노드 종료 시 호출"""
        self.running = False
        self.camera_thread.join()  # 스레드 종료 대기
        self.processing_thread.join()
        self.capture.release()  # 카메라 자원 해제
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)  # ROS 2 초기화
    node = CalibrationNode()  # 노드 객체 생성

    try:
        rclpy.spin(node)  # 노드 실행
    except KeyboardInterrupt:
        pass  # Ctrl+C로 종료 시 예외 처리
    finally:
        node.destroy_node()  # 노드 정리
        rclpy.shutdown()  # ROS 2 종료


if __name__ == '__main__':
    main()