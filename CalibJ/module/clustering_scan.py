import numpy as np
from sklearn.cluster import DBSCAN
import cv2


def polar_to_cartesian(scan_data):
    """
    LaserScan 메시지 데이터를 데카르트 좌표계로 변환하고, 90도 왼쪽으로 회전.

    Args:
        scan_data (LaserScan): ROS2 LaserScan 메시지.

    Returns:
        np.ndarray: 변환된 (x, y) 좌표 배열, 크기 (N, 2)
    """
    num_ranges = len(scan_data.ranges)
    angles = np.arange(scan_data.angle_min, scan_data.angle_max, scan_data.angle_increment)
    angles = angles[:num_ranges]  # ranges 길이에 맞춰 각도 배열 조정

    ranges = np.array(scan_data.ranges)

    # 유효하지 않은 거리 값 필터링 (range_min과 range_max 기준)
    valid_indices = np.isfinite(ranges) & (ranges >= scan_data.range_min) & (ranges <= scan_data.range_max)
    ranges = ranges[valid_indices]
    angles = angles[valid_indices]

    # 극좌표 → 데카르트 좌표 변환
    x_coords = ranges * np.cos(angles)
    y_coords = ranges * np.sin(angles)

    points = np.stack((x_coords, y_coords), axis=1)  # (N, 2) 형태

    # 90도 왼쪽 회전 변환 적용
    rotation_matrix = np.array([[0, -1], [1, 0]])  # 90도 회전 행렬
    rotated_points = points @ rotation_matrix.T  # 행렬 곱으로 회전 적용

    return rotated_points


def dbscan_clustering(scan_data, epsilon=12, min_samples=5):
    """
    DBSCAN을 사용하여 스캔 데이터를 클러스터링.

    Args:
        scan_data (LaserScan): ROS2 LaserScan 메시지.
        epsilon (float): DBSCAN의 epsilon (근접 거리 허용 범위).
        min_samples (int): DBSCAN의 최소 샘플 개수.

    Returns:
        labels (np.ndarray): 유효 데이터의 클러스터 레이블 (-1은 노이즈).
        cluster_points (np.ndarray): 유효 데이터의 클러스터링된 좌표.
    """
    # LaserScan 데이터를 데카르트 좌표로 변환
    points = polar_to_cartesian(scan_data)

    # DBSCAN 클러스터링 적용
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    labels = dbscan.fit_predict(points)

    return labels, points

def display_clusters(labels, cluster_points, max_distance=5, base_canvas_size=800, padding=50):
    """
    클러스터링된 점들을 검은 바탕 이미지에 그린 후 반환하며,
    차를 중앙에 배치하고, 12시 방향을 차의 앞 방향으로 설정.
    캔버스의 최대 거리를 max_distance로 맞춥니다.
    클러스터 중앙점과 라벨 번호를 표시.

    Args:
        labels (np.ndarray): 클러스터 라벨.
        cluster_points (np.ndarray): 클러스터링된 점들의 좌표.
        max_distance (float): 캔버스의 최대 거리 (데이터 범위를 이 거리로 맞춤).
        base_canvas_size (int): 기본 캔버스 크기 (기본값: 800).
        padding (int): 캔버스 여백 (픽셀 단위).

    Returns:
        np.ndarray: 시각화를 위한 캔버스 이미지.
    """
    canvas_size = base_canvas_size
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    if len(cluster_points) > 0:
        # 캔버스에 데이터 비율 맞추기
        scale = (canvas_size - 2 * padding) / (2 * max_distance)  # 2 * max_distance: 양쪽 최대 범위 포함
        canvas_center = np.array([canvas_size / 2, canvas_size / 2])  # 캔버스 중심

        # 좌표 변환: 스케일링, y축 반전, 캔버스 중앙 이동
        normalized_points = cluster_points * scale
        normalized_points[:, 1] *= -1
        normalized_points += canvas_center

        # 클러스터별 색상 생성
        unique_labels = np.unique(labels)  # 모든 라벨 포함
        colors = generate_colors(len(unique_labels))

        for label in unique_labels:
            # 해당 라벨의 점들만 필터링
            label_indices = (labels == label)
            cluster_points_label = normalized_points[label_indices]

            # 라벨이 노이즈(-1)일 경우 색상 설정
            color = (128, 128, 128) if label == -1 else colors[label]

            # 클러스터 점들 그리기
            for point in cluster_points_label:
                x, y = int(point[0]), int(point[1])
                if 0 <= x < canvas_size and 0 <= y < canvas_size:  # 캔버스 범위 내 점만 그리기
                    cv2.circle(canvas, (x, y), 2, color, -1)

            if label != -1:  # 노이즈가 아닌 경우에만 중심점과 라벨 표시
                # 클러스터 중심 계산
                cluster_center = np.mean(cluster_points_label, axis=0)

                # 중심점 표시
                center_x, center_y = int(cluster_center[0]), int(cluster_center[1])
                cv2.circle(canvas, (center_x, center_y), 4, (0, 255, 0), -1)  # 초록색 원

                # 라벨 번호 표시
                cv2.putText(
                    canvas,
                    str(label),
                    (center_x, center_y - 10),  # 중심점 바로 위에 텍스트 배치
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,  # 텍스트 크기
                    (0, 255, 0),  # 초록색
                    2,
                    cv2.LINE_AA,
                )

    return canvas



def generate_colors(num_colors):
    """
    고유 라벨의 개수만큼 색상을 생성합니다.

    Args:
        num_colors (int): 생성할 색상의 개수.

    Returns:
        dict: {라벨: (B, G, R)} 형식의 색상 맵.
    """
    np.random.seed(0)
    colors = np.random.randint(0, 255, size=(num_colors, 3), dtype=np.uint8)
    return {i: tuple(map(int, colors[i])) for i in range(num_colors)}

