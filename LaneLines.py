import cv2
import numpy as np

class LaneLines:
    """
    Lane and curve detection with polynomial fitting + lane departure warning.
    """
    def __init__(self):
        # Canny thresholds
        self.low_threshold = 50
        self.high_threshold = 150

        # Morphology to clean up edges
        self.kernel = np.ones((5, 5), np.uint8)

        # Lane departure threshold (pixels)
        self.departure_threshold = 80  

    def region_of_interest(self, img):
        """Apply a polygonal mask to focus on the road area."""
        height, width = img.shape[:2]
        mask = np.zeros_like(img)

        polygon = np.array([[
            (0, height),
            (width * 0.4, height * 0.6),
            (width * 0.6, height * 0.6),
            (width, height)
        ]], dtype=np.int32)

        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(img, mask)

    def preprocess(self, frame):
        """Convert to grayscale, blur, and edge detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.low_threshold, self.high_threshold)
        edges = cv2.dilate(edges, self.kernel, iterations=1)
        edges = cv2.erode(edges, self.kernel, iterations=1)
        return edges

    def fit_polyline(self, points, frame, color=(0, 255, 0)):
        """Fit a polynomial curve and draw it on the frame."""
        if len(points) < 10:
            return None, frame

        points = np.array(points)
        x = points[:, 0]
        y = points[:, 1]

        # Fit quadratic polynomial: x = f(y)
        poly = np.poly1d(np.polyfit(y, x, 2))

        y_new = np.linspace(min(y), max(y), num=50)
        x_new = poly(y_new).astype(int)
        pts = np.array(list(zip(x_new, y_new.astype(int))))

        # Draw curve
        for i in range(len(pts) - 1):
            cv2.line(frame, tuple(pts[i]), tuple(pts[i + 1]), color, 8)

        return poly, frame

    def separate_lines(self, edges):
        """Find left and right lane points from edges."""
        lines = cv2.HoughLinesP(edges, 2, np.pi / 180, 100, minLineLength=40, maxLineGap=150)
        left_points, right_points = [], []
        if lines is None:
            return left_points, right_points

        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if slope < 0:  # left lane
                left_points.append((x1, y1))
                left_points.append((x2, y2))
            else:  # right lane
                right_points.append((x1, y1))
                right_points.append((x2, y2))

        return left_points, right_points

    def lane_departure_warning(self, frame, left_poly, right_poly):
        """Check if car deviates from lane center."""
        height, width = frame.shape[:2]
        vehicle_center = width // 2
        bottom_y = height

        # Estimate left and right lane positions at the bottom of the frame
        left_x = int(left_poly(bottom_y)) if left_poly is not None else None
        right_x = int(right_poly(bottom_y)) if right_poly is not None else None

        if left_x and right_x:
            lane_center = (left_x + right_x) // 2
            deviation = vehicle_center - lane_center

            # Draw vehicle center & lane center
            cv2.line(frame, (vehicle_center, height), (vehicle_center, height - 50), (255, 0, 0), 3)
            cv2.line(frame, (lane_center, height), (lane_center, height - 50), (0, 255, 255), 3)

            # Check deviation
            if abs(deviation) > self.departure_threshold:
                cv2.putText(frame, "LANE DEPARTURE!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        return frame

    def forward(self, frame):
        """Main pipeline with LDWS."""
        edges = self.preprocess(frame)
        cropped = self.region_of_interest(edges)

        left_pts, right_pts = self.separate_lines(cropped)

        result = frame.copy()
        left_poly, result = self.fit_polyline(left_pts, result, color=(0, 255, 0))
        right_poly, result = self.fit_polyline(right_pts, result, color=(0, 255, 0))

        # Lane departure warning
        result = self.lane_departure_warning(result, left_poly, right_poly)

        return result


# 🔹 Optional YOLO object detection hook
# from ultralytics import YOLO
# yolo = YOLO("yolov8n.pt")
#
# def detect_objects(frame):
#     results = yolo(frame)
#     return results[0].plot()


if __name__ == "__main__":
    lane_detector = LaneLines()
    cap = cv2.VideoCapture("road_video.mp4")  # or 0 for webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        lanes = lane_detector.forward(frame)
        # lanes = detect_objects(lanes)  # optional YOLO integration

        cv2.imshow("Lane, Curve & LDWS", lanes)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
