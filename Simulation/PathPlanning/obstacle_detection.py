import numpy as np
import json
import torch
import torchvision.transforms as T
from torchvision.models.detection import ssd300_vgg16
import os, logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

class CameraObstacleDetection:
    def __init__(self, vehicle_mapping_path, confidence_threshold=0.5):
        logging.info("Initializing CameraObstacleDetection...")
        self.vehicle_mapping_path = vehicle_mapping_path
        self.camera_id = self._get_sensor_id("camera")
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = self._load_camera_model()

    def _load_camera_model(self):
        logging.info("Loading camera detection model...")
        model = ssd300_vgg16(pretrained=True).to(self.device)
        model.eval()
        logging.info("Camera detection model loaded successfully.")
        return model

    def _get_sensor_id(self, sensor_type):
        logging.info(f"Fetching sensor ID for {sensor_type}...")
        with open(self.vehicle_mapping_path, "r") as f:
            vehicle_mapping = json.load(f)
        sensors = vehicle_mapping.get("ego_veh", {}).get("sensors", [])
        sensor_id = sensors[2] if sensor_type == "camera" and len(sensors) > 2 else None
        logging.info(f"Sensor ID for {sensor_type}: {sensor_id}")
        return sensor_id

    def preprocess_image(self, image):
        logging.debug("Preprocessing image for model inference...")
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((300, 300)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0).to(self.device)

    def detect_obstacles(self, image):
        logging.info("Detecting obstacles using camera data...")
        preprocessed_image = self.preprocess_image(image)
        with torch.no_grad():
            detections = self.detector(preprocessed_image)[0]
        results = self._postprocess_detections(detections, image.shape[1], image.shape[0])
        logging.info(f"Camera detections: {len(results)} obstacles found.")
        return results

    def _postprocess_detections(self, detections, image_width, image_height):
        labels = detections['labels'].cpu().numpy()
        scores = detections['scores'].cpu().numpy()
        boxes = detections['boxes'].cpu().numpy()

        results = []
        for label, score, box in zip(labels, scores, boxes):
            if score >= self.confidence_threshold:
                x_min, y_min, x_max, y_max = box
                results.append({
                    "class": label,
                    "bbox": [
                        int(x_min * image_width / 300),
                        int(y_min * image_height / 300),
                        int(x_max * image_width / 300),
                        int(y_max * image_height / 300),
                    ],
                    "confidence": float(score),
                })
        logging.debug(f"Postprocessed detections: {results}")
        return results


class LidarObstacleDetection:
    def __init__(self, vehicle_mapping_path, confidence_threshold=0.5):
        logging.info("Initializing LidarObstacleDetection...")
        self.vehicle_mapping_path = vehicle_mapping_path
        self.lidar_id = self._get_sensor_id("lidar")
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = self._load_lidar_model()
        self.combined_lidar_data = None

    def _load_lidar_model(self):
        logging.info("Loading LiDAR detection model...")
        model = ssd300_vgg16(pretrained=True).to(self.device)
        model.eval()
        logging.info("LiDAR detection model loaded successfully.")
        return model

    def _get_sensor_id(self, sensor_type):
        logging.info(f"Fetching sensor ID for {sensor_type}...")
        with open(self.vehicle_mapping_path, "r") as f:
            vehicle_mapping = json.load(f)
        sensors = vehicle_mapping.get("ego_veh", {}).get("sensors", [])
        sensor_id = sensors[3] if sensor_type == "lidar" and len(sensors) > 3 else None
        logging.info(f"Sensor ID for {sensor_type}: {sensor_id}")
        return sensor_id
    
    def update_combined_lidar_data(self, combined_data):
        logging.info("Updating combined LiDAR data...")
        self.combined_lidar_data = combined_data

    def detect_obstacles(self, ego_lidar_data):
        logging.info("Detecting obstacles using LiDAR data...")
        lidar_data_to_use = self.combined_lidar_data if self.combined_lidar_data is not None else ego_lidar_data
        bev_image = self._preprocess_lidar_data(lidar_data_to_use)
        with torch.no_grad():
            detections = self.detector(bev_image)[0]
        results = self._postprocess_detections(detections, bev_image.shape[2:])
        logging.info(f"LiDAR detections: {len(results)} obstacles found.")
        return results

    def _preprocess_lidar_data(self, point_cloud, bev_image_size=(300, 300)):
        logging.debug("Converting LiDAR point cloud to BEV image...")
        x, y, intensity = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 3]
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        x_scaled = ((x - x_min) / (x_max - x_min) * (bev_image_size[0] - 1)).astype(int)
        y_scaled = ((y - y_min) / (y_max - y_min) * (bev_image_size[1] - 1)).astype(int)

        bev_image = np.zeros(bev_image_size, dtype=np.uint8)
        bev_image[y_scaled, x_scaled] = (intensity * 255).astype(np.uint8)
        bev_image = np.stack([bev_image] * 3, axis=-1)

        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((300, 300)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        logging.debug("BEV image preprocessing completed.")
        return transform(bev_image).unsqueeze(0).to(self.device)

    def _postprocess_detections(self, detections, bev_image_size):
        labels = detections['labels'].cpu().numpy()
        scores = detections['scores'].cpu().numpy()
        boxes = detections['boxes'].cpu().numpy()

        results = []
        for label, score, box in zip(labels, scores, boxes):
            if score >= self.confidence_threshold:
                x_min, y_min, x_max, y_max = box
                results.append({
                    "class": label,
                    "bbox": [
                        int(x_min * bev_image_size[0] / 300),
                        int(y_min * bev_image_size[1] / 300),
                        int(x_max * bev_image_size[0] / 300),
                        int(y_max * bev_image_size[1] / 300),
                    ],
                    "confidence": float(score),
                })
        logging.debug(f"Postprocessed LiDAR detections: {results}")
        return results


class DataFusion:
    def __init__(self, vehicle_mapping_path, iou_threshold=0.5, camera_confidence=0.5, lidar_confidence=0.5):
        logging.info("Initializing DataFusion...")
        self.camera_detector = CameraObstacleDetection(vehicle_mapping_path, camera_confidence)
        self.lidar_detector = LidarObstacleDetection(vehicle_mapping_path, lidar_confidence)
        self.iou_threshold = iou_threshold

    @staticmethod
    def compute_iou(box1, box2):
        x_min = max(box1[0], box2[0])
        y_min = max(box1[1], box2[1])
        x_max = min(box1[2], box2[2])
        y_max = min(box1[3], box2[3])

        intersection = max(0, x_max - x_min) * max(0, y_max - y_min)
        area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area_box1 + area_box2 - intersection
        iou = intersection / union if union > 0 else 0
        logging.debug(f"IoU computed: {iou}")
        return iou

    def fuse_data(self, camera_frame, lidar_point_cloud, save_folder="fused_results"):
        """
        Fuse camera and LiDAR detections into unified results and save to a JSON file.

        :param camera_frame: Camera frame as a numpy array.
        :param lidar_point_cloud: LiDAR point cloud as a numpy array.
        :param save_folder: Folder to save the JSON output.
        :return: List of fused detections.
        """
        logging.info("Fusing data from camera and LiDAR...")
        # Detect obstacles using camera and LiDAR
        camera_detections = self.camera_detector.detect_obstacles(camera_frame)
        lidar_detections = self.lidar_detector.detect_obstacles(lidar_point_cloud)

        # Match and fuse detections
        matched, unmatched_camera, unmatched_lidar = self._match_detections(camera_detections, lidar_detections)
        fused_results = [
            {
                "source": "fused",
                "class": pair["camera"]["class"],
                "confidence": max(
                    pair["camera"]["confidence"], pair["lidar"]["confidence"]
                ),
                "bbox": pair["lidar"]["bbox"],
            }
            for pair in matched
        ]
        fused_results.extend({"source": "camera", **det} for det in unmatched_camera)
        fused_results.extend({"source": "lidar", **det} for det in unmatched_lidar)

        # Save results to a JSON file
        self._save_results(fused_results, save_folder)
        logging.info(f"Data fusion completed with {len(fused_results)} total detections.")
        return fused_results

    def _match_detections(self, camera_detections, lidar_detections):
        """Match camera and LiDAR detections based on IoU."""
        logging.info("Matching detections between camera and LiDAR...")
        matched = []
        unmatched_camera = []
        unmatched_lidar = lidar_detections.copy()

        for cam_det in camera_detections:
            best_iou = 0
            best_lidar_det = None

            for lidar_det in unmatched_lidar:
                iou = self.compute_iou(cam_det['bbox'], lidar_det['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_lidar_det = lidar_det
                
                # Log the IoU score for the current camera-LiDAR pair
                logging.debug(f"Camera Detection: {cam_det['bbox']} vs LiDAR Detection: {lidar_det['bbox']} -> IoU: {iou}")

            # If a match is found, log and save the best IoU along with the detections
            if best_iou >= self.iou_threshold and best_lidar_det:
                matched.append({
                    "camera": cam_det, 
                    "lidar": best_lidar_det, 
                    "iou": best_iou  # Save the best IoU score for the matched pair
                })
                unmatched_lidar.remove(best_lidar_det)
            else:
                unmatched_camera.append(cam_det)
        logging.info(f"Matched detections: {len(matched)}. Unmatched camera: {len(unmatched_camera)}. Unmatched LiDAR: {len(unmatched_lidar)}.")
        return matched, unmatched_camera, unmatched_lidar

    def _save_results(self, fused_results, save_folder):
        """
        Save the fused detection results to a JSON file with a unique timestamp-based name.

        :param fused_results: Dictionary containing matched, unmatched, and sensor-specific results.
        :param save_folder: Folder where the JSON file will be saved.
        """
        logging.info("Saving fused detection results to a JSON file...")

        # Ensure the save folder exists
        os.makedirs(save_folder, exist_ok=True)

        # Generate a unique filename with the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(save_folder, f"fused_results_{timestamp}.json")

        # Helper function to convert non-serializable types
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)  # Convert numpy integers to Python int
            elif isinstance(obj, np.floating):
                return float(obj)  # Convert numpy floats to Python float
            elif isinstance(obj, np.ndarray):
                return obj.tolist()  # Convert numpy arrays to lists
            elif isinstance(obj, set):
                return list(obj)  # Convert sets to lists
            else:
                raise TypeError(f"Type {type(obj)} not serializable")

        try:
            # Save fused results to the JSON file with a custom converter
            with open(output_file, "w") as f:
                json.dump(fused_results, f, indent=4, default=convert_to_serializable)
            logging.info(f"Fused detection results saved to {output_file}")

        except TypeError as e:
            logging.error(f"Error saving fused results: {e}")
        except Exception as e:
            logging.error(f"Unexpected error while saving fused results: {e}")