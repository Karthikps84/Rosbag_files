import os
import cv2
import torch
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage
from deep_sort_realtime.deepsort_tracker import DeepSort
from cv_bridge import CvBridge
from plyer import notification
import pyrealsense2 as rs2
import argparse
from mmdet.apis import init_detector, inference_detector

class DetectionProcessor:
    # Initialize the DetectionProcessor class with the image bag file path and output directory
    def __init__(self, image_bag_file_path, output_dir, camera_index):
        self.image_bag_file_path = image_bag_file_path  # Path to the ROS bag file containing images
        self.output_dir = output_dir  # Directory to save the output images
        self.image_count = 0  # Counter for the number of images processed
        self.bridge = CvBridge()  # Bridge to convert ROS messages to OpenCV images
        self.camera_index = camera_index

        # Create a directory to save images
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Initialize the mmdetection model
        self.model_cfg = 'rtmdet/rtmdet-ins_m_8xb32-300e_coco.py'  # Configuration file for the model
        self.checkpoint = 'rtmdet-ins_m_8xb32-300e_coco_20221123_001039-6eba602e.pth'
        self.model = init_detector(self.model_cfg, self.checkpoint, device='cuda:0') 

        self.class_names = ["obstacle"]  # List of class names
        # Create a list of random colors to represent each class
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3))
        
        # Initialize the DeepSort tracker
        self.tracker = DeepSort(max_age=10, n_init=3)
        
    # Function to get the camera topics based on the provided camera index
    def get_camera_topics(self, camera_index):
        image_topics = ['/cam_right/color/image_raw/compressed',
                        '/cam_left/color/image_raw/compressed',
                        '/cam_center/color/image_raw/compressed',
                        '/cam_hand/color/image_raw/compressed']

        depth_topics = ['/cam_right/depth/image_rect_raw',
                        '/cam_left/depth/image_rect_raw',
                        '/cam_center/depth/image_rect_raw',
                        '/cam_hand/depth/image_rect_raw']

        camera_topics = ['/cam_right/depth/camera_info',
                         '/cam_left/depth/camera_info',
                         '/cam_center/depth/camera_info',
                         '/cam_hand/depth/camera_info']

        color_topic = image_topics[camera_index]
        depth_topic = depth_topics[camera_index]
        cam_topic = camera_topics[camera_index]

        return color_topic, depth_topic, cam_topic

    # Function to process images from the ROS bag file
    def process_images(self):
        with AnyReader([Path(self.image_bag_file_path)]) as reader:
            color_topic, depth_topic, cam_topic = self.get_camera_topics(self.camera_index) 
            color_connections = [x for x in reader.connections if x.topic == color_topic]
            depth_connections = [x for x in reader.connections if x.topic == depth_topic]
            
            color_connection_images = reader.messages(connections=color_connections)
            depth_connection_images = reader.messages(connections=depth_connections)
            
        
            for i, (color_image, depth_image) in enumerate(zip(color_connection_images, depth_connection_images)):

                img_msg = reader.deserialize(color_image[2], color_image[0].msgtype)
                color_image = message_to_cvimage(img_msg, 'bgr8')

                depth_msg = reader.deserialize(depth_image[2], depth_image[0].msgtype)
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, depth_msg.encoding)

                with torch.no_grad():
                    result = inference_detector(self.model, color_image)
                    
                segm_info = result.pred_instances.masks
                bbox_info = result.pred_instances.bboxes.tolist()
                scores = result.pred_instances.scores.tolist()
                
                detect = []
                for segm, bbox, score in zip(segm_info, bbox_info, scores):
                    
                    if score > 0.40:
                        x1, y1, x2, y2 = map(int, bbox)
                        class_id = 0
                        detect.append([[x1, y1, x2 - x1, y2 - y1], score, class_id])
                                    
                tracks = self.tracker.update_tracks(detect, frame=color_image)
                distance_threshold = 500
                min_x, min_y = None, None  
                            
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    track_id = track.track_id
                    ltrb = track.to_ltrb()
                    class_id = track.get_det_class()
                    x1, y1, x2, y2 = map(int, ltrb)

                    # Find minimum distance within the segmented region
                    mask = segm_info[class_id][y1:y2, x1:x2]
                    segm_coords = np.where(mask.cpu().numpy()) #array format
                    min_distance = float('inf')

                    
                    if segm_coords[0].size > 0 and segm_coords[1].size > 0:
                        for y, x in zip(segm_coords[0] + y1, segm_coords[1] + x1):
                            depth_value = depth_image[y, x]
                            distance = depth_value
                            if distance != 0 and distance < min_distance:
                                min_distance = distance
                                min_x, min_y = x, y
                                
                        if min_distance / 10 <= distance_threshold:
                                mask = mask.cpu().numpy().astype(np.uint8)
                                mask = mask * 255  # Convert boolean to uint8

                                # Create a colored mask image
                                colored_mask = np.zeros_like(color_image, dtype=np.uint8)
                                colored_mask[y1:y2, x1:x2][mask == 255] = [0, 255, 0]  # Set the mask pixels to green color

                                # Overlay the colored mask on the original image
                                alpha = 0.5  # Adjust the transparency value (0 to 1)
                                cv2.addWeighted(color_image, 1 - alpha, colored_mask, alpha, 0, color_image)
                                color = self.colors[class_id]
                                B, G, R = map(int, color)
                                text = f"{track_id} - {self.class_names[class_id]}"
                                object_dist = f'{(min_distance / 10):.2f}'

                                cv2.rectangle(color_image, (x1, y1), (x2, y2), (B, G, R), 2)
                                cv2.rectangle(color_image, (x1 - 1, y1 - 20), (x1 + len(text) * 12, y1), (B, G, R), -1)
                                cv2.putText(color_image, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                cv2.circle(color_image, (min_x, min_y), 1, (0, 0, 255), 2)
                                cv2.putText(color_image, object_dist, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                output_path = os.path.join(self.output_dir, f'image_{i:04d}.png')
                cv2.imwrite(output_path, color_image)
                
    def run(self):
            self.process_images()

# Usage
image_bag_file_path = '../files/environment_lots_of_humans_1'
output_dir = 'Outputs'

parser = argparse.ArgumentParser()
parser.add_argument('--camera', type=int, required=True, help='Index of the camera to process (0, 1, 2, or 3)')
args = parser.parse_args()

processor = DetectionProcessor(image_bag_file_path, output_dir, args.camera)
processor.run()
