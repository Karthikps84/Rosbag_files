#!/usr/bin/env python3

import sys
import rclpy
from rclpy.node import Node

import torch
import numpy as np
import pyrealsense2 as rs2
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from sensor_msgs.msg import CameraInfo
from message_filters import Subscriber, ApproximateTimeSynchronizer
from vision_msgs.msg import BoundingBox3D
from mmdet.apis import init_detector, inference_detector

# Dictionary for labels
label_map = {
    0: u'__background__', 1: u'person', 2: u'bicycle', 3: u'car', 4: u'motorcycle',
    5: u'airplane', 6: u'bus', 7: u'train', 8: u'truck', 9: u'boat', 10: u'traffic light',
    11: u'fire hydrant', 12: u'stop sign', 13: u'parking meter', 14: u'bench', 15: u'bird',
    16: u'cat', 17: u'dog', 18: u'horse', 19: u'sheep', 20: u'cow', 21: u'elephant',
    22: u'bear', 23: u'zebra', 24: u'giraffe', 25: u'backpack', 26: u'umbrella',
    27: u'handbag', 28: u'tie', 29: u'suitcase', 30: u'frisbee', 31: u'skis', 32: u'snowboard',
    33: u'sports ball', 34: u'kite', 35: u'baseball bat', 36: u'baseball glove', 37: u'skateboard',
    38: u'surfboard', 39: u'tennis racket', 40: u'bottle', 41: u'wine glass', 42: u'cup',
    43: u'fork', 44: u'knife', 45: u'spoon', 46: u'bowl', 47: u'banana', 48: u'apple',
    49: u'sandwich', 50: u'orange', 51: u'broccoli', 52: u'carrot', 53: u'hot dog',
    54: u'pizza', 55: u'donut', 56: u'cake', 57: u'chair', 58: u'couch', 59: u'potted plant',
    60: u'bed', 61: u'dining table', 62: u'toilet', 63: u'tv', 64: u'laptop', 65: u'mouse',
    66: u'remote', 67: u'keyboard', 68: u'cell phone', 69: u'microwave', 70: u'oven',
    71: u'toaster', 72: u'sink', 73: u'refrigerator', 74: u'book', 75: u'clock',
    76: u'vase', 77: u'scissors', 78: u'teddy bear', 79: u'hair drier', 80: u'toothbrush'
}

bridge = CvBridge()

class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')

        self.declare_parameter('camera_frame', 'camera_link')
        
        # Create ApproximateTimeSynchronizer objects for each camera
        self.hand_cam = self.create_subscription(CameraInfo, 
                                                 '/cam_hand/depth/camera_info', 
                                                 self.imageDepthInfoCallback, 10)
        self.hand_sync = ApproximateTimeSynchronizer([
            Subscriber(self, Image, '/cam_hand/color/image_raw/compressed'),
            Subscriber(self, Image, '/cam_hand/depth/image_rect_raw')
        ], 10, 0.1)
        self.hand_sync.registerCallback(self.process_images)
        
        self.center_cam = self.create_subscription(CameraInfo, 
                                                 '/cam_hand/depth/camera_info', 
                                                 self.imageDepthInfoCallback, 10)
        self.center_sync = ApproximateTimeSynchronizer([
            Subscriber(self, Image, '/cam_center/color/image_raw/compressed'),
            Subscriber(self, Image, '/cam_center/depth/image_rect_raw')
        ], 10, 0.1)
        self.center_sync.registerCallback(self.process_images)
        

        self.right_cam = self.create_subscription(CameraInfo, 
                                                 '/cam_hand/depth/camera_info', 
                                                 self.imageDepthInfoCallback, 10)
        self.right_sync = ApproximateTimeSynchronizer([
            Subscriber(self, Image, '/cam_right/color/image_raw/compressed'),
            Subscriber(self, Image, '/cam_right/depth/image_rect_raw')
        ], 10, 0.1)
        self.right_sync.registerCallback(self.process_images)

        self.left_cam = self.create_subscription(CameraInfo, 
                                                 '/cam_hand/depth/camera_info', 
                                                 self.imageDepthInfoCallback, 10)
        self.left_sync = ApproximateTimeSynchronizer([
            Subscriber(self, Image, '/cam_left/color/image_raw/compressed'),
            Subscriber(self, Image, '/cam_left/depth/image_rect_raw')
        ], 10, 0.1)
        self.left_sync.registerCallback(self.process_images)
        
        self.publisher_distance = self.create_publisher(Float32, 'distance_topic', 10)
        self.publisher_coordinates = self.create_publisher(BoundingBox3D, 'world_coordinates', 10)
        
        self.intrinsics = None

        self.init_model()  # Initialize the model

    
    def init_model(self):
        # Initialize the mmdetection model
        model_cfg = 'rtmdet-ins_m_8xb32-300e_coco.py'  # Configuration file for the model
        checkpoint = 'rtmdet-ins_m_8xb32-300e_coco_20221123_001039-6eba602e.pth' # Checkpoint for the model
        self.model = init_detector(model_cfg, checkpoint, device='cuda:0')
        
    def imageDepthInfoCallback(self, cameraInfo):
        try:
            if self.intrinsics:
                return
            # Initialize the camera intrinsics with values from the CameraInfo message
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = cameraInfo.width
            self.intrinsics.height = cameraInfo.height
            self.intrinsics.ppx = cameraInfo.k[2]  # Principal point x-coordinate
            self.intrinsics.ppy = cameraInfo.k[5]  # Principal point y-coordinate
            self.intrinsics.fx = cameraInfo.k[0]   # Focal length x-coordinate
            self.intrinsics.fy = cameraInfo.k[4]   # Focal length y-coordinate
            if cameraInfo.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs2.distortion.brown_conrady  # Set distortion model
            elif cameraInfo.distortion_model == 'equidistant':
                self.intrinsics.model = rs2.distortion.kannala_brandt4
            self.intrinsics.coeffs = [i for i in cameraInfo.d]  # Set distortion coefficients
        except CvBridgeError as e:
            print(e)  # Print error if exception occurs
            return

    def map_to_3d(self, x, y, depth_image):
        # Map 2D pixel coordinates to 3D real-world coordinates
        depth_pixel = [x, y]
        depth_value = depth_image[int(y), int(x)]

        # Deproject pixel to 3D point using intrinsics and depth value
        depth_point = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [depth_pixel[0], depth_pixel[1]], depth_value)

        # Return the 3D point coordinates
        return depth_point

    def calculate_distance(self, x1, y1, segm_coords, depth_img):
        if segm_coords[0].size > 0 and segm_coords[1].size > 0:
            # Iterate over the segmented coordinates
            for y, x in zip(segm_coords[0] + y1, segm_coords[1] + x1):
                min_distance = float('inf')
                depth_value = depth_img[y, x]
                distance = depth_value
                # Update the minimum distance if the current distance is smaller
                if distance != 0 and distance < min_distance:
                    min_distance = distance
        return min_distance / 10  # Return the minimum distance in centimeters

    
    def process_images(self, color_data, depth_data):
        # Process both color and depth images
        color_img = bridge.imgmsg_to_cv2(color_data, "bgr8")  # Convert color image from ROS message to OpenCV format
        depth_img = bridge.imgmsg_to_cv2(depth_data, desired_encoding='passthrough')  # Convert depth image similarly
        
        # Inference using the pre-trained model
        with torch.no_grad():
            result = inference_detector(self.model, color_img)  # Perform inference on the color image

        # Extract segmentation and bounding box information
        segm_info = result.pred_instances.masks
        bbox_info = result.pred_instances.bboxes.tolist()
        scores = result.pred_instances.scores.tolist()

        detect = []
        # Filter detections based on a confidence threshold
        for bbox, score in zip(bbox_info, scores):
            if score > 0.40:
                x1, y1, x2, y2 = map(int, bbox)
                class_id = 0
                detect.append([[x1, y1, x2 - x1, y2 - y1], score, class_id])

        # Update tracks with detections
        tracks = self.tracker.update_tracks(detect, frame=color_img)
        distance_threshold = 60

        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb) #Updated bounding boxes

            # Find minimum distance within the segmented region
            mask = segm_info[0][y1:y2, x1:x2]
            segm_coords = np.where(mask.cpu().numpy())

            # Calculate the distance to the object
            object_dist = self.calculate_distance(x1, y1, segm_coords, depth_img)
            if object_dist <= distance_threshold:
                # Publish distance information
                distance_msg = Float32(data=object_dist)
                self.publisher_distance.publish(distance_msg) # Publish the distance message

                # Publish 3D world coordinates
                coordinates_data = self.map_to_3d(int((x1 + x2) / 2), int((y1 + y2) / 2), depth_img)
                message = BoundingBox3D()
                message.header.stamp = self.get_clock().now().to_msg()
                message.header.frame_id = self.get_parameter('camera_frame').get_parameter_value().string_value
                message.pose.position.z = ((coordinates_data[0]) / 1000)  # Convert mm to meters
                message.pose.position.y = -(coordinates_data[1]) / 1000  # Convert mm to meters and adjust for ROS coordinate system
                message.pose.position.x = ((coordinates_data[2]) / 1000)  # Convert mm to meters
                message.dimensions.x = (x2 - x1)  # Width of the bounding box
                message.dimensions.y = (y2 - y1)  # Height of the bounding box
                message.dimensions.z = object_dist  # Distance to the object
                self.publisher_coordinates.publish(message)  # Publish the message

# Main function to initialize and run the node
if __name__ == '__main__':
    rclpy.init(args=None)  # Initialize the ROS2 Python client library
    camera_subscriber = CameraSubscriber()  # Create an instance of the CameraSubscriber node
    rclpy.spin(camera_subscriber)  # Keep the node running until interrupted
    rclpy.shutdown()  # Shutdown the ROS2 client library
