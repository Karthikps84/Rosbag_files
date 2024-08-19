#!/usr/bin/env python3
import cv2
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

import torch
import argparse
import numpy as np
import pyrealsense2 as rs2
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from vision_msgs.msg import BoundingBox3D
from deep_sort_realtime.deepsort_tracker import DeepSort
from mmdet.apis import init_detector, inference_detector
from message_filters import Subscriber, ApproximateTimeSynchronizer

import traceback
from collections import deque
import signal



class CameraSubscriber(Node):
    """
    A ROS2 node that subscribes to camera topics, processes color and depth images,
    and publishes the distance and 3D world coordinates of detected objects.
    """

    def __init__(self, distance_threshold, detection_confidence_score=0.5):
        super().__init__('camera_subscriber')

        self.declare_parameter('camera_frame', 'camera_link')
        self.bridge = CvBridge()  # Initialize CvBridge for converting ROS Image messages to OpenCV images
        self.counters = {'cam_hand': 0, 'cam_center': 0, 'cam_right': 0, 'cam_left': 0}  # Counters for camera frames

        self.distance_threshold = distance_threshold
        self.detection_confidence_score = detection_confidence_score  # Confidence score threshold for detections

        # Initialize queues for custom scheduling
        self.queues = {name: deque() for name in ['cam_hand', 'cam_center', 'cam_right', 'cam_left']}

        # Initialize intrinsics for each camera
        self.intrinsics = {name: None for name in ['cam_hand', 'cam_center', 'cam_right', 'cam_left']}

        # Initialize DeepSort trackers
        self.trackers = {name: DeepSort(max_age=10, n_init=3) for name in
                         ['cam_hand', 'cam_center', 'cam_right', 'cam_left']}

        self.init_model()  # Initialize the detection model

        # Timer to alternate processing between cameras
        self.timer = self.create_timer(0.01, self.process_images_from_queue)

        # Add signal handler for graceful shutdown
        signal.signal(signal.SIGINT, lambda sig, frame: self.stop_node())

        # Camera topic names
        camera_topics = {
            'cam_hand': (
                '/cam_hand/cam_hand/color/image_raw', '/cam_hand/cam_hand/depth/image_rect_raw', '/cam_hand/cam_hand/depth/camera_info'),
            'cam_center': (
                '/cam_center/cam_center/color/image_raw', '/cam_center/cam_center/depth/image_rect_raw', '/cam_center/cam_center/depth/camera_info'),
            'cam_right': (
                '/cam_right/cam_right/color/image_raw', '/cam_right/cam_right/depth/image_rect_raw', '/cam_right/cam_right/depth/camera_info'),
            'cam_left': (
                '/cam_left/cam_left/color/image_raw', '/cam_left/cam_left/depth/image_rect_raw', '/cam_left/cam_left/depth/camera_info')
        }

        # Checking available Topics and raise error message if any camera topic is missing
        available_topics = [topic[0] for topic in self.get_topic_names_and_types()]

        # Subscriptions and synchronization for each camera
        for cam_name, topics in camera_topics.items():
            if topics[0] in available_topics and topics[1] in available_topics and topics[2] in available_topics:
                self.create_subscription(CameraInfo, topics[2],
                                         lambda msg, cam=cam_name: self.imageDepthInfoCallback(msg, cam), 10)
                self.get_logger().info(f"Subscribed to {topics[2]}")

                sync = ApproximateTimeSynchronizer([
                    Subscriber(self, Image, topics[0]),
                    Subscriber(self, Image, topics[1])
                ], queue_size=10, slop=0.1)
                sync.registerCallback(
                    lambda color_data, depth_data, cam=cam_name: self.queue_images(cam, color_data, depth_data))
                self.get_logger().info(f"Synchronized {topics[0]} and {topics[1]}")
            else:
                self.get_logger().info(f"Failed to subscribe or synchronize {cam_name}")

        # Publishers for distance and coordinates
        self.publisher_distance = self.create_publisher(String, 'distance', 10)
        self.publisher_coordinates = self.create_publisher(BoundingBox3D, 'boundingbox_3D', 10)

    def init_model(self):
        """
        Initialize the model.
        """
        self.get_logger().info("Initializing vision guided detection model...")
        try:
            model_cfg = 'model_cfg.py'  # Configuration file for the model
            checkpoint = 'vision_guided_checkpoint.pth'  # Checkpoint for the model
            self.model = init_detector(model_cfg, checkpoint, device='cuda:0')
        except Exception as e:
            self.get_logger().error(f"Error initializing model: {e}")

    def get_intrinsics(self, camera_info):
        """
        Get camera intrinsics from CameraInfo message.
        """
        intrinsics = rs2.intrinsics()
        intrinsics.width = camera_info.width
        intrinsics.height = camera_info.height
        intrinsics.ppx = camera_info.k[2]  # Principal point x-coordinate
        intrinsics.ppy = camera_info.k[5]  # Principal point y-coordinate
        intrinsics.fx = camera_info.k[0]  # Focal length x-coordinate
        intrinsics.fy = camera_info.k[4]  # Focal length y-coordinate
        if camera_info.distortion_model == 'plumb_bob':
            intrinsics.model = rs2.distortion.brown_conrady  # Set distortion model
        elif camera_info.distortion_model == 'equidistant':
            intrinsics.model = rs2.distortion.kannala_brandt4  # Set distortion coefficients
        intrinsics.coeffs = [i for i in camera_info.d]
        return intrinsics

    def imageDepthInfoCallback(self, cameraInfo, camera_name):
        """
        Callback function to process the camera info and initialize the camera intrinsics.
        """
        try:
            if self.intrinsics[camera_name] is None:
                self.intrinsics[camera_name] = self.get_intrinsics(cameraInfo)
                self.get_logger().info(f"{camera_name} intrinsics initialized: {self.intrinsics[camera_name]}")
        except CvBridgeError as e:
            self.get_logger().error(f"Error initializing camera intrinsics: {e}")

    def convert_bbox_to_meters(self, x1, y1, x2, y2, depth_scale, intrinsics):
        """
        Convert bounding box dimensions from pixels to meters.

        Args:
        x1, y1, x2, y2: Bounding box coordinates in pixels.
        depth_scale: Depth scale to convert depth values.
        intrinsics: Camera intrinsics with fx, fy, cx, cy.

        Returns:
        bbox_width_m: Width of the bounding box in meters.
        bbox_height_m: Height of the bounding box in meters.
        """
        # Convert pixel width and height to meters using the depth scale and intrinsics
        bbox_width_pixels = x2 - x1
        bbox_height_pixels = y2 - y1

        # Calculate width in meters
        bbox_width_m = (bbox_width_pixels * depth_scale) / intrinsics.fx
        bbox_height_m = (bbox_height_pixels * depth_scale) / intrinsics.fy

        return bbox_width_m, bbox_height_m

    def map_to_3d(self, x, y, depth_image, intrinsics):
        """
        Map 2D pixel coordinates to 3D real-world coordinates.
        """
        try:
            depth_pixel = [x, y]
            depth_value = depth_image[int(y), int(x)]

            # Deproject pixel to 3D point using intrinsics and depth value
            depth_point = rs2.rs2_deproject_pixel_to_point(intrinsics, [depth_pixel[0], depth_pixel[1]], depth_value)

            # Return the 3D point coordinates
            return depth_point
        except Exception as e:
            self.get_logger().error(f"Error mapping to 3D coordinates: {e}")
            return None

    def queue_images(self, camera_name, color_data, depth_data):
        """
        Queue images from a camera.
        """
        self.queues[camera_name].append((color_data, depth_data))

    def process_images_from_queue(self):
        """
        Process images from the queue, alternating between cameras.
        """
        for cam_name in ['cam_hand', 'cam_center', 'cam_right', 'cam_left']:
            if self.queues[cam_name]:
                color_data, depth_data = self.queues[cam_name].popleft()
                self.process_images(color_data, depth_data, cam_name)

    def calculate_min_distance(self, segm_coords, depth_image):
        """
        Calculate the minimum distance within the segmented region.
        """
        min_distance = float('inf')
        min_x, min_y = None, None

        if segm_coords[0].size > 0 and segm_coords[1].size > 0:
            for y, x in zip(segm_coords[0], segm_coords[1]):
                # Ensure coordinates are within bounds of the resized depth image
                if y >= depth_image.shape[0] or x >= depth_image.shape[1]:
                    continue
                distance = depth_image[y, x] * 100  # Converting from meter to CM
                if distance != 0 and distance < min_distance:
                    min_distance = distance
                    min_x, min_y = x, y

        return min_distance, min_x, min_y

    def process_images(self, color_data, depth_data, camera_name):
        """
        Process the color and depth images, detect objects, and publish distance and coordinates.
        """
        try:
            color_image = self.bridge.imgmsg_to_cv2(color_data, desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_data, desired_encoding='passthrough')

            # Check if depth image has valid data
            if depth_image is None or np.all(depth_image == 0):
                self.get_logger().warning("Depth image is invalid or not available.")
                return

            # Resize depth image to match the color image size
            depth_image = np.asanyarray(depth_image, dtype=np.float32)  # Ensure depth image is in float32 format
            resized_depth_img = cv2.resize(depth_image, (color_image.shape[1], color_image.shape[0]))

            # Running a model forward pass to get predictions
            with torch.no_grad():
                result = inference_detector(self.model, color_image)

            masks = result.pred_instances.masks  # Extract masks from the result
            bbox_info = result.pred_instances.bboxes.tolist()  # Extract bounding boxes from the result
            scores = result.pred_instances.scores.tolist()  # Extract scores from the result
            classes = result.pred_instances.labels.tolist()
            detect = []
            filtered_masks = []

            for idx, (bbox, score, class_id) in enumerate(zip(bbox_info, scores, classes)):
                if score >= self.detection_confidence_score:  # Filter detections based on confidence score
                    x1, y1, x2, y2 = map(int, bbox)
                    detect.append([[x1, y1, x2 - x1, y2 - y1], score, class_id])  # Append detection
                    filtered_masks.append(masks[idx].cpu())  # Append corresponding mask

            # Update tracker with detections and masks
            for track in self.trackers[camera_name].update_tracks(detect, instance_masks=filtered_masks,
                                                                  frame=color_image):
                if not track.is_confirmed() and track.get_instance_mask() is None:
                    continue
                    # Find minimum distance within the instance mask from tracked bounding boxes
                mask = track.get_instance_mask()
                segm_cords = np.where(mask)
                object_distance, min_x, min_y = self.calculate_min_distance(segm_cords, resized_depth_img)

                if object_distance <= self.distance_threshold:  # Check if object is within distance threshold

                    # Compute updated bbox from instance mask
                    y_cords, x_cords = segm_cords
                    if len(y_cords) > 0 and len(x_cords) > 0:
                        y1, y2 = np.min(y_cords), np.max(y_cords)
                        x1, x2 = np.min(x_cords), np.max(x_cords)
                    else:
                        continue  # Skip if mask is empty

                    # Publish distance information
                    object_distance = object_distance / 100  # Convert mm to meters

                    distance_msg = String()
                    distance_msg.data = f"From {camera_name}, Distance: {object_distance} meters"
                    self.publisher_distance.publish(distance_msg)  # Publish the distance message
                    # self.get_logger().info(f" Obstacle detected! from {camera_name} distance: {object_distance} meters")

                    # Get the depth at the center of the bounding box
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    bbox_width_m, bbox_height_m = self.convert_bbox_to_meters(x1, y1, x2, y2, object_distance,
                                                                              self.intrinsics[camera_name])

                    # Publish 3D BoundingBox with World Coordinates
                    coordinates_data = self.map_to_3d(center_x, center_y, resized_depth_img,
                                                      self.intrinsics[camera_name])
                    message = BoundingBox3D()
                    message.center.position.x = ((coordinates_data[0]) / 1000)  # Convert mm to meters
                    message.center.position.y = -(
                    coordinates_data[1]) / 1000  # Convert mm to meters and adjust for ROS coordinate system
                    message.center.position.z = ((coordinates_data[2]) / 1000)  # Convert mm to meters
                    message.size.x = float(bbox_width_m)  # Width of the bounding box
                    message.size.y = float(bbox_height_m)  # Height of the bounding box
                    message.size.z = object_distance  # Distance to the object
                    self.publisher_coordinates.publish(message)  # Publish the message
                    # self.get_logger().info(f" Published Message from {camera_name}: {message}")

        except Exception as e:
            self.get_logger().error(f"Unexpected Error: {e}")
            self.get_logger().error(traceback.format_exc())

# Add a method to handle graceful shutdown
    def stop_node(self):
        self.get_logger().info("Shutting down...")
        self.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--distance_threshold', type=float, default=50,
                        help='Distance threshold for object detection. Default is set to 50 CM.')
    parser.add_argument('--detection_confidence_score', type=float, default=0.5,
                        help='Detection confidence score to filter detections. Default is 0.5.')
    parser.add_argument('--num_cameras', type=int, default=4,
                        help='Number of cameras to subscribe. Same for number of CPU Cores to be utilized. Default is 4.')
    args = parser.parse_args()

    rclpy.init(args=None)
    camera_subscriber = CameraSubscriber(args.distance_threshold, args.detection_confidence_score)

    # Create a multi-threaded executor
    executor = MultiThreadedExecutor(num_threads=args.num_cameras)
    # Add the node to the executor
    executor.add_node(camera_subscriber)

    # Spin the node using the executor
    try:
        executor.spin()
    finally:
        # Shutdown the node and the executor gracefully
        camera_subscriber.destroy_node()
        rclpy.shutdown()
