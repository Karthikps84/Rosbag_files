import os
import cv2
import torch
from rosbags.image import message_to_cvimage
from deep_sort_realtime.deepsort_tracker import DeepSort
from cv_bridge import CvBridge
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
    def __init__(self, image_bag_file_path, output_dir, camera_index):
        os.makedirs(output_dir, exist_ok=True)

        # Initialize the mmdetection model
        self.model_cfg = 'rtmdet-ins_m_8xb32-300e_coco.py'  # Configuration file for the model
        self.checkpoint = 'rtmdet-ins_m_8xb32-300e_coco_20221123_001039-6eba602e.pth'
        self.model = init_detector(self.model_cfg, self.checkpoint, device='cuda:0') 

    def process_images(self):
                    ltrb = track.to_ltrb()
                    class_id = track.get_det_class()
                    x1, y1, x2, y2 = map(int, ltrb)
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    # Find minimum distance within the segmented region
                    mask = segm_info[class_id][y1:y2, x1:x2]
                    segm_coords = np.where(mask.cpu().numpy()) #array format
                    min_distance = float('inf')


                    if segm_coords[0].size > 0 and segm_coords[1].size > 0:
                        for y, x in zip(segm_coords[0] + y1, segm_coords[1] + x1):
    def process_images(self):
                                cv2.rectangle(color_image, (x1 - 1, y1 - 20), (x1 + len(text) * 12, y1), (B, G, R), -1)
                                cv2.putText(color_image, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                cv2.circle(color_image, (min_x, min_y), 1, (0, 0, 255), 2)
                                cv2.putText(color_image, object_dist, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                                cv2.putText(color_image, object_dist, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

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
def main():
	# Usage
	rosbags = ['../files/environment_colaboration_arm_only_1','../files/environment_far_from_car_outside_1','../files/environment_lots_of_humans_1','../files/environment_part_detail_1','../files/environment_with_obs_1'
	]
	outs = ['../files/env_arm-colab', '../files/car_outside_1','../files/humans1', '../files/part_1', '../files/obs1']
	for bag, out in zip(rosbags, outs):
		image_bag_file_path = bag
		output_dir = out

		camera = 1

		processor = DetectionProcessor(image_bag_file_path, output_dir, camera)
		processor.run()
if __name__ == "__main__":
	main()
