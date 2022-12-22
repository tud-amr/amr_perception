from easy_inference.providers.realsense import Realsense
from easy_inference.providers.utils import combine
from easy_inference.utils.boundingbox import BoundingBox, drawBoxes
from easy_inference.utils.skeleton import Skeleton, drawSkeletons
from easy_inference.utils.filters import filter_iou3d
from easy_inference.utils.pad_frames import pad_frames

import onnxruntime as ort
import numpy as np
import cv2
import os

ROS = os.getenv("ROS", 0)
SHOW = os.getenv("SHOW", 0)

# ort.set_default_logger_severity(0)
models_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/models'
ort_sess = ort.InferenceSession(f'{models_dir}/yolov7-w6-pose.onnx', providers=['CUDAExecutionProvider'])

cam5 = Realsense(width=640, height=480, depth=True, pointcloud=True, device='215122255929')
cam4 = Realsense(width=640, height=480, depth=True, pointcloud=True, device='215122255934')
cam3 = Realsense(width=640, height=480, depth=True, pointcloud=True, device='215122254701')
cam2 = Realsense(width=640, height=480, depth=True, pointcloud=True, device='114222251376')
cam1 = Realsense(width=640, height=480, depth=True, pointcloud=True, device='215122255869')
providers = [cam1, cam2, cam3, cam4, cam5]

if ROS:
    from easy_inference.utils.ros_connector import RosConnector
    ros_connector = RosConnector(num_cameras=len(providers), fixed_frame='base_link')

for frames in combine(*providers):
    rgb_frames = np.stack([f[0] for f in frames]).transpose((0, 3, 1, 2))
    depth_frames = np.stack([f[1] for f in frames])
    pointclouds = [f[2] for f in frames]

    network_input_dim = [640, 512]
    rgb_frames = pad_frames(rgb_frames, width=network_input_dim[0], height=network_input_dim[1])
    depth_frames = pad_frames(depth_frames, width=network_input_dim[0], height=network_input_dim[1])

    input = np.ascontiguousarray(rgb_frames)
    input = input.astype(np.float32)
    input /= 255

    all_outputs = ort_sess.run(None, {'images': input})

    # convert to Skeleton for convenience
    persons = []
    for batch_id, outputs in enumerate(all_outputs):
        persons += [Skeleton(*output[0:4], class_id=output[5], confidence=output[4], kpts=output[6:], batch_id=batch_id) for output in outputs]

    # filter classes
    persons = [person for person in persons if person.class_id == 0]

    # filter confidence
    persons = [person for person in persons if person.confidence > 0.7]

    persons3d = [person.to3d(depth_frames[person.batch_id], providers[person.batch_id]._depth_intr) for person in persons]

    if ROS: 
        ros_connector.publishPersons3d(persons3d)
        ros_connector.publishPointclouds(pointclouds)

    if SHOW:
        rgb_frames = rgb_frames.transpose((0, 2, 3, 1))
        rgb_frames = np.ascontiguousarray(rgb_frames)
        for person in persons:
            drawBoxes(rgb_frames[person.batch_id], [person])
            drawBoxes(depth_frames[person.batch_id], [person])

            drawSkeletons(rgb_frames[person.batch_id], [person])
            drawSkeletons(depth_frames[person.batch_id], [person])

        all_frames = np.concatenate(
            (np.concatenate(rgb_frames[:3], axis=1), 
            np.concatenate((rgb_frames[3], rgb_frames[4], np.zeros_like(rgb_frames[4])), axis=1)),
            axis=0)
        cv2.imshow(f'frame', all_frames)

        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()

