"""
This node locates Aruco AR markers in images and publishes their ids and poses.

Subscriptions:
   /camera/image_raw (sensor_msgs.msg.Image)
   /camera/camera_info (sensor_msgs.msg.CameraInfo)
   /camera/camera_info (sensor_msgs.msg.CameraInfo)

Published Topics:
    /aruco_poses (geometry_msgs.msg.PoseArray)
       Pose of all detected markers (suitable for rviz visualization)

    /aruco_markers (ros2_aruco_interfaces.msg.ArucoMarkers)
       Provides an array of all poses along with the corresponding
       marker ids.

Parameters:
    marker_size - size of the markers in meters (default .0625)
    aruco_dictionary_id - dictionary that was used to generate markers
                          (default DICT_5X5_250)
    image_topic - image topic to subscribe to (default /camera/image_raw)
    camera_info_topic - camera info topic to subscribe to
                         (default /camera/camera_info)

Author: Nathan Sprague
Version: 10/26/2020

"""

import rclpy
import rclpy.node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import numpy as np
import cv2
from ros2_aruco import transformations

from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
from ros2_aruco_interfaces.msg import ArucoMarkers
from ros2_aruco_interfaces.srv import GetMaskImage
from math import *
import copy
from matplotlib import pyplot as plt

DEPTH_THRESHOLD = 8

class ArucoImageMasker:
    def __init__(self, ad, ap):
        self.aruco_dictionary = ad
        self.aruco_parameters = ap
        # self.id_list = aruco_id_list

        self.masked_rgb = dict()
        self.masked_depth = dict()
        self.mask_corners = dict()
        self.rgb_masks = dict()
        self.depth_masks = dict()
        self.empty_depth_img = dict()

    def set_empty_depth_img(self, depth_img, id):
        self.empty_depth_img[id] = copy.deepcopy(depth_img)

    def get_masked_img(self, rgb_img=None, depth_img=None, id=1):
        assert rgb_img.shape[:2] == depth_img.shape[:2]
        if id not in self.mask_corners \
           and (depth_img is None or not self.find_mask_corners(rgb_img, id)):
            return None, None, None
        if id not in self.rgb_masks or id not in self.depth_masks:
            self.rgb_masks[id] = self.generate_mask(self.mask_corners[id], rgb_img, np.uint8)
            self.depth_masks[id] = self.generate_mask(self.mask_corners[id], depth_img, np.uint16)
            
        if id in self.empty_depth_img:
            rgb_mask, depth_mask, seg_mask_uint8 = self.compute_segmentation_mask(
                depth_img, self.empty_depth_img[id], self.mask_corners[id])
        else:
            rgb_mask = self.rgb_masks[id]
            depth_mask = self.depth_masks[id]
            bound = [np.min(self.mask_corners[id], axis=0), 
                     np.max(self.mask_corners[id], axis=0)]
            size = [bound[1][1] - bound[0][1], bound[1][0] - bound[0][0]]
            seg_mask_uint8 = 255 * np.ones(size, dtype=np.uint8)
    
        masked_rgb = np.bitwise_and(rgb_img, rgb_mask)
        masked_depth = np.bitwise_and(depth_img, depth_mask)


        return masked_rgb, masked_depth, seg_mask_uint8, self.mask_corners[id]
    
    def compute_segmentation_mask(self, img, empty_img, corners):
        bound = [np.min(corners, axis=0), np.max(corners, axis=0)]
        bounded_img = img[bound[0][1]:bound[1][1], bound[0][0]:bound[1][0]]
        bounded_empty_img = empty_img[bound[0][1]:bound[1][1], bound[0][0]:bound[1][0]]
        sub_img = np.array(bounded_empty_img, dtype=np.int32) - np.array(bounded_img, dtype=np.int32)
        rgb_mask_color = [np.iinfo(np.uint8).max]
        depth_mask_color = np.iinfo(np.uint16).max
        rgb_seg_mask = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        depth_seg_mask = np.zeros(img.shape, np.uint16)
        bounded_rgb_seg_mask = rgb_seg_mask[bound[0][1]:bound[1][1], bound[0][0]:bound[1][0]]
        bounded_depth_seg_mask = depth_seg_mask[bound[0][1]:bound[1][1], bound[0][0]:bound[1][0]]
        seg_mask_uint8 = np.zeros(sub_img.shape, dtype=np.uint8)
        for i, row in enumerate(sub_img):
            for j, x in enumerate(row):
                if x > DEPTH_THRESHOLD:
                    bounded_rgb_seg_mask[i][j] = rgb_mask_color
                    bounded_depth_seg_mask[i][j] = depth_mask_color
                    seg_mask_uint8[i][j] = np.iinfo(np.uint8).max
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (DEPTH_THRESHOLD, DEPTH_THRESHOLD))
        seg_mask_uint8 = cv2.dilate(cv2.erode(seg_mask_uint8, kernel), kernel)
        
        rgb_seg_mask[bound[0][1]:bound[1][1], bound[0][0]:bound[1][0]] = bounded_rgb_seg_mask
        depth_seg_mask[bound[0][1]:bound[1][1], bound[0][0]:bound[1][0]] = bounded_depth_seg_mask
        return rgb_seg_mask, depth_seg_mask, seg_mask_uint8

    def find_mask_corners(self, rgb_img, target_id):
        corners, ids = self.find_aruco(rgb_img)
        num_of_id = dict()
        for [id] in ids:
            num_of_id[id] = 0
        for [id] in ids: 
            num_of_id[id] += 1
        if target_id not in num_of_id or num_of_id[target_id] != 4:
            return False
        self.mask_corners[id] = []
        for i, [id] in enumerate(ids):
            if id != target_id:
                continue
            self.mask_corners[id].append([int(x) for x in corners[i][0][0]])
        return True
        # for i, [id] in enumerate(ids):
        #     if num_of_id[id] != 4 or id not in self.id_list:
        #         print('id = {}, num_of_id[id] = {}'.format(id, num_of_id[id]))
        #         continue
        #     if id not in self.mask_corners:
        #         self.mask_corners[id] = []
        #     print('corners = {}'.format(corners))
        #     self.mask_corners[id].append([int(x) for x in corners[i][0][0]])

    # def generate_masked_img(self, rgb_img, depth_img, id):
    #     self.masked_rgb[id] = np.bitwise_and(rgb_img, self.rgb_masks[id])
    #     self.masked_depth[id] = np.bitwise_and(depth_img, self.depth_masks[id])
        # for id, corner in self.mask_corners.items():
        #     self.masked_rgb[id] = self.mask_image(corner, rgb_img, np.uint8)
        #     self.masked_depth[id] = self.mask_image(corner, depth_img, np.uint16)

    def generate_mask(self, corner, img, dtype=np.uint8):
        mask = np.zeros(img.shape, dtype)
        mask_color = [np.iinfo(dtype).max] * (1 if len(img.shape) < 3 else img.shape[2])
        print('corner = {}'.format(corner))
        poly_points = np.array([corner[0], corner[1], corner[2]])
        cv2.fillPoly(mask, pts=[poly_points], color=mask_color)
        poly_points = np.array([corner[0], corner[1], corner[3]])
        cv2.fillPoly(mask, pts=[poly_points], color=mask_color)
        poly_points = np.array([corner[0], corner[2], corner[3]])
        cv2.fillPoly(mask, pts=[poly_points], color=mask_color)
        return mask

    def has_rgb_img(self, id):
        return id in self.masked_rgb
        
    def has_depth_img(self, id):
        return id in self.masked_depth
    
    def get_masked_rgb_img(self, id):
        return self.masked_rgb[id] if self.has_rgb_img(id) else None
    
    def get_masked_depth_img(self, id):
        return self.masked_depth[id] if self.has_depth_img(id) else None
    
    def get_mask_corners(self, id):
        return self.mask_corners[id] if self.has_rgb_img(id) else None
    
    def find_aruco(self, rgb_img):
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY) 
        corners, marker_ids, _ = cv2.aruco.detectMarkers(
            gray_img, self.aruco_dictionary, parameters=self.aruco_parameters)
        return corners, marker_ids


class MaskColorAndDepthImage(rclpy.node.Node):

    def __init__(self):
        super().__init__('aruco_node')

        # Declare and read parameters
        # self.declare_parameter("marker_size", .053)
        # self.declare_parameter("aruco_dictionary_id", "DICT_5X5_250")
        # self.declare_parameter("image_topic", "/camera/color/image_raw")#???
        # self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        # self.declare_parameter("camera_frame", None)
        # self.declare_parameter("depth_image_topic", "/camera/aligned_depth_to_color/image_raw")
        # self.declare_parameter("marker_info_topic", "/aruco_markers")

        self.declare_parameter("marker_size", .0457)
        self.declare_parameter("aruco_dictionary_id", "DICT_4X4_50")
        self.declare_parameter("image_topic", "/rgb/image_raw")
        self.declare_parameter("rgb_info_topic", "/rgb/camera_info")
        self.declare_parameter("depth_image_topic", "/depth_to_rgb/image_raw")
        self.declare_parameter("depth_info_topic", "/depth_to_rgb/camera_info")
        self.declare_parameter("camera_frame", "camera_link")
        self.declare_parameter("marker_info_topic", "/aruco_markers")

        self.marker_size = self.get_parameter("marker_size").get_parameter_value().double_value
        dictionary_id_name = self.get_parameter(
            "aruco_dictionary_id").get_parameter_value().string_value
        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        rgb_info_topic = self.get_parameter("rgb_info_topic").get_parameter_value().string_value
        depth_image_topic = self.get_parameter("depth_image_topic").get_parameter_value().string_value
        depth_info_topic = self.get_parameter("rgb_info_topic").get_parameter_value().string_value
        self.camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value
        self.armarker_info_topic = self.get_parameter("marker_info_topic").get_parameter_value().string_value

        self.corners = []
        self.marker_ids = []
        
        self.rgb_img = None
        self.depth_img = None

        self.masked_image = []
        self.get_marker = False
        self.get_mask_image = False

        # Make sure we have a valid dictionary id:
        try:
            dictionary_id = cv2.aruco.__getattribute__(dictionary_id_name)
            if type(dictionary_id) != type(cv2.aruco.DICT_4X4_50):
                raise AttributeError
        except AttributeError:
            self.get_logger().error("bad aruco_dictionary_id: {}".format(dictionary_id_name))
            options = "\n".join([s for s in dir(cv2.aruco) if s.startswith("DICT")])
            self.get_logger().error("valid options: {}".format(options))


        # Set up subscriptions
        self.rgb_info_sub = self.create_subscription(CameraInfo,
                                                 rgb_info_topic,
                                                 self.rgb_info_callback,
                                                 qos_profile_sensor_data)
        
        self.depth_info_sub = self.create_subscription(CameraInfo,
                                                 depth_info_topic,
                                                 self.depth_info_callback,
                                                 qos_profile_sensor_data)

        self.create_subscription(Image, image_topic,
                                 self.image_callback, qos_profile_sensor_data)


        self.create_subscription(Image, depth_image_topic,
                                 self.depth_image_callback, qos_profile_sensor_data)
        # Set up publishers
        # self.poses_pub = self.create_publisher(PoseArray, 'aruco_poses', 10)
        # self.markers_pub = self.create_publisher(ArucoMarkers, 'aruco_markers', 10)

        # self.armarker_Info_subscriber_ = self.create_subscription(ArucoMarkers, self.armarker_info_topic, 
        #                                            self.armarker_Info_callback, 10)
        self.image_masking_service = self.create_service(GetMaskImage, "image_masking", 
                                           self.image_masking_cb)


        # Set up fields for camera parameters
        self.rgb_info_msg = None
        self.depth_info_msg = None
        self.intrinsic_mat = None
        self.distortion = None

        self.aruco_dictionary = cv2.aruco.Dictionary_get(dictionary_id)
        self.aruco_parameters = cv2.aruco.DetectorParameters_create()
        self.img_masker = ArucoImageMasker(self.aruco_dictionary, self.aruco_parameters)

        self.bridge = CvBridge()

    def rgb_info_callback(self, info_msg):
        self.rgb_info_msg = info_msg
        self.intrinsic_mat = np.reshape(np.array(self.rgb_info_msg.k), (3, 3))
        self.distortion = np.array(self.rgb_info_msg.d)
        # Assume that camera parameters will remain the same...
        self.destroy_subscription(self.rgb_info_sub)

    def depth_info_callback(self, info_msg):
        self.depth_info_msg = info_msg
        self.destroy_subscription(self.depth_info_sub)

    def image_callback(self, img_msg):

        # if self.info_msg is None:
        #     self.get_logger().warn("No camera info has been received!")
        #     return

        # cv_image = self.bridge.imgmsg_to_cv2(img_msg,
        #                                      desired_encoding='mono8')
        self.rgb_img = self.bridge.imgmsg_to_cv2(img_msg,
            desired_encoding='bgr8')
        
        # markers = ArucoMarkers()
        # pose_array = PoseArray()
        # if self.camera_frame is None:
        #     markers.header.frame_id = self.info_msg.header.frame_id
        #     pose_array.header.frame_id = self.info_msg.header.frame_id
        # else:
        #     markers.header.frame_id = self.camera_frame
        #     pose_array.header.frame_id = self.camera_frame
            
            
        # markers.header.stamp = img_msg.header.stamp
        # pose_array.header.stamp = img_msg.header.stamp

        # self.corners, self.marker_ids, rejected = cv2.aruco.detectMarkers(cv_image,
        #                                                         self.aruco_dictionary,
        #                                                         parameters=self.aruco_parameters)
        # if self.marker_ids is not None:
        #     self.get_marker = True

        #     if cv2.__version__ > '4.0.0':
        #         rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(self.corners,
        #                                                               self.marker_size, self.intrinsic_mat,
        #                                                               self.distortion)
        #     else:
        #         rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(self.corners,
        #                                                            self.marker_size, self.intrinsic_mat,
        #                                                            self.distortion)
            
        #     # for i in range(len(rvecs)):
        #     #     self.cv_image = cv2.aruco.drawAxis(self.cv_image, 
        #     #                                        self.intrinsic_mat, 
        #     #                                        self.distortion, 
        #     #                                        rvecs[i], 
        #     #                                        tvecs[i], 
        #     #                                        0.02)
        #     # cv2.imshow('QueryImage', self.cv_image)
        #     # cv2.waitKey(0)
        #     # cv2.destroyAllWindows()
        #     for i, marker_id in enumerate(self.marker_ids):
        #         pose = Pose()
        #         pose.position.x = tvecs[i][0][0]
        #         pose.position.y = tvecs[i][0][1]
        #         pose.position.z = tvecs[i][0][2]

        #         rot_matrix = np.eye(4)
        #         rot_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
        #         quat = transformations.quaternion_from_matrix(rot_matrix)

        #         pose.orientation.x = quat[0]
        #         pose.orientation.y = quat[1]
        #         pose.orientation.z = quat[2]
        #         pose.orientation.w = quat[3]

        #         pose_array.poses.append(pose)
        #         markers.poses.append(pose)
        #         markers.marker_ids.append(marker_id[0])

        #     self.poses_pub.publish(pose_array)
        #     self.markers_pub.publish(markers)
        # else:
        #     self.get_marker = False

    def depth_image_callback(self, depth_img_msg):
        self.depth_img = self.bridge.imgmsg_to_cv2(depth_img_msg,
            desired_encoding='16UC1')
        
        

    # def armarker_Info_callback(self, data):

    #     # print(self.marker_ids)
    #     id1_marker_pos = []
    #     id2_marker_pos = []
    #     if len(data.marker_ids) >= 8:
    #         for i in range(8):
    #             if self.marker_ids[i] == 1:
    #                 id1_marker_pos.append([int(self.corners[i][0][0][0]), int(self.corners[i][0][0][1])])
    #             elif self.marker_ids[i] == 2:
    #                 id2_marker_pos.append([int(self.corners[i][0][0][0]), int(self.corners[i][0][0][1])])
    #         marker_pos = [id1_marker_pos, id2_marker_pos]
    #         self.mask_image(marker_pos)
    #     elif len(data.marker_ids) == 4 and self.marker_ids[:] == 1:

    #         for i in range(4):
    #             id1_marker_pos.append([int(self.corners[i][0][0][0]), int(self.corners[i][0][0][1])])
    #         marker_pos = [id1_marker_pos]

    #         self.mask_image(marker_pos)
    #     elif len(data.marker_ids) == 4 and self.marker_ids[:] == 2:

    #         for i in range(4):
    #             id2_marker_pos.append([int(self.corners[i][0][0][0]), int(self.corners[i][0][0][1])])
    #         marker_pos = [id2_marker_pos]

    #         self.mask_image(marker_pos)
    #     else:
    #         self.get_mask_image = False
    #         return
                 
    # def mask_image(self, marker_pos):

    #     masked_img_list = []
        
    #     print(marker_pos)
    #     for i in range(len(marker_pos)):
    #         mask = np.zeros(self.cv_image.shape[:3], np.uint8)
    #         poly_points = np.array([marker_pos[i][0], marker_pos[i][1], marker_pos[i][2]])
    #         cv2.fillPoly(mask, pts=[poly_points], color=(255, 255, 255))
    #         poly_points = np.array([marker_pos[i][0], marker_pos[i][1], marker_pos[i][3]])
    #         cv2.fillPoly(mask, pts=[poly_points], color=(255, 255, 255))
    #         poly_points = np.array([marker_pos[i][0], marker_pos[i][2], marker_pos[i][3]])
    #         cv2.fillPoly(mask, pts=[poly_points], color=(255, 255, 255))

    #         print(type(self.cv_image[0][0][0]))
    #         masked_img = np.bitwise_and(self.cv_image, mask)
        
    #         masked_img_list.append(self.bridge.cv2_to_imgmsg(masked_img, encoding="passthrough"))

    #     for k in range(len(marker_pos)):
    #         depth_mask = np.zeros(self.depth_image.shape[:2], np.uint16)
    #         poly_points = np.array([marker_pos[k][0], marker_pos[k][1], marker_pos[k][2]])
    #         cv2.fillPoly(depth_mask, pts=[poly_points], color=(255, 255, 255))
    #         poly_points = np.array([marker_pos[k][0], marker_pos[k][1], marker_pos[k][3]])
    #         cv2.fillPoly(depth_mask, pts=[poly_points], color=(255, 255, 255))
    #         poly_points = np.array([marker_pos[k][0], marker_pos[k][2], marker_pos[k][3]])
    #         cv2.fillPoly(depth_mask, pts=[poly_points], color=(255, 255, 255))
            
    #         print(type(self.depth_image[0][0]))
    #         depth_masked_img = np.bitwise_and(self.depth_image, depth_mask)
        
    #         masked_img_list.append(self.bridge.cv2_to_imgmsg(depth_masked_img, encoding="passthrough"))

    #     self.masked_image = masked_img_list
    #     if type(self.masked_image[0]) == type('str'):
    #         del self.masked_image[0]
    #     self.get_mask_image = True

    def resize_img(self, imgs, infos, seg, rsl, cns):
        for img in imgs:
            print('img shape = {}'.format(img.shape))
        assert len(imgs) == len(infos)
        corner_center = np.average(cns, axis=0)
        bound = [np.min(cns, axis=0), np.max(cns, axis=0)]
        scale_factor = min(np.min(np.asarray(rsl, dtype=np.float32) / \
                              np.asarray(bound[1] - bound[0], dtype=np.float32)), 1.0)
        print('cns = {}, bound = {}, scale_factor = {}'.format(cns, bound, scale_factor))
        resized_imgs = []
        resized_infos = []
        for img, info in zip(imgs, infos):
            resized_info = copy.deepcopy(info)
            img_shape = list(img.shape)
            img_shape[:2] = (rsl[1], rsl[0])
            resized_img = np.zeros(img_shape, dtype=img.flatten()[0].dtype)
            resized_seg = np.zeros((rsl[1], rsl[0]), np.uint8)
            bounded_img = img[bound[0][1]:bound[1][1], bound[0][0]:bound[1][0]]
            print('bounded_img shape = {}'.format(bounded_img.shape))
            size = np.array([x for x in reversed(bounded_img.shape[:2])], dtype=np.int32)
            if scale_factor < 1.0:
                size = np.array(np.array(size, dtype=np.float32) * scale_factor, dtype=np.int32)
                bounded_img = cv2.resize(bounded_img, size)
                seg = cv2.resize(seg, size)
            half_size = size / 2
            resized_center = np.array(rsl) / 2
            lu_px_pos = np.array([max(x, 0) for x in (resized_center - half_size)], dtype=np.int32)
            rd_px_pos = lu_px_pos + size
            print('lu_px_pos = {}, rd_px_pos = {}, size = {}, resized_center = {}, half_size = {}'.format(lu_px_pos, rd_px_pos, size, resized_center, half_size))
            resized_img[lu_px_pos[1]:rd_px_pos[1], lu_px_pos[0]:rd_px_pos[0]] = bounded_img
            resized_seg[lu_px_pos[1]:rd_px_pos[1], lu_px_pos[0]:rd_px_pos[0]] = seg

            f, c= np.array([info.k[0], info.k[4]]), np.array([info.k[2], info.k[5]])
            # lu_px_pos_in_origin = corner_center - resized_center / scale_factor
            # c -= lu_px_pos_in_origin * scale_factor
            c = (c - corner_center) * scale_factor + resized_center
            f *= scale_factor
            resized_info.k[0] = f[0]
            resized_info.k[4] = f[1]
            resized_info.k[2] = c[0]
            resized_info.k[5] = c[1]

            resized_info.p[0] = f[0]
            resized_info.p[5] = f[1]
            resized_info.p[2] = c[0]
            resized_info.p[6] = c[1]

            resized_imgs.append(resized_img)
            resized_infos.append(resized_info)

        return resized_imgs, resized_infos, resized_seg




    def image_masking_cb(self, req, res):
        if req.create_depth_mask and not self.depth_img is None:
            self.img_masker.set_empty_depth_img(self.depth_img, req.mask_id)
            res.success = True
            return res

        check_none = lambda val: any([x is None for x in val])
        prepared = not check_none(
            [self.rgb_info_msg, self.depth_info_msg, self.rgb_img, self.depth_img]
        )
        if not prepared:
            res.success = False
            print('img not prepared')
            return res

        # masked_image = ArucoMaskImage(self.rgb_img, self.depth_img, [req.mask_id], 
        #                               self.aruco_dictionary, self.aruco_parameters)
        # corners = masked_image.get_mask_corners(req.mask_id)
        # if corners is None:
        #     res.success = False
        #     print('corners is none')
        #     return res
        
        rgb_img, depth_img, seg_mask_uint8, corners = self.img_masker.get_masked_img(self.rgb_img, self.depth_img, req.mask_id)

        if check_none([rgb_img, depth_img, corners]):
            res.success = False
            return res

        imgs, infos, seg_mask_uint8 = self.resize_img(
            [rgb_img, depth_img], 
            [self.rgb_info_msg, self.depth_info_msg], 
            seg_mask_uint8,
            req.resolution, 
            corners
        )
        res.rgb_camera_info = infos[0]
        res.depth_camera_info = infos[1]
        plt.imshow(imgs[0], vmin=0, vmax=255)
        plt.show()
        plt.imshow(imgs[1], cmap='gray', vmin=0, vmax=2000)
        plt.show()
        res.rgb_img = self.bridge.cv2_to_imgmsg(imgs[0], encoding="rgb8")
        res.depth_img = self.bridge.cv2_to_imgmsg(imgs[1], encoding="mono16")
        res.segmask = self.bridge.cv2_to_imgmsg(seg_mask_uint8, encoding="mono8")
        res.success = True
        return res

def main():
    rclpy.init()
    node = MaskColorAndDepthImage()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
