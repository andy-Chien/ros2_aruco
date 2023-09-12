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
# from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import numpy as np
import cv2
import quaternion as qtn

from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
from ros2_aruco_interfaces.msg import ArucoMarkers
from ros2_aruco_interfaces.srv import GetMaskImage
from math import *
import copy
import open3d as o3d
import trimesh

DEPTH_THRESHOLD = 8
DRAW_MARKER_POSE = False
IM_SHOW = False
IM_PUBLISH = False
ID_LIST = [5, 15]
PICKING_BOX_ID = 5

class ArucoImageMasker:
    def __init__(self, ad, ap):
        self.aruco_dictionary = ad
        self.aruco_parameters = ap
        # self.id_list = aruco_id_list

        self.masked_rgb = dict()
        self.masked_depth = dict()
        self.mask_corners = dict()
        self.marker_corners = dict()
        self.rgb_masks = dict()
        self.depth_masks = dict()
        self.empty_depth_img = dict()
        self.corners_depth = dict()
        mesh_file = "/home/andy/packing_ws/install/mr_description/share/mr_description/meshes/black_box_bigger_2.stl"
        # box_mesh = o3d.io.read_triangle_mesh(mesh_file)
        self.box_mesh_trimesh = trimesh.load_mesh(mesh_file)
        # print(self.box_mesh_trimesh.bounds)
        # print(self.box_mesh_trimesh.center_mass)
        # print(self.box_mesh_trimesh.centroid)
        # o3d.visualization.draw_geometries([box_mesh], window_name='Open3D', width=1920, height=1080, \
        #     left=50, top=50, point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)
        # vis = o3d.visualization.Visualizer()
        # Create a window, name it and scale it
        # vis.create_window(window_name='Bunny Visualize', width=800, height=600)

        # # Add the voxel grid to the visualizer
        # vis.add_geometry(box_mesh)

        # # We run the visualizater
        # vis.run()
        # # Once the visualizer is closed destroy the window and clean up
        # vis.destroy_window()
        # self.box_voxel = o3d.geometry.VoxelGrid.create_from_triangle_mesh(box_mesh, 0.001)
        # o3d.visualization.draw_geometries([self.box_voxel], window_name='Open3D', width=1920, height=1080, \
        #     left=50, top=50, point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)
        # voxel_list = self.box_voxel.get_voxels()
        # voxel_list = [np.array(voxel.grid_index, dtype=np.uint32) for voxel in voxel_list]
        # box_indx_size = np.max(voxel_list, axis=0) + 1
        # self.box_indx_size = box_indx_size
        # self.box_voxel_bool = [[[False] * box_indx_size[2]] * box_indx_size[1]] * box_indx_size[0]
        # # print('self.box_voxel.origin = {} \n self.box_voxel_bool = {}'.format(self.box_voxel.origin, self.box_voxel_bool))
        # for idx in voxel_list:
        #     self.box_voxel_bool[idx[0]][idx[1]][idx[2]] = True
        
        # print('self.box_voxel.origin = {} \n self.box_voxel_bool = {}'.format(self.box_voxel.origin, self.box_voxel_bool))


    def set_empty_depth_img(self, depth_img, id):
        self.empty_depth_img[id] = copy.deepcopy(depth_img)

    def check_and_update_mask(self, rgb_img=None, depth_img=None, id=1, force_update=False):
        assert rgb_img.shape[:2] == depth_img.shape[:2]
        if (id not in self.mask_corners or force_update) \
           and (depth_img is None or not self.find_mask_corners(rgb_img, id)):
            return False
        if force_update or id not in self.rgb_masks or id not in self.depth_masks:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('self.rgb_masks = {}, self.mask_corners = {}'.format(self.rgb_masks, self.mask_corners))
            self.rgb_masks[id] = self.generate_mask(self.mask_corners[id], rgb_img, np.uint8)
            self.depth_masks[id] = self.generate_mask(self.mask_corners[id], depth_img, np.uint16)
            self.corners_depth[id] = []
            for corner in self.mask_corners[id]:
                self.corners_depth[id].append(float(depth_img[corner[1], corner[0]]) / 1000.0)
        return True

    def get_masked_img(self, marker_size, intrinsic_mat, distortion, rgb_img=None, depth_img=None, id=1, force_update=False, get_seg_mask=False):
        assert rgb_img.shape[:2] == depth_img.shape[:2]
        self.marker_size = marker_size
        self.intrinsic_mat = intrinsic_mat
        self.distortion = distortion
        
        if not self.check_and_update_mask(rgb_img, depth_img, id, force_update):
            return None

        rgb_mask = self.rgb_masks[id]
        depth_mask = self.depth_masks[id]
        # if id in self.empty_depth_img:
            # rgb_seg_mask, depth_seg_mask, seg_mask_uint8 = self.compute_segmentation_mask(
            #     depth_img, self.empty_depth_img[id], self.mask_corners[id])
        if get_seg_mask and id in self.mask_corners and id in self.marker_corners:
            seg_mask_uint8 = self.compute_uint8_segmentation_mask(
                depth_img, self.mask_corners[id], self.marker_corners[id])
        else:
            bound = [np.min(self.mask_corners[id], axis=0), 
                     np.max(self.mask_corners[id], axis=0)]
            size = [bound[1][1] - bound[0][1], bound[1][0] - bound[0][0]]
            seg_mask_uint8 = 255 * np.ones(size, dtype=np.uint8)
    
        masked_rgb = np.bitwise_and(rgb_img, rgb_mask)
        masked_depth = np.bitwise_and(depth_img, depth_mask)
        return masked_rgb, masked_depth, seg_mask_uint8
    
    def get_corners(self, id):
        if id not in self.mask_corners or id not in self.marker_corners or id not in self.corners_depth:
            return None
        return self.mask_corners[id], self.marker_corners[id], self.corners_depth[id]
    
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
        for i, (sub_row, empty_row) in enumerate(zip(sub_img, bounded_empty_img)):
            for j, (sub_x, empty_x) in enumerate(zip(sub_row, empty_row)):
                if sub_x > DEPTH_THRESHOLD or (empty_x < DEPTH_THRESHOLD and sub_x < -1* DEPTH_THRESHOLD):
                    bounded_rgb_seg_mask[i][j] = rgb_mask_color
                    bounded_depth_seg_mask[i][j] = depth_mask_color
                    seg_mask_uint8[i][j] = np.iinfo(np.uint8).max
        kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (DEPTH_THRESHOLD+5, DEPTH_THRESHOLD+5))
        kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (DEPTH_THRESHOLD, DEPTH_THRESHOLD))
        seg_mask_uint8 = cv2.dilate(cv2.erode(seg_mask_uint8, kernel_1), kernel_2)
        
        rgb_seg_mask[bound[0][1]:bound[1][1], bound[0][0]:bound[1][0]] = bounded_rgb_seg_mask
        depth_seg_mask[bound[0][1]:bound[1][1], bound[0][0]:bound[1][0]] = bounded_depth_seg_mask
        return rgb_seg_mask, depth_seg_mask, seg_mask_uint8
    
    def compute_uint8_segmentation_mask(self, img, box_corners, marker_corners):
        marker_poses = self.compute_marker_poses(marker_corners)
        box_pos, box_q = self.compute_box_pose(marker_poses)
        c_x = self.intrinsic_mat[0][2]
        c_y = self.intrinsic_mat[1][2]
        f_x = self.intrinsic_mat[0][0]
        f_y = self.intrinsic_mat[1][1]
        bound = [np.min(box_corners, axis=0), np.max(box_corners, axis=0)]
        bounded_img = img[bound[0][1]:bound[1][1], bound[0][0]:bound[1][0]]
        mat = np.identity(4)
        mat[:3, :3] = qtn.as_rotation_matrix(box_q)
        mat[:3, 3] = box_pos
        mat[2, 3] -= 0.01
        mat = np.linalg.inv(mat)
        seg_mask_uint8 = np.zeros(bounded_img.shape, dtype=np.uint8)
        pos_list = []
        indx_list = []
        for i, row in enumerate(bounded_img):
            for j, z in enumerate(row):
                if z < DEPTH_THRESHOLD or z > 1100:
                    continue
                pixel_x = j + bound[0][0]
                pixel_y = i + bound[0][1]
                pos = np.array([0., 0., 0., 1.])
                pos[0] = (pixel_x - c_x) * (float(z) / 1000.0) / f_x
                pos[1] = (pixel_y - c_y) * (float(z) / 1000.0) / f_y
                pos[2] = float(z + DEPTH_THRESHOLD) / 1000.0
                pos = np.matmul(mat, pos)
                pos_list.append(pos[:3])
                indx_list.append([i, j])
                # indx = self.box_voxel.get_voxel(pos.reshape((3,1)))
                # if np.any(indx + 2 > self.box_indx_size) or \
                #     np.any(indx + 2 < 0):
                #     # print(pos, indx, self.box_indx_size)
                #     continue
                # # print('dfsdfdfdfdfdf')
                # if not self.box_voxel_bool[indx[0]][indx[1]][indx[2]]:
                #     seg_mask_uint8[i][j] = np.iinfo(np.uint8).max
        
        results = self.box_mesh_trimesh.contains(pos_list)
        for indx, result in zip(indx_list, results):
            if result: continue
            seg_mask_uint8[indx[0]][indx[1]] = np.iinfo(np.uint8).max

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (DEPTH_THRESHOLD, DEPTH_THRESHOLD))
        seg_mask_uint8 = cv2.dilate(cv2.erode(seg_mask_uint8, kernel), kernel)
        return seg_mask_uint8
                
        

    def compute_box_pose(self, marker_poses):
        pos_list = []
        for p in marker_poses:
            pos = p.position
            pos_list.append(np.array([pos.x, pos.y, pos.z]))

        center_pos = np.mean(pos_list, axis=0)
        v1 = pos_list[-1][:2] - pos_list[0][:2]
        v2 = pos_list[1][:2] - pos_list[0][:2]
        v = v1 if np.linalg.norm(v1) < np.linalg.norm(v2) else v2
        q = qtn.from_rotation_vector(np.array([.0, .0, atan2(v[1], v[0])]))
        qz = qtn.quaternion(0., 1., 0., 0.)
        q *= qz

        return center_pos, q

    def find_mask_corners(self, rgb_img, target_id):
        corners, ids = self.find_aruco(rgb_img)
        if ids is None or corners is None:
            return False
        num_of_id = dict()
        for [id] in ids:
            num_of_id[id] = 0
        for [id] in ids: 
            num_of_id[id] += 1
        print('num_of_id = {}'.format(num_of_id))
        if target_id not in num_of_id or num_of_id[target_id] != 4:
            return False
        mask_corners = []
        marker_corners = []
        for i, [id] in enumerate(ids):
            if id != target_id:
                continue
            mask_corners.append([int(x) for x in corners[i][0][0]])
            marker_corners.append(corners[i])

        order = [0]
        for _ in range(len(mask_corners) - 1):
            c = mask_corners[order[-1]]
            d_min = 99999
            d_min_idx = 0
            for j, cn in enumerate(mask_corners):
                if j in order:
                    continue
                d = np.linalg.norm(np.array(c) - np.array(cn))
                if d < d_min:
                    d_min = d
                    d_min_idx = j
            order.append(d_min_idx)

        self.mask_corners[target_id] = [mask_corners[i] for i in order]
        self.marker_corners[target_id] = [marker_corners[i] for i in order]
        return True

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
    
    def compute_marker_poses(self, corners):
        if cv2.__version__ > '4.0.0':
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.intrinsic_mat, self.distortion)
        else:
            rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.intrinsic_mat, self.distortion)
            
        poses = []
        for (tv, rv) in zip(tvecs, rvecs):
            pose = Pose()
            pose.position.x = tv[0][0]
            pose.position.y = tv[0][1]
            pose.position.z = tv[0][2]

            rot_matrix = cv2.Rodrigues(np.array(rv[0]))[0]
            quat = qtn.from_rotation_matrix(rot_matrix)

            pose.orientation.x = quat.x
            pose.orientation.y = quat.y
            pose.orientation.z = quat.z
            pose.orientation.w = quat.w
            poses.append(pose)
        return poses


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
        depth_info_topic = self.get_parameter("depth_info_topic").get_parameter_value().string_value
        self.camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value
        self.armarker_info_topic = self.get_parameter("marker_info_topic").get_parameter_value().string_value

        self.corners = []
        self.marker_ids = []
        
        self.rgb_img = None
        self.depth_img = None

        self.masked_image = []
        self.get_marker = False
        self.get_mask_image = False

        self.rgb_img_header = None
        self.depth_img_header = None

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
                                                 1)
        
        self.depth_info_sub = self.create_subscription(CameraInfo,
                                                 depth_info_topic,
                                                 self.depth_info_callback,
                                                 1)

        self.img_sub = self.create_subscription(Image, image_topic,
                                 self.image_callback, 1)


        self.depth_img_sub = self.create_subscription(Image, depth_image_topic,
                                 self.depth_image_callback, 1)

        self.image_masking_service = self.create_service(GetMaskImage, "image_masking", 
                                           self.image_masking_cb)


        # Set up fields for camera parameters
        self.rgb_info_msg = None
        self.depth_info_msg = None
        self.intrinsic_mat = None
        self.distortion = None

        self.img_result = dict()
        self.corners_result = dict()
        self.marker_occupied = dict()
        for id in ID_LIST:
            self.marker_occupied[id] = False
            self.img_result[id] = None
            self.corners_result[id] = None

        self.id_list = []

        self.aruco_dictionary = cv2.aruco.Dictionary_get(dictionary_id)
        self.aruco_parameters = cv2.aruco.DetectorParameters_create()
        self.img_masker = ArucoImageMasker(self.aruco_dictionary, self.aruco_parameters)

        self.bridge = CvBridge()
        self.update_timer = None

    def start_update_timer(self):
        if self.update_timer == None:
            timer_period = 0.2
            self.update_timer = self.create_timer(timer_period, self.update_timer_cb)

    def update_timer_cb(self):
        for id in self.id_list:
            if not self.marker_occupied[id] and (self.img_result[id] is None or self.corners_result[id] is None):
                self.update_masked_img(id)

    def rgb_info_callback(self, info_msg):
        self.rgb_info_msg = info_msg
        self.intrinsic_mat = np.reshape(np.array(self.rgb_info_msg.k), (3, 3))
        self.distortion = np.array(self.rgb_info_msg.d)
        # Assume that camera parameters will remain the same...
        self.destroy_subscription(self.rgb_info_sub)
        print('rgb_info_callback')


    def depth_info_callback(self, info_msg):
        self.depth_info_msg = info_msg
        self.destroy_subscription(self.depth_info_sub)
        print('depth_info_callback')

    def image_callback(self, img_msg):
        if self.rgb_img is None:
            print('image_callback')
        self.rgb_img_header = img_msg.header
        self.rgb_img = self.bridge.imgmsg_to_cv2(img_msg,
            desired_encoding='bgr8')
        
    def depth_image_callback(self, depth_img_msg):
        if self.depth_img is None:
            print('depth_image_callback')
        self.depth_img_header = depth_img_msg.header
        self.depth_img = self.bridge.imgmsg_to_cv2(depth_img_msg,
            desired_encoding='16UC1')

    def compute_marker_poses(self, corners):
        if cv2.__version__ > '4.0.0':
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.intrinsic_mat, self.distortion)
        else:
            rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.intrinsic_mat, self.distortion)
            
        poses = []
        for (tv, rv) in zip(tvecs, rvecs):
            pose = Pose()
            pose.position.x = tv[0][0]
            pose.position.y = tv[0][1]
            pose.position.z = tv[0][2]

            rot_matrix = cv2.Rodrigues(np.array(rv[0]))[0]
            quat = qtn.from_rotation_matrix(rot_matrix)

            pose.orientation.x = quat.x
            pose.orientation.y = quat.y
            pose.orientation.z = quat.z
            pose.orientation.w = quat.w
            poses.append(pose)
        return poses, rvecs, tvecs

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
    
    def update_masked_img(self, id):
        img_result = self.img_masker.get_masked_img(
            self.marker_size, self.intrinsic_mat, self.distortion,
            self.rgb_img, self.depth_img, id, False, get_seg_mask=False)

        corners_result = self.img_masker.get_corners(id)
        if corners_result is None:
            print('corners_result is None')
            return
        depth_img = img_result[1]
        mask_corners, _, corners_depth = corners_result
        bound = [np.min(mask_corners, axis=0), np.max(mask_corners, axis=0)]
        max_z = min(corners_depth) * 1000.0
        img = depth_img[bound[0][1]:bound[1][1], bound[0][0]:bound[1][0]]
        for row in img:
            for x in row:
                if 20 < x < max_z - 100:
                    print('something higher')
                    return
        get_seg_mask = id == PICKING_BOX_ID
        img_result = self.img_masker.get_masked_img(
            self.marker_size, self.intrinsic_mat, self.distortion,
            self.rgb_img, self.depth_img, id, True, get_seg_mask=get_seg_mask)
        if img_result is None:
            print('img_result is None!')
            return
        print('Got img result')
        self.img_result[id] = img_result
        self.corners_result[id] = self.img_masker.get_corners(id)
        return
        
    def image_masking_cb(self, req, res):
        check_none = lambda val: any([x is None for x in val])
        prepared = not check_none(
            [self.rgb_info_msg, self.depth_info_msg, self.rgb_img, self.depth_img]
        )
        if not prepared:
            res.success = False
            print('img not prepared')
            return res
        
        rq = GetMaskImage.Request

        print('req.mode = {}, req.mode & rq.GET_MASKED_IMG = {}, not req.mode & rq.GET_MASKED_IMG = {}'.format(
            req.mode, req.mode & rq.GET_MASKED_IMG, not req.mode & rq.GET_MASKED_IMG))

        res.success = True
        if req.mode & rq.CREATE_DEPTH_MASK:
            self.img_masker.set_empty_depth_img(self.depth_img, req.mask_id)

        if req.mode & rq.UPDATE_MARK_MASK:
            if req.mask_id not in self.id_list:
                self.id_list.append(req.mask_id)
            res.success = self.img_masker.check_and_update_mask(
                self.rgb_img, self.depth_img, req.mask_id, True)
            corners_result = self.img_masker.get_corners(req.mask_id)
            if corners_result is None:
                res.success = False
            _, marker_corners, corners_depth = corners_result
            poses, _, _ = self.compute_marker_poses(marker_corners)
            res.marker_poses = poses
            res.corners_depth = corners_depth
            res.poses_frame = self.rgb_img_header.frame_id


        if req.mode & rq.START_UPDATE_TIMER:
            if req.mask_id not in self.id_list:
                self.id_list.append(req.mask_id)
            self.start_update_timer()
        
        if req.mode & rq.STOP_UPDATE_TIMER:
            self.destroy_timer(self.update_timer)
            self.update_timer = None

        if req.mode & rq.MARK_RELEASE:
            self.marker_occupied[req.mask_id] = False

        if not req.mode & rq.GET_MASKED_IMG:
            self.img_result[id] = None
            self.corners_result[id] = None
            return res
        
        if self.img_result[req.mask_id] is None or self.corners_result[req.mask_id] is None:
            res.success = False
            return res
        
        rgb_img, depth_img, seg_mask_uint8 = copy.deepcopy(self.img_result[req.mask_id])
        mask_corners, marker_corners, corners_depth = copy.deepcopy(self.corners_result[req.mask_id])

        self.marker_occupied[req.mask_id] = True
        self.img_result[req.mask_id] = None
        self.corners_result[req.mask_id] = None

        imgs, infos, seg_mask_uint8 = self.resize_img(
            [rgb_img, depth_img], 
            [self.rgb_info_msg, self.depth_info_msg], 
            seg_mask_uint8,
            req.resolution, 
            mask_corners
        )
        poses, rvecs, tvecs = self.compute_marker_poses(marker_corners)

        if DRAW_MARKER_POSE:
            for (tv, rv) in zip(tvecs, rvecs):
                rgb_img = cv2.aruco.drawAxis(
                    rgb_img, self.intrinsic_mat, self.distortion, tv, rv, 0.02)
            if IM_SHOW:
                cv2.imshow('QueryImage', self.cv_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        res.rgb_camera_info = infos[0]
        res.depth_camera_info = infos[1]
        res.rgb_img = self.bridge.cv2_to_imgmsg(imgs[0], encoding="rgb8")
        res.depth_img = self.bridge.cv2_to_imgmsg(imgs[1], encoding="mono16")
        res.segmask = self.bridge.cv2_to_imgmsg(seg_mask_uint8, encoding="mono8")
        res.rgb_img.header = self.rgb_img_header
        res.depth_img.header = self.depth_img_header
        res.marker_poses = poses
        res.poses_frame = self.rgb_img_header.frame_id
        res.corners_depth = corners_depth
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
