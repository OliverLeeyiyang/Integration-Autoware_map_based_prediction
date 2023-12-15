# Generate 39 by 39 image for the prediction model

# System and Projects imports
from typing import List
from typing import Tuple
import numpy as np
from sensor_msgs.msg import Image as ImageMsg
from cv_bridge import CvBridge
from PIL import Image
from PIL import ImageDraw

# Autoware auto msgs
from autoware_auto_perception_msgs.msg import TrackedObject

# From lanelet2 imports
from lanelet2.core import ConstLanelet

# Local imports
from tum_prediction.utils_tier4 import Tier4Utils

# Color values
GREY = -1
WHITE = 0
BLACK = 1


class LaneletPolygonData:
    def __init__(self):
        self.polygon: List # Polygon data of the lanelet
        self.lanelet_yaw: float # Yaw of the lanelet


class LaneletsInfoData:
    def __init__(self):
        self.possible_lanelets: List = []
        self.surrounding_lanelets: List = []
        self.current_lanelets: List = []
        self.possible_ids: List = []
        self.surrounding_ids: List = []


class ImageInfoData:
    def __init__(self, color, object_pose_2d, rotate_angle, search_dist):
        self.color: int = color
        self.object_pose: Tuple = object_pose_2d
        self.rotate_angle: float = rotate_angle
        self.search_dist: float = search_dist



class PILImagePathGenerator:
    def __init__(self, image_size_=39):
        self.image_size = image_size_
        self.lanelets_data = {} # Key: lanelet.id, Value: LaneletPolygonData
        self.tu = Tier4Utils()

        self.image_second_origin = (self.image_size / 2, self.image_size)
        self.last_lanelet_yaw = 0.0
    
    def generateImageForOffLaneObject(self, _off_lane_object: TrackedObject) -> ImageMsg:
        img = Image.new("RGB", (self.image_size, self.image_size), "white")
        # return self.create_sensor_msg_from_pil_image(img)
        return self.create_mapInView_from_pil_image(img)
    
    def generateImageForOnLaneObject(
        self,
        Tobject: TrackedObject,
        search_dist,
        lanelets_info_data: LaneletsInfoData
    ) -> ImageMsg:
        lanelets_data = {} # Key: lanelet.id, Value: LaneletPolygonData
        
        # Step1. Compute image data
        object_orientation = Tobject.kinematics.pose_with_covariance.pose.orientation
        object_pose = Tobject.kinematics.pose_with_covariance.pose
        pose_2d = (object_pose.position.x, object_pose.position.y)
        object_yaw = self.tu.getYawFromQuaternion(object_orientation)
        lanelet_yaw = self.get_crucial_lanelet_yaw(lanelets_info_data.current_lanelets, object_pose, object_yaw)
        rotate_angle = -np.pi / 2 - lanelet_yaw
        grey_image = ImageInfoData(GREY, pose_2d, rotate_angle, search_dist)
        black_image = ImageInfoData(BLACK, pose_2d, rotate_angle, search_dist)

        # Step2. Generate polygon areas from the lanelets
        not_possible_lanelets = self.get_not_possible_lanelets(lanelets_info_data)
        self.generatePolygonAreas(not_possible_lanelets, lanelets_data, grey_image)
        self.generatePolygonAreas(lanelets_info_data.possible_lanelets, lanelets_data, black_image)

        # Step3. Create an image with white background and filled with lanelet polygons
        img = Image.new("RGB", (self.image_size, self.image_size), "white")
        filled_image = self.generateImageMap(img, lanelets_data)

        image_msg = self.create_sensor_msg_from_pil_image(filled_image)
        return image_msg
    
    def generateMapInViewForOnLaneObject(
        self,
        Tobject: TrackedObject,
        search_dist,
        lanelets_info_data: LaneletsInfoData
    ):
        lanelets_data = {} # Key: lanelet.id, Value: LaneletPolygonData
        
        # Step1. Compute image data
        object_orientation = Tobject.kinematics.pose_with_covariance.pose.orientation
        object_pose = Tobject.kinematics.pose_with_covariance.pose
        pose_2d = (object_pose.position.x, object_pose.position.y)
        object_yaw = self.tu.getYawFromQuaternion(object_orientation)
        lanelet_yaw = self.get_crucial_lanelet_yaw(lanelets_info_data.current_lanelets, object_pose, object_yaw)
        rotate_angle = -np.pi / 2 - lanelet_yaw
        grey_image = ImageInfoData(GREY, pose_2d, rotate_angle, search_dist)
        black_image = ImageInfoData(WHITE, pose_2d, rotate_angle, search_dist)

        # Step2. Generate polygon areas from the lanelets
        not_possible_lanelets = self.get_not_possible_lanelets(lanelets_info_data)
        self.generatePolygonAreas(not_possible_lanelets, lanelets_data, grey_image)
        self.generatePolygonAreas(lanelets_info_data.possible_lanelets, lanelets_data, black_image)

        # Step3. Create an image with white background and filled with lanelet polygons, and draw in Greyscale
        img = Image.new("L", (self.image_size, self.image_size), "black")
        filled_image = self.generateImageMap(img, lanelets_data)

        mapInView = self.create_mapInView_from_pil_image(filled_image)
        return mapInView
    
    def generateImageMap(self, img, lanelets_data):
        for poly_data in lanelets_data.values():
            color = poly_data.lanelet_state
            geom = poly_data.polygon
            img1 = ImageDraw.Draw(img)
            img1.polygon(geom, fill =color, outline =color)
        
        return img
    
    def generatePolygonAreas(self, lanelets, lanelets_data, image_data: ImageInfoData):
        for a_lanelet in lanelets:
            lanID = a_lanelet.id

            polygonData = LaneletPolygonData()
            polygonData.lanelet_state = self.set_lanelet_fill_color(image_data.color)
            # When there are too many lanelets, could use this to reduce the computation time
            #if not self.checkIfPolygonInImage(a_lanelet, image_data) and (image_data.color == GREY):
            #    continue
            polygonData.polygon = self.convert2Polygon(a_lanelet, image_data)
            lanelets_data[lanID] = polygonData
    
    def convert2Polygon(self, a_lanelet: ConstLanelet, image_data: ImageInfoData) -> List:
        points = []
        for p in a_lanelet.leftBound:
            points.append(self.remap_point((p.x, p.y), image_data))
        temp_list = []
        for p in a_lanelet.rightBound:
            temp_list.append(self.remap_point((p.x, p.y), image_data))
        temp_list.reverse()

        points.extend(temp_list)
        points.append(points[0])

        return points
    
    def remap_point(self, point: Tuple, image_data: ImageInfoData) -> Tuple:
        '''Remap the point to the image.

        Map coordinate: x axis is to the left, y axis is to the bottom

        Image coordinate: x axis is to the right, y axis is to the bottom
        
        Rotation: 
        ----------------
        rotate_angle: -np.pi / 2 - lanelet_yaw
        (-np.pi / 2 means we want the main lanelet to be vertical in the image, which is -y axis in the map coordinate.
         also, we need to rotate the point back by lanelet_yaw, so the rotate_angle is -np.pi / 2 - lanelet_yaw)

        Translation:
        ----------------
        rescale_factor: (self.image_size / 2) / image_data.search_dist

        Also reverse the x axis.
        '''
        x1 = point[0]
        y1 = point[1]
        x0 = image_data.object_pose[0]
        y0 = image_data.object_pose[1]
        ra = image_data.rotate_angle
        rotated_point = ((x1 - x0) * np.cos(ra) - (y1 - y0) * np.sin(ra) + x0,(x1 - x0) * np.sin(ra) + (y1 - y0) * np.cos(ra) + y0)

        move_dist_in_image = (self.image_size / 2) / image_data.search_dist
        x2 = rotated_point[0]
        y2 = rotated_point[1]
        final_point = (self.image_second_origin[0] - (x2 - x0) * move_dist_in_image, self.image_second_origin[1] + (y2 - y0) * move_dist_in_image) # X axis is reversed in the image

        return final_point
    
    def get_not_possible_lanelets(self, lanelets_info_data: LaneletsInfoData) -> List:
        common_ids = set(lanelets_info_data.possible_ids).intersection(lanelets_info_data.surrounding_ids)
        not_possible_lanelets = [
            lanelet for lanelet in lanelets_info_data.surrounding_lanelets if lanelet.id not in common_ids
        ]
        return not_possible_lanelets
    
    def get_crucial_lanelet_yaw(self, current_lanelets, object_pose, object_yaw) -> float:
        # Choose the lanelet with the smallest yaw difference with the object
        if len(current_lanelets) == 1:
            lanelet_yaw = self.tu.getLaneletAngle(current_lanelets[0], object_pose.position)
        else:
            yaw_list = []
            for lanelet in current_lanelets:
                lane_yaw = self.tu.getLaneletAngle(lanelet, object_pose.position)
                yaw_list.append(lane_yaw - object_yaw)
            abs_yaw_list = [abs(yaw) for yaw in yaw_list]
            min_index = abs_yaw_list.index(min(abs_yaw_list))
            lanelet_yaw = yaw_list[min_index] + object_yaw
        
        # To avoid the sudden change of the yaw
        if self.last_lanelet_yaw == 0.0:
            self.last_lanelet_yaw = lanelet_yaw
        else:
            if abs(object_yaw - lanelet_yaw) > abs(object_yaw - self.last_lanelet_yaw):
                lanelet_yaw = self.last_lanelet_yaw
            else:
                self.last_lanelet_yaw = lanelet_yaw

        return lanelet_yaw
    
    def create_sensor_msg_from_pil_image(self, pil_image):
        image_np = np.array(pil_image)

        # Create a ROS 2 Image message
        bridge = CvBridge()
        image_msg = bridge.cv2_to_imgmsg(image_np, encoding="rgb8")

        return image_msg
    
    def create_mapInView_from_pil_image(self, pil_image):
        '''this is the local view map of the ego vehicle with.
        0.0 (white) if this is not a road
        0.5 (grey)  if the position is a road
        1.0 (black) if the position is reachable(driveable) from the current lanelet.
        '''
        image_np = np.array(pil_image)
        mapInView = image_np / 255.0
        mask_half = mapInView == (128 / 255.0)
        mapInView[mask_half] = 0.5

        return mapInView
    
    def set_lanelet_fill_color(self, color_value: int) -> str:
        color = ""
        if color_value == 1:
            color = "black"
        elif color_value == -1:
            color = "grey"
        else:
            color = "white"
        
        return color
    
    # Unused functions
    def check_lanelet_image(self, ID: int, lanelet_layer):
        """Display the lanelet on the center of the image."""
        a_lanelet = lanelet_layer[ID] # This can find the lanelet with the ID and map.laneletLayer
        centerline = a_lanelet.centerline
        center_point_index = int((len(centerline) - 1) / 2)
        center_point = centerline[center_point_index]
        pose_2d = (center_point.x, center_point.y)
        search_dist = self.tu.getLaneletLength3d(a_lanelet) / 2 + 15
        
        self.image_size = self.image_size*10
        self.image_second_origin = (self.image_size / 2, self.image_size / 2)

        grey_image = ImageInfoData(GREY, pose_2d, 0.0, search_dist)
        lanelets_data = {}
        self.generatePolygonAreas([a_lanelet], lanelets_data, grey_image)
        img = Image.new("RGB", (self.image_size, self.image_size), "white")
        filled_image = self.generateImageMap(img, lanelets_data)
        filled_image.show()
    
    def checkIfPolygonInImage(self, a_lanelet: ConstLanelet, image_data: ImageInfoData) -> bool:
        '''Check if the polygon is still in the image area.'''
        points = [a_lanelet.leftBound[0], a_lanelet.leftBound[-1], a_lanelet.rightBound[-1], a_lanelet.rightBound[0]]
        remapped_points = [self.remap_point((p.x, p.y), image_data) for p in points]
        for rp in remapped_points:
            if rp[0] >= 0 and rp[0] < self.image_size and rp[1] >= 0 and rp[1] <= self.image_size:
                return True
        return False

