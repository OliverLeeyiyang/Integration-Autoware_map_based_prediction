# System and Projects imports
import time

# New data types
from typing import List

# Autoware auto msgs
from autoware_auto_perception_msgs.msg import TrackedObject
import geometry_msgs.msg as gmsgs

# From lanelet2 imports
from lanelet2.core import BasicPoint2d
from lanelet2.core import Lanelet
import lanelet2.geometry as l2_geom
import numpy as np
from sensor_msgs.msg import Image as ImageMsg

# Local imports
from tum_prediction.utils_tier4 import Tier4Utils

PosePath = List[gmsgs.Pose]


class LaneletData:
    def __init__(self):
        self.lanelet: Lanelet()
        self.probability: float


LaneletsData = List[LaneletData]


def calcoffsetpose(position, yaw, x, y, z):
    """Only use numpy to calculate the offset pose."""
    c_yaw = np.cos(yaw)
    s_yaw = np.sin(yaw)
    transform_q_matrix = np.array(
        [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], dtype=np.float64
    )
    tr = position
    pose_matrix = np.array(
        [[c_yaw, -s_yaw, 0, tr[0]], [s_yaw, c_yaw, 0, tr[1]], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    offset_position = np.dot(pose_matrix, transform_q_matrix)

    return offset_position[:2, 3]


class ImagePathGenerator:
    """Generates an image (size: 39 x 39) for prediction algorithm based on NN."""

    def __init__(self, image_size_=39):
        self.image_size = image_size_

        self.gmg = GridMapGenerator()
        self.tu = Tier4Utils()

    def generateImageForOffLaneObject(self, off_lane_object: TrackedObject) -> ImageMsg:
        """Generate an image for off-lane object."""
        matrix = self._generateMatrixForOffLaneObject(off_lane_object)

        return self.gmg.matrix_to_gridmap(matrix)

    def generateImageForOnLaneObject(
        self,
        Tobject: TrackedObject,
        search_dist,
        possible_lanalets,
        surrounding_lanelets,
        current_lanelets,
        possible_ids,
        surrounding_ids,
    ) -> ImageMsg:
        """Generate an image for on-lane object."""
        matrix = self._generateMatrixForOnLaneObject(
            Tobject,
            search_dist,
            possible_lanalets,
            surrounding_lanelets,
            current_lanelets,
            possible_ids,
            surrounding_ids,
        )

        return self.gmg.matrix_to_gridmap(matrix)

    def _generateMatrixForOffLaneObject(self, _object: TrackedObject) -> np.ndarray:
        """Generate a state matrix for off-lane object."""
        return np.zeros((self.image_size, self.image_size))

    def _generateMatrixForOnLaneObject(
        self,
        Tobject: TrackedObject,
        search_dist,
        possible_lanalets,
        surrounding_lanelets,
        current_lanelets,
        possible_ids,
        surrounding_ids,
    ) -> np.ndarray:
        """Generate a state matrix for on-lane object.

        ----------------
        Currently only for Lane_Follow situation. No Lane_Change situation.
        """
        start_time = time.time()
        object_position = Tobject.kinematics.pose_with_covariance.pose.position
        object_orientation = Tobject.kinematics.pose_with_covariance.pose.orientation
        object_pose = Tobject.kinematics.pose_with_covariance.pose
        # obj_vel = abs(Tobject.kinematics.twist_with_covariance.twist.linear.x)
        # search_dist = 1.4 * self.prediction_time_horizon_ * obj_vel
        # print('[gmg] search_dist: ', search_dist)

        # Step1: generate possible points matrix from object pose and search distance
        object_position_2d = np.array([object_position.x, object_position.y])
        object_yaw = self.tu.getYawFromQuaternion(object_orientation)

        # Get the crucial lanelet yaw from current lanelets
        lanelet_yaw = self.get_possible_lanelet_yaw(current_lanelets, object_pose, object_yaw)

        # Step2: generate state matrix from possible points matrix

        # use lanelet yaw to generate possible points matrix
        # common_ids = set(possible_ids).intersection(surrounding_ids)
        # not_possible_lanelets = [lanelet for lanelet in surrounding_lanelets if lanelet.id not in common_ids]
        # point_state_matrix = self.generate_points_state_matrix(object_position_2d, lanelet_yaw, search_dist, self.image_size, possible_lanalets, not_possible_lanelets)

        points_matrix = self.generate_points_matrix(
            object_position_2d, lanelet_yaw, search_dist, self.image_size
        )
        # point_state_matrix = self.get_point_state_matrix(points_matrix, possible_lanalets, surrounding_lanelets, possible_ids, surrounding_ids)
        # grid_state_matrix = self.get_grid_state_matrix(point_state_matrix)
        grid_state_matrix = self.generate_grid_state_matrix(
            points_matrix, possible_lanalets, surrounding_lanelets, possible_ids, surrounding_ids
        )

        end_time = time.time()
        print("[gmg] time generateMatrixForOnLaneObject: ", end_time - start_time)

        return grid_state_matrix

    def get_possible_lanelet_yaw(self, current_lanelets, object_pose, object_yaw):
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

        return lanelet_yaw

    def get_point_state(self, point: np.ndarray, possible_lanalets, not_possible_lanelets) -> int:
        # 4e-5s
        """Get the state of a point.

        ----------------

        Inputs:
        -------
        point: 2D point np.array([x, y])

        possible_lanalets: lanelets that are in same direction with the object

        not_possible_lanelets: lanelets that are in opposite direction with the object

        Output:
        -------
        state:

        0: white, out of lanelet area

        1: black, on lane and lanelet has same direction as object

        -1: grey, on lane and lanelet has opposite direction as object
        """
        state = 0
        for lanelet in possible_lanalets:
            if l2_geom.inside(lanelet, BasicPoint2d(point[0], point[1])):
                state = 1
                return state
        for lanelet in not_possible_lanelets:
            if l2_geom.inside(lanelet, BasicPoint2d(point[0], point[1])):
                state = -1
                return state

        return state

    # Test later by image
    def generate_points_matrix(
        self, object_position_2d, lanelet_yaw: float, search_dist: float, image_size: int
    ) -> np.ndarray:
        # 0.016s
        """Generate a points matrix.

        ----------------
        TODO: reduce the number of points in the matrix. 1600 is too much.

        Inputs:
        -------
        object_position_2d: 2D position of the object shape (2, 1)

        lanelet_yaw: yaw of the lanelet

        search_dist: search distance from the object

        Output:
        -------
        points_matrix: size (image_size + 1) by (image_size + 1) by 2.
            40 x 40 x 2 in this project.

        And the points are in the same direction as lanelet.
        """
        start_time = time.time()
        points = np.zeros((image_size + 1, image_size + 1, 2))
        point_dist = 1 * search_dist / image_size

        # For Planning Simulation, x point to the front, y point to the left

        # When object is at the center of the image, the points matrix is like:
        # x_offsets =  point_dist * image_size / 2 - point_dist * np.arange(image_size + 1)
        # y_offsets =  point_dist * image_size / 2 - point_dist * np.arange(image_size + 1)

        # When object is at the bottom of the image, the points matrix is like:
        x_offsets = point_dist * image_size - point_dist * np.arange(image_size + 1)
        y_offsets = point_dist * image_size / 2 - point_dist * np.arange(image_size + 1)
        for i, x_offset in enumerate(x_offsets):
            for j, y_offset in enumerate(y_offsets):
                points[i, j] = calcoffsetpose(
                    object_position_2d, lanelet_yaw, x_offset, y_offset, 0
                )

        end_time = time.time()
        print("[gmg] time generate_points_matrix: ", end_time - start_time)
        return points

    def get_point_state_matrix(
        self,
        points_matrix: np.ndarray,
        possible_lanalets,
        surrounding_lanelets,
        possible_ids,
        surrounding_ids,
    ) -> np.ndarray:
        # 0.06s
        start_time = time.time()
        # get the lanelets that are in another direction with the object
        common_ids = set(possible_ids).intersection(surrounding_ids)
        not_possible_lanelets = [
            lanelet for lanelet in surrounding_lanelets if lanelet.id not in common_ids
        ]

        # Get state for all points in the points matrix
        length = points_matrix.shape[0]
        point_state_matrix = np.zeros((length, length))
        for i in range(length):
            for j in range(length):
                point_state_matrix[i][j] = self.get_point_state(
                    points_matrix[i][j], possible_lanalets, not_possible_lanelets
                )

        end_time = time.time()
        print("[gmg] time get_point_state_matrix: ", end_time - start_time)
        return point_state_matrix

    def get_grid_state_matrix(self, point_state_matrix):
        # 0.02s
        """For every inner point, use its state to decide the state of surrounding 4 grid.

        ----------------

        0: white, all points are out of lanelet area

        1: black, at least one point on lane and lanelet has same direction as object

        -1: grey, at least one point on lane and lanelet has opposite direction as object.
        """
        s1 = time.time()
        length = point_state_matrix.shape[0]
        grid_state_matrix = np.zeros((length - 1, length - 1))

        for i in range(length - 1):
            for j in range(length - 1):
                sub_matrix = point_state_matrix[i : i + 2, j : j + 2]
                if np.any(sub_matrix == 1):
                    grid_state_matrix[i][j] = 1
                    continue
                elif np.any(sub_matrix == -1):
                    grid_state_matrix[i][j] = -1
                    continue
                else:
                    grid_state_matrix[i][j] = 0

        e1 = time.time()
        print("[gmg] time get_grid_state_matrix: ", e1 - s1)

        return grid_state_matrix

    def generate_grid_state_matrix(
        self,
        points_matrix: np.ndarray,
        possible_lanalets,
        surrounding_lanelets,
        possible_ids,
        surrounding_ids,
    ) -> np.ndarray:
        start_time = time.time()
        # get the lanelets that are in another direction with the object
        common_ids = set(possible_ids).intersection(surrounding_ids)
        not_possible_lanelets = [
            lanelet for lanelet in surrounding_lanelets if lanelet.id not in common_ids
        ]

        # Get state for all points in the points matrix
        length = points_matrix.shape[0] - 1
        grid_state_matrix = np.zeros((length, length))
        center_j = length // 2

        for i in range(1, length):  # 1-38
            for j in range(1, length):
                if (i + j) % 2 == 0 or ((i + j) % 2 == 1 and j == center_j):
                    point_state = self.get_point_state(
                        points_matrix[i][j], possible_lanalets, not_possible_lanelets
                    )
                    if i != length - 1 and j != length - 1:
                        if point_state == 1:
                            grid_state_matrix[i - 1 : i + 1, j - 1 : j + 1] = 1
                            continue
                        elif point_state == -1:
                            sub = grid_state_matrix[i - 1 : i + 1, j - 1 : j + 1]
                            sub[sub == 0] = -1
                            continue
                    elif i == length - 1 and j != length - 1:
                        if point_state == 1:
                            grid_state_matrix[i - 1, j - 1 : j + 1] = 1
                            continue
                        elif point_state == -1:
                            sub = grid_state_matrix[i - 1, j - 1 : j + 1]
                            sub[sub == 0] = -1
                            continue
                    elif i != length - 1 and j == length - 1:
                        if point_state == 1:
                            grid_state_matrix[i - 1 : i + 1, j - 1] = 1
                            continue
                        elif point_state == -1:
                            sub = grid_state_matrix[i - 1 : i + 1, j - 1]
                            sub[sub == 0] = -1
                            continue
                    elif i == length - 1 and j == length - 1:
                        if point_state == 1:
                            grid_state_matrix[i - 1, j - 1] = 1
                            continue
                        elif point_state == -1:
                            grid_state_matrix[i - 1, j - 1] = -1
                            continue

        e1 = time.time()
        print("[gmg] time generate_grid_state_matrix: ", e1 - start_time)
        return grid_state_matrix

    """ def generate_points_state_matrix(self, object_position_2d, lanelet_yaw: float, search_dist: float, image_size: int, possible_lanalets, not_possible_lanelets) -> np.ndarray:
        #0.016s
        '''Generate a points matrix.
        ----------------

        Inputs:
        -------
        object_position_2d: 2D position of the object shape (2, 1)

        lanelet_yaw: yaw of the lanelet

        search_dist: search distance from the object

        Output:
        -------
        points_matrix: size (image_size + 1) by (image_size + 1) by 2.
            40 x 40 x 2 in this project.

        And the points are in the same direction as lanelet.'''
        start_time = time.time()
        point_dist = 1 * search_dist / image_size

        # For Planning Simulation, x point to the front, y point to the left

        # When object is at the center of the image, the points matrix is like:
        # x_offsets =  point_dist * image_size / 2 - point_dist * np.arange(image_size + 1)
        # y_offsets =  point_dist * image_size / 2 - point_dist * np.arange(image_size + 1)

        # When object is at the bottom of the image, the points matrix is like:
        point_state_matrix = np.zeros((image_size + 1, image_size + 1))
        x_offsets =  point_dist * image_size  - point_dist * np.arange(image_size + 1)
        y_offsets =  point_dist * image_size / 2 - point_dist * np.arange(image_size + 1)
        for i, x_offset in enumerate(x_offsets):
            for j, y_offset in enumerate(y_offsets):
                point = calcoffsetpose(object_position_2d, lanelet_yaw, x_offset, y_offset, 0)
                point_state_matrix[i, j] = self.get_point_state(point, possible_lanalets, not_possible_lanelets)

        end_time = time.time()
        print('[gmg] time generate_points_state_matrix: ', end_time - start_time)
        return point_state_matrix """

    """ def generate_grid_state_matrix(self, points_matrix, possible_lanalets, not_possible_lanelets) -> np.ndarray:
        start_time = time.time()

        length = points_matrix.shape[0]
        grid_state_matrix = np.zeros((length-1, length-1))

        # Start the searching from the bottom center of the image
        index_yl = length // 2 - 1
        index_yr = length // 2
        index_x = length - 1

        while index_x >= 0:
            point_state_l = self.get_point_state(points_matrix[index_x, index_yl], possible_lanalets, not_possible_lanelets)
            if point_state_l == 1:
                if index_x < length-1 and index_x > 0 and index_yl > 0:
                    grid_state_matrix[index_x-1:index_x+1, index_yl-1:index_yl+1] = 1
                elif index_x == length - 1:
                    grid_state_matrix[index_x-1, index_yl-1:index_yl+1] = 1
                elif index_x == 0:
                    grid_state_matrix[index_x, index_yl-1:index_yl+1] = 1

                if index_x >= 2:
                    index_x -= 2
                elif index_x == 1:
                    index_x -= 1

        end_time = time.time()
        print('[gmg] time generate_points_state_matrix: ', end_time - start_time)
        return grid_state_matrix """

class GridMapGenerator:
    """Generates a grid map image from a state matrix.

    ----------------
    In this project:
    This matrix is a 39 X 39 matrix. Each element is 0, -1 or 1.
    """

    def __init__(self):
        pass

    def matrix_to_gridmap(self, matrix: np.array) -> ImageMsg:
        """Convert state matrix to grid map image.

        ----------------

        Input:
        ------
        state matrix (39 X 39). Each element is 0, -1 or 1.

        0: white, out of lanelet area
        1: black, on lane and lanelet has same direction as object
        -1: grey, on lane and lanelet has opposite direction as object

        Output:
        -------
        grid map as sensor_msgs/Image. In three colors, white, black and grey.
        """
        if np.shape(matrix)[0] == 0:
            print("matrix is empty")
            return None

        white = [255, 255, 255]
        black = [0, 0, 0]
        grey = [128, 128, 128]

        length = np.shape(matrix)[0]
        image_data = 255 * np.ones(
            (length, length, 3), dtype=np.uint8
        )  # set the background to white
        for i in range(length):
            for j in range(length):
                if matrix[i][j] == 0:
                    image_data[i][j] = white
                elif matrix[i][j] == 1:
                    image_data[i][j] = black
                else:
                    image_data[i][j] = grey

        return self.show_image_in_rviz(image_data)

    def show_image_in_rviz(self, image_data: np.array) -> ImageMsg:
        """Convert image data array into sensor_msgs/Image type."""
        image_msg = ImageMsg()
        image_msg.header.frame_id = "map"
        image_msg.width = image_data.shape[1]
        image_msg.height = image_data.shape[0]
        image_msg.encoding = "rgb8"  # Assuming data is in RGB format
        image_msg.data = image_data.tobytes()

        return image_msg