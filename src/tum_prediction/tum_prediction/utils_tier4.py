# From system and projects imports
# Outside imports
import math
from typing import List

import geometry_msgs.msg as gmsgs

# From lanelet2
from lanelet2.core import BasicPoint2d
from lanelet2.core import ConstLanelet
from lanelet2.core import ConstLineString3d
from lanelet2.core import ConstPoint2d
from lanelet2.core import ConstPoint3d
from lanelet2.core import LineString3d
from lanelet2.core import Point2d
from lanelet2.core import Point3d
import lanelet2.geometry as l2_geom
import numpy as np

# Local imports
from tum_prediction.utils_interpolation import Interpolation
import unique_identifier_msgs.msg._uuid as uuid

# Constants
CLOSE_S_THRESHOLD = 1e-6


class Tier4Utils:
    """Methods for map_based_prediction_node."""

    def __init__(self):
        self.interpolation = Interpolation()

    def calcoffsetpose_np(self, p: gmsgs.Pose, x: float, y: float, z: float) -> gmsgs.Pose:
        """Calculate offset pose.

        The offset values are defined in the local coordinate of the input pose.
        """
        # Runtime: 6.318092346191406e-05
        transform_q_matrix = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])

        tx = p.position.x
        ty = p.position.y
        tz = p.position.z
        x = p.orientation.x
        y = p.orientation.y
        z = p.orientation.z
        w = p.orientation.w
        transform_quaternion_matrix = np.array(
            [
                [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y, tx],
                [2 * x * y + 2 * w * z, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * w * x, ty],
                [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x**2 - 2 * y**2, tz],
                [0, 0, 0, 1],
            ]
        )

        new_matrix = np.dot(transform_quaternion_matrix, transform_q_matrix)

        position = gmsgs.Point()
        position.x = new_matrix[0, 3]
        position.y = new_matrix[1, 3]
        position.z = new_matrix[2, 3]

        pose = gmsgs.Pose()
        pose.position = position

        R = new_matrix[:3, :3]
        q_w = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
        q_x = (R[2, 1] - R[1, 2]) / (4 * q_w)
        q_y = (R[0, 2] - R[2, 0]) / (4 * q_w)
        q_z = (R[1, 0] - R[0, 1]) / (4 * q_w)

        pose.orientation = self.createQuaternion(q_x, q_y, q_z, q_w)

        return pose

    # Passed pytest
    def createTranslation(self, x: float, y: float, z: float) -> gmsgs.Vector3:
        v = gmsgs.Vector3()
        v.x = x
        v.y = y
        v.z = z

        return v

    # Passed pytest
    def createQuaternion(self, x: float, y: float, z: float, w: float) -> gmsgs.Quaternion:
        q = gmsgs.Quaternion()
        q.x = x
        q.y = y
        q.z = z
        q.w = w

        return q

    # Passed pytest
    def createPoint(self, x: float, y: float, z: float) -> gmsgs.Point:
        p = gmsgs.Point()
        p.x = x
        p.y = y
        p.z = z

        return p

    # Passed pytest
    # Self defined methods to get yaw from quaternion
    def getYawFromQuaternion(self, q: gmsgs.Quaternion) -> float:
        sqx = q.x * q.x
        sqy = q.y * q.y
        sqz = q.z * q.z
        sqw = q.w * q.w

        sarg = -2.0 * (q.x * q.z - q.w * q.y) / (sqx + sqy + sqz + sqw)

        """ if sarg <= -0.99999:
            yaw = -2.0 * math.atan2(q.y, q.x)
        elif sarg >= 0.99999:
            yaw = 2.0 * math.atan2(q.y, q.x)
        else:
            yaw = math.atan2(2 * (q.x * q.y + q.w * q.z), sqw + sqx - sqy - sqz)

        return yaw """

        yaw = np.where(
            sarg <= -0.99999,
            -2.0 * np.arctan2(q.y, q.x),
            np.where(
                sarg >= 0.99999,
                2.0 * np.arctan2(q.y, q.x),
                np.arctan2(2 * (q.x * q.y + q.w * q.z), sqw + sqx - sqy - sqz),
            ),
        )

        return yaw

    # Passed by test with original code
    def toHexString(self, iid: uuid.UUID) -> str:
        hex_string = ""
        for i in range(16):
            hex_string += format(iid.uuid[i], "02x")
        return hex_string

    # Passed pytest
    def createQuaternionFromYaw(self, yaw: float) -> gmsgs.Quaternion:
        # q = tf_transformations.quaternion_from_euler(0.0, 0.0, yaw)
        half_yaw = yaw / 2
        cos_half_yaw = np.cos(half_yaw)
        sin_half_yaw = np.sin(half_yaw)

        # q = np.array([0, 0, sin_half_yaw, cos_half_yaw])
        # return self.createQuaternion(q[0], q[1], q[2], q[3])
        return self.createQuaternion(0.0, 0.0, sin_half_yaw, cos_half_yaw)

    # Passed pytest
    def createQuaternionFromRPY(self, roll: float, pitch: float, yaw: float) -> gmsgs.Quaternion:
        # q = tf_transformations.quaternion_from_euler(roll, pitch, yaw)
        half_roll = roll / 2
        half_pitch = pitch / 2
        half_yaw = yaw / 2

        cos_half_roll = np.cos(half_roll)
        sin_half_roll = np.sin(half_roll)
        cos_half_pitch = np.cos(half_pitch)
        sin_half_pitch = np.sin(half_pitch)
        cos_half_yaw = np.cos(half_yaw)
        sin_half_yaw = np.sin(half_yaw)

        q_w = (
            cos_half_roll * cos_half_pitch * cos_half_yaw
            + sin_half_roll * sin_half_pitch * sin_half_yaw
        )
        q_x = (
            sin_half_roll * cos_half_pitch * cos_half_yaw
            - cos_half_roll * sin_half_pitch * sin_half_yaw
        )
        q_y = (
            cos_half_roll * sin_half_pitch * cos_half_yaw
            + sin_half_roll * cos_half_pitch * sin_half_yaw
        )
        q_z = (
            cos_half_roll * cos_half_pitch * sin_half_yaw
            - sin_half_roll * sin_half_pitch * cos_half_yaw
        )

        # q = np.array([q_x, q_y, q_z, q_w])
        # return self.createQuaternion(q[0], q[1], q[2], q[3])
        return self.createQuaternion(q_x, q_y, q_z, q_w)

    # Passed pytest
    def getPoint(self, point) -> gmsgs.Point:
        match type(point):
            case gmsgs.Point:
                return point
            case gmsgs.Pose:
                return point.position
            case gmsgs.PoseStamped:
                return point.pose.position
            case gmsgs.PoseWithCovarianceStamped:
                return point.pose.pose.position
            case _:
                return point.pose.position  # TrajectoryPoint
                # raise TypeError("point's type is not supported")

    # Passed pytest
    def calcAzimuthAngle(self, p_from: gmsgs.Point, p_to: gmsgs.Point) -> float:
        dx = p_to.x - p_from.x
        dy = p_to.y - p_from.y

        # return math.atan2(dy, dx)
        return np.arctan2(dy, dx)

    # Passed pytest
    def calcElevationAngle(self, p_from: gmsgs.Point, p_to: gmsgs.Point) -> float:
        """Calculate elevation angle of two points."""
        dz = p_to.z - p_from.z
        dist_2d = self.calcDistance2d(p_from, p_to)

        # return math.atan2(dz, dist_2d)
        return np.arctan2(dz, dist_2d)

    # Passed pytest
    def calcSquaredDistance2d(self, point1, point2) -> float:
        p1 = self.getPoint(point1)
        p2 = self.getPoint(point2)
        return (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2

    # Passed pytest
    def calcDistance2d(self, point1, point2) -> float:
        p1 = self.getPoint(point1)
        p2 = self.getPoint(point2)
        return np.hypot(p1.x - p2.x, p1.y - p2.y)

    # Passed pytest
    def calcDistance3d(self, point1, point2) -> float:
        p1 = self.getPoint(point1)
        p2 = self.getPoint(point2)
        return np.hypot(self.calcDistance2d(point1, point2), p1.z - p2.z)

    # Passed pytest
    def calcLongitudinalOffsetToSegment(
        self, points, seg_idx: int, p_target: gmsgs.Point, throw_exception: bool = False
    ) -> float:
        """Calculate longitudinal offset (length along trajectory from seg_idx point to nearest point to p_target on trajectory).

        If seg_idx point is after that nearest point, length is negative.

        Segment is straight path between two continuous points of trajectory.
        """
        if seg_idx >= len(points) - 1:
            if throw_exception:
                raise IndexError("Segment index is invalid.")
            return np.nan

        overlap_removed_points = self.removeOverlapPoints(points, seg_idx)

        if throw_exception:
            self.validateNonEmpty(overlap_removed_points)
        else:
            try:
                self.validateNonEmpty(overlap_removed_points)
            except Exception as e:
                print(e)
                return np.nan

        if seg_idx >= len(overlap_removed_points) - 1:
            if throw_exception:
                raise RuntimeError("Same points are given.")
            return np.nan

        p_front = self.getPoint(overlap_removed_points[seg_idx])
        p_back = self.getPoint(overlap_removed_points[seg_idx + 1])

        segment_vec = np.array([p_back.x - p_front.x, p_back.y - p_front.y, 0.0])
        target_vec = np.array([p_target.x - p_front.x, p_target.y - p_front.y, 0.0])

        return np.dot(segment_vec, target_vec) / np.linalg.norm(segment_vec)

    # Passed pytest
    def removeOverlapPoints(self, points, start_idx: int = 0) -> list:
        if len(points) < start_idx + 1:
            return points

        points_type = type(points)
        dst = points_type()

        for i in range(start_idx + 1):
            dst.append(points[i])

        eps = 1e-8
        for i in range(start_idx + 1, len(points)):
            prev_p = dst[-1]
            curr_p = points[i]
            dist = self.calcDistance2d(prev_p, curr_p)
            if dist < eps:
                continue
            dst.append(points[i])

        return dst

    # Passed pytest
    def findNearestSegmentIndex(self, points, point: gmsgs.Point) -> int:
        """Find nearest segment index to point.

        Segment is straight path between two continuous points of trajectory.

        When point is on a trajectory point whose index is nearest_idx, return nearest_idx - 1.

        When input is a gmsgs.pose, write a new method(trajactory.hpp line 380)
        """
        nearest_idx = self.findNearestIndex(points, point)

        if nearest_idx == 0:
            return 0
        if nearest_idx == len(points) - 1:
            return len(points) - 2

        signed_length = self.calcLongitudinalOffsetToSegment(points, nearest_idx, point)

        if signed_length <= 0:
            return nearest_idx - 1

        return nearest_idx

    # Passed pytest
    def findNearestIndex(self, points, point: gmsgs.Point) -> int:
        """When input is a gmsgs.pose, write a new method(trajactory.hpp line 380)."""
        self.validateNonEmpty(points)

        min_dist = float("inf")
        min_idx = 0

        for i in range(len(points)):
            dist = self.calcSquaredDistance2d(points[i], point)
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        return min_idx

    # Passed pytest
    def validateNonEmpty(self, points):
        if len(points) == 0:
            raise ValueError("Points is empty")

    # Passed pytest
    def calcLateralOffset(
        self, points, p_target: gmsgs.Point, throw_exception: bool = False
    ) -> float:
        """Calculate lateral offset from p_target (length from p_target to trajectory).

        The function gets the nearest segment index between the points of trajectory and the given target point,
        then uses that segment index to calculate lateral offset. Segment is straight path between two continuous points of trajectory.
        """
        overlap_removed_points = self.removeOverlapPoints(points, 0)

        if throw_exception:
            self.validateNonEmpty(overlap_removed_points)
        else:
            try:
                self.validateNonEmpty(overlap_removed_points)
            except Exception as e:
                print(e)
                return np.nan

        if len(overlap_removed_points) == 1:
            if throw_exception:
                raise RuntimeError("Same points are given.")

        seg_idx = self.findNearestSegmentIndex(overlap_removed_points, p_target)

        return self.calcLateralOffset_later(points, p_target, seg_idx, throw_exception)

    # Passed pytest
    def calcLateralOffset_later(
        self, points, p_target: gmsgs.Point, seg_idx: int, throw_exception: bool = False
    ) -> float:
        """Calculate lateral offset from p_target (length from p_target to trajectory) using given segment index.

        Segment is straight path between two continuous points of trajectory.
        """
        overlap_removed_points = self.removeOverlapPoints(points, 0)

        if throw_exception:
            self.validateNonEmpty(overlap_removed_points)
        else:
            try:
                self.validateNonEmpty(overlap_removed_points)
            except Exception as e:
                print(e)
                return np.nan

        if len(overlap_removed_points) == 1:
            if throw_exception:
                raise RuntimeError("Same points are given.")

        p_front = self.getPoint(overlap_removed_points[seg_idx])
        p_back = self.getPoint(overlap_removed_points[seg_idx + 1])

        segment_vec = np.array([p_back.x - p_front.x, p_back.y - p_front.y, 0.0])
        target_vec = np.array([p_target.x - p_front.x, p_target.y - p_front.y, 0.0])

        cross_vec = np.cross(segment_vec, target_vec)
        return cross_vec[2] / np.linalg.norm(segment_vec)

    # Passed pytest
    def calcSignedArcLength(self, points, src_idx: int, dst_idx: int) -> float:
        try:
            self.validateNonEmpty(points)
        except Exception as e:
            print(e)
            return 0.0

        if src_idx > dst_idx:
            return -self.calcSignedArcLength(points, dst_idx, src_idx)

        dist_sum = 0.0
        for i in range(src_idx, dst_idx):
            dist_sum += self.calcDistance2d(points[i], points[i + 1])

        return dist_sum

    # Passed pytest
    def calcArcLength(self, points: List[gmsgs.Pose]) -> float:
        try:
            self.validateNonEmpty(points)
        except Exception as e:
            print(e)
            return 0.0

        return self.calcSignedArcLength(points, 0, len(points) - 1)

    # Passed pytest
    def normalizeRadian(self, rad: float, min_rad: float = -math.pi) -> float:
        max_rad = min_rad + 2 * math.pi
        value = rad % (2 * math.pi)

        if min_rad <= value < max_rad:
            return value

        return value - math.copysign(2 * math.pi, value)

    # Passed pytest
    def toGeomMsgPt(self, src) -> gmsgs.Point:
        if type(src) == ConstPoint3d:
            return gmsgs.Point(x=src.x, y=src.y, z=src.z)
        if type(src) == Point3d:
            return gmsgs.Point(x=src.x, y=src.y, z=src.z)
        elif type(src) == ConstPoint2d:
            return gmsgs.Point(x=src.x, y=src.y, z=0.0)
        elif type(src) == Point2d:
            return gmsgs.Point(x=src.x, y=src.y, z=0.0)
        elif type(src) == gmsgs.Point32:
            return gmsgs.Point(x=src.x, y=src.y, z=src.z)
        else:
            return gmsgs.Point(x=src.x, y=src.y, z=src.z)

    # Passed by test with original code
    def to_cpp_seconds(self, nano_seconds_tuple) -> float:
        """Return the seconds in float which is same as the seconds methods in rclcpp."""
        return nano_seconds_tuple[0] + nano_seconds_tuple[1] * 1e-9

    # Passed by test with original code
    def getLaneletLength3d(self, lanelet: ConstLanelet) -> float:
        """Get lanelet centerline length in 3d space."""
        return l2_geom.length3d(lanelet)

    # Passed pytest
    def resamplePoseVector(
        self,
        points: List[gmsgs.Pose],
        resample_interval: float,
        use_akima_spline_for_xy=False,
        use_lerp_for_z=True,
    ) -> List[gmsgs.Pose]:
        input_length = self.calcArcLength(points)

        resampling_arclength = []
        s = 0.0
        while s < input_length:
            resampling_arclength.append(s)
            s += resample_interval

        if len(resampling_arclength) == 0:
            print("resampling_arclength is empty")
            return points

        # Insert terminal point
        overlap_threshold = 0.1
        if input_length - resampling_arclength[-1] < overlap_threshold:
            resampling_arclength[-1] = input_length
        else:
            resampling_arclength.append(input_length)

        return self.resamplePoseVector_later(
            points, resampling_arclength, use_akima_spline_for_xy, use_lerp_for_z
        )

    # Passed pytest
    def resamplePoseVector_later(
        self,
        points: List[gmsgs.Pose],
        resampled_arclength: List[float],
        use_akima_spline_for_xy: bool,
        use_lerp_for_z: bool,
    ) -> List[gmsgs.Pose]:
        # validate arguments
        if not self.validate_arguments(points, resampled_arclength):
            return points

        position = []  # List[gmsgs.Point]
        for i in range(len(points)):
            position.append(points[i].position)

        resampled_position = self.resamplePointVector_later(
            position, resampled_arclength, use_akima_spline_for_xy, use_lerp_for_z
        )

        resampled_points = []

        # Insert Position
        for i in range(len(resampled_position)):
            pose = gmsgs.Pose()
            pose.position.x = resampled_position[i].x
            pose.position.y = resampled_position[i].y
            pose.position.z = resampled_position[i].z
            resampled_points.append(pose)

        is_driving_forward = self.isDrivingForward(points[0], points[1])
        self.insertOrientation(resampled_points, is_driving_forward)

        # Initial orientation is depend on the initial value of the resampled_arclength when backward driving
        if not is_driving_forward and resampled_arclength[0] < 1e-3:
            resampled_points[0].orientation = points[0].orientation

        return resampled_points

    # Passed pytest
    def resamplePointVector_later(
        self,
        points: List[gmsgs.Point],
        resampled_arclength: List[float],
        use_akima_spline_for_xy: bool,
        use_lerp_for_z: bool,
    ) -> List[gmsgs.Point]:
        # validate arguments
        if not self.validate_arguments(points, resampled_arclength):
            return points

        # Input Path Information
        input_arclength = []
        x = []
        y = []
        z = []

        input_arclength.append(0.0)
        x.append(points[0].x)
        y.append(points[0].y)
        z.append(points[0].z)

        for i in range(1, len(points)):
            prev_pt = points[i - 1]
            curr_pt = points[i]
            ds = self.calcDistance2d(prev_pt, curr_pt)
            input_arclength.append(input_arclength[-1] + ds)
            x.append(curr_pt.x)
            y.append(curr_pt.y)
            z.append(curr_pt.z)

        # Interpolate
        if use_akima_spline_for_xy:
            interpolated_x = self.interpolation.lerp(input_arclength, x, resampled_arclength)
            interpolated_y = self.interpolation.lerp(input_arclength, y, resampled_arclength)
        else:
            interpolated_x = self.interpolation.spline_by_akima(
                input_arclength, x, resampled_arclength
            )
            interpolated_y = self.interpolation.spline_by_akima(
                input_arclength, y, resampled_arclength
            )

        if use_lerp_for_z:
            interpolated_z = self.interpolation.lerp(input_arclength, z, resampled_arclength)
        else:
            interpolated_z = self.interpolation.spline(input_arclength, z, resampled_arclength)

        resampled_points = []  # len(interpolated_x)

        # Insert Position
        for i in range(len(interpolated_x)):
            point = gmsgs.Point()
            point.x = interpolated_x[i]
            point.y = interpolated_y[i]
            point.z = interpolated_z[i]
            resampled_points.append(point)

        return resampled_points

    # Passed pytest
    def isDrivingForward(self, src_pose: gmsgs.Pose, dst_pose: gmsgs.Pose) -> bool:
        # check the first point direction
        src_yaw = self.getYawFromQuaternion(src_pose.orientation)
        pose_direction_yaw = self.calcAzimuthAngle(self.getPoint(src_pose), self.getPoint(dst_pose))

        return abs(self.normalizeRadian(src_yaw - pose_direction_yaw)) < math.pi / 2.0

    # Passed pytest
    def insertOrientation(self, points: List[gmsgs.Pose], is_driving_forward: bool):
        """Insert orientation to each point in points container (trajectory, path, ...)."""
        if is_driving_forward:
            for i in range(len(points) - 1):
                src_point = self.getPoint(points[i])
                dst_point = self.getPoint(points[i + 1])
                pitch = self.calcElevationAngle(src_point, dst_point)
                yaw = self.calcAzimuthAngle(src_point, dst_point)
                self.setOrientation(self.createQuaternionFromRPY(0.0, pitch, yaw), points[i])
                if i == len(points) - 2:
                    # Terminal orientation is same as the point before it
                    self.setOrientation(points[i].orientation, points[i + 1])
        else:
            for i in range(len(points) - 1, 0, -1):
                src_point = self.getPoint(points[i])
                dst_point = self.getPoint(points[i - 1])
                pitch = self.calcElevationAngle(src_point, dst_point)
                yaw = self.calcAzimuthAngle(src_point, dst_point)
                self.setOrientation(self.createQuaternionFromRPY(0.0, pitch, yaw), points[i])
            # Initial orientation is same as the point after it
            self.setOrientation(points[1].orientation, points[0])

    # Passed pytest
    def setOrientation(self, orientation: gmsgs.Quaternion, p: gmsgs.Pose):
        p.orientation = orientation

    # Passed pytest
    def validate_arguments(
        self, input_points: List[gmsgs.Pose], resampling_intervals: List[float]
    ) -> bool:
        # Check size of the arguments
        if not self.validate_size(input_points):
            print("The number of input points is less than 2")
            return False
        elif not self.validate_size(resampling_intervals):
            print("The number of resampling intervals is less than 2")
            return False

        # Check resampling range
        if not self.validate_resampling_range(input_points, resampling_intervals):
            print("resampling interval is longer than input points")
            return False

        # Check duplication
        if not self.validate_points_duplication(input_points):
            print("input points has some duplicated points")
            return False

        return True

    # Passed pytest
    def validate_size(self, points: List[gmsgs.Pose]) -> bool:
        if len(points) < 2:
            return False
        return True

    # Passed pytest
    def validate_resampling_range(
        self, points: List[gmsgs.Pose], resampling_intervals: List[float]
    ) -> bool:
        points_length = self.calcArcLength(points)
        if points_length < resampling_intervals[-1]:
            return False

        return True

    # Passed pytest
    def validate_points_duplication(self, points: List[gmsgs.Pose]) -> bool:
        for i in range(0, len(points) - 1):
            curr_pt = self.getPoint(points[i])
            next_pt = self.getPoint(points[i + 1])
            ds = self.calcDistance2d(curr_pt, next_pt)
            if ds < CLOSE_S_THRESHOLD:
                return False

        return True

    # Passed by test with original code
    def getLaneletAngle(self, lanelet: ConstLanelet, search_point: gmsgs.Point) -> float:
        llt_search_point = BasicPoint2d(search_point.x, search_point.y)
        segment = self.getClosestSegment(llt_search_point, lanelet.centerline)

        return math.atan2(segment[-1].y - segment[0].y, segment[-1].x - segment[0].x)

    # Passed by test with original code
    def getClosestSegment(
        self, search_pt: BasicPoint2d, linestring: ConstLineString3d
    ) -> ConstLineString3d:
        if len(linestring) < 2:
            raise LineString3d()

        # closest_segment: ConstLineString3d
        min_distance = float("inf")

        for i in range(1, len(linestring)):
            prev_basic_pt = linestring[i - 1].basicPoint()
            current_basic_pt = linestring[i].basicPoint()

            prev_pt = Point3d(
                0, prev_basic_pt.x, prev_basic_pt.y, prev_basic_pt.z
            )  # 0 means InvalId
            current_pt = Point3d(0, current_basic_pt.x, current_basic_pt.y, current_basic_pt.z)

            current_segment = LineString3d(0, [prev_pt, current_pt])
            distance = l2_geom.distance(
                l2_geom.to2D(current_segment), search_pt
            )  # Maybe don't need to use basicLineString
            if distance < min_distance:
                closest_segment = current_segment
                min_distance = distance

        return closest_segment

    """ # Passed by test with original code
    def calcoffsetpose(self, p: gmsgs.Pose, x: float, y: float, z: float) -> gmsgs.Pose:
        '''Calculate offset pose. The offset values are defined in the local coordinate of the input pose.'''

        # get transform matrix from translation(twist) and don't do rotation
        transform_quaternion_matrix = self.create_homo_matrix([x,y,z], [0.0, 0.0, 0.0, 1.0])

        # create transform matrix from given pose
        translation = [p.position.x, p.position.y, p.position.z]
        orientation = [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
        pose_quaternion_matrix = self.create_homo_matrix(translation, orientation)

        # compute new pose by matrix multiply
        new_quaternion_matrix = pose_quaternion_matrix @ transform_quaternion_matrix

        return self.homo_matrix_to_pose(new_quaternion_matrix)

    # Passed followed by calcoffsetpose
    # self-defined method
    def create_homo_matrix(self, translation, orientation):
        ''' create homogenous transformation matrix from translation and orientation.

        example:
        translation = [3,0,0], orietation = [0.0, 0.0, 0.0, 1.0]
        return: numpy.ndarray
        '''
        transform_quaternion_matrix = tf_transformations.quaternion_matrix(orientation)
        transform_translation_matrix = tf_transformations.translation_matrix(translation)
        transform_quaternion_matrix[:3,3] = transform_translation_matrix[:3,3]

        return transform_quaternion_matrix

    # Passed followed by calcoffsetpose
    # self-defined method
    def homo_matrix_to_pose(self, homo_matrix) -> gmsgs.Pose:
        translation = homo_matrix[:3,3]

        position = gmsgs.Point()
        position.x = translation[0]
        position.y = translation[1]
        position.z = translation[2]

        pose = gmsgs.Pose()
        pose.position = position

        homo_matrix[:3,3] = np.array([0,0,0]).T
        orientation = tf_transformations.quaternion_from_matrix(homo_matrix)
        pose.orientation = self.createQuaternion(orientation[0], orientation[1], orientation[2], orientation[3])

        return pose """
