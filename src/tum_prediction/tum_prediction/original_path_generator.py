# System and Projects imports
# New data types
from typing import List
from typing import Tuple

# Autoware auto msgs
from autoware_auto_perception_msgs.msg import PredictedPath
from autoware_auto_perception_msgs.msg import TrackedObject
import geometry_msgs.msg as gmsgs
import numpy as np
from rclpy.duration import Duration

# Local imports
from tum_prediction.utils_tier4 import Tier4Utils
from tum_prediction.utils_interpolation import Interpolation


class FrenetPoint:
    def __init__(self):
        self.s: float
        self.d: float
        self.s_vel: float
        self.d_vel: float
        self.s_acc: float
        self.d_acc: float


FrenetPath = List[FrenetPoint]
Vector2d = Tuple[float, float]
EntryPoint = Tuple[Vector2d, Vector2d]
PosePath = List[
    gmsgs.Pose
]  # PosePath cannot be instantiated as a class, can be used as e.g. posepath1: PosePath = []


class PathGenerator:
    """Genertate path for other vehicles and crosswalk users.

    Parameters: time_horizon, sampling_time_interval, min_crosswalk_user_velocity

    Output:     PredictedPath
    """

    def __init__(self, time_horizon_, sampling_time_interval_, min_crosswalk_user_velocity_):
        self.tu = Tier4Utils()
        self.interpolation = Interpolation()

        self.time_horizon = time_horizon_
        self.sampling_time_interval = sampling_time_interval_
        self.min_crosswalk_user_velocity = min_crosswalk_user_velocity_

        self.object = TrackedObject()

    def generatePathForNonVehicleObject(self, Tobject: TrackedObject) -> PredictedPath:
        return self._generateStraightPath(Tobject)

    # TODO: generatePathToTargetPoint, this is for cross work user, so don't write for now
    """ def generatePathToTargetPoint(self, object: TrackedObject, point: Vector2d) -> PredictedPath:
        pass """

    # TODO: generatePathForCrosswalkUser
    """ def generatePathForCrosswalkUser(
        self, object: TrackedObject, reachable_crosswalk: EntryPoint
    ) -> PredictedPath:
        pass """

    def generatePathForLowSpeedVehicle(self, Tobject: TrackedObject) -> PredictedPath:
        path = PredictedPath()
        path.time_step = Duration.to_msg(Duration(seconds=self.sampling_time_interval))
        ep = 0.001
        duration = self.time_horizon + ep
        for dt in np.arange(0.0, duration, self.sampling_time_interval):
            path.path.append(Tobject.kinematics.pose_with_covariance.pose)

        return path

    def generatePathForOffLaneVehicle(self, Tobject: TrackedObject) -> PredictedPath:
        return self._generateStraightPath(Tobject)

    def generatePathForOnLaneVehicle(
        self, Tobject: TrackedObject, ref_paths: PosePath
    ) -> PredictedPath:
        if len(ref_paths) < 2:
            return self._generateStraightPath(Tobject)
        else:
            return self._generatePolynomialPath(Tobject, ref_paths)

    def _generateStraightPath(self, Tobject: TrackedObject) -> PredictedPath:
        object_pose = Tobject.kinematics.pose_with_covariance.pose
        object_twist = Tobject.kinematics.twist_with_covariance.twist
        ep = 0.001
        duration = self.time_horizon + ep

        path = PredictedPath()
        path.time_step = Duration.to_msg(Duration(seconds=self.sampling_time_interval))
        path.path = []

        dt = 0.0
        while dt < duration:
            future_obj_pose = self.tu.calcoffsetpose_np(
                object_pose, object_twist.linear.x * dt, object_twist.linear.y * dt, 0.0
            )
            path.path.append(future_obj_pose)
            dt += self.sampling_time_interval

        return path

    def _generatePolynomialPath(self, Tobject: TrackedObject, ref_path: PosePath) -> PredictedPath:
        # Get current Frenet Point
        ref_path_len = self.tu.calcArcLength(ref_path)
        current_point = self._getFrenetPoint(Tobject, ref_path)

        # Step1. Set Target Frenet Point
        # Note that we do not set position s, since we don't know the target longitudinal position
        terminal_point = FrenetPoint()
        terminal_point.s_vel = current_point.s_vel
        terminal_point.s_acc = 0.0
        terminal_point.d = 0.0
        terminal_point.d_vel = 0.0
        terminal_point.d_acc = 0.0

        # Step2. Generate Predicted Path on a Frenet coordinate
        frenet_predicted_path = self._generateFrenetPath(
            current_point, terminal_point, ref_path_len
        )

        # Step3. Interpolate Reference Path for converting predicted path coordinate
        interpolated_ref_path = self._interpolateReferencePath(ref_path, frenet_predicted_path)

        if len(frenet_predicted_path) < 2 or len(interpolated_ref_path) < 2:
            return self._generateStraightPath(Tobject)

        # Step4. Convert predicted trajectory from Frenet to Cartesian coordinate
        return self._convertToPredictedPath(Tobject, frenet_predicted_path, interpolated_ref_path)

    def _interpolateReferencePath(
        self, base_path: PosePath, frenet_predicted_path: FrenetPath
    ) -> PosePath:
        interpolated_path: PosePath = []
        interpolate_num = len(frenet_predicted_path)
        if interpolate_num < 2:
            interpolated_path.append(base_path[0])
            return interpolated_path

        base_path_x = [point.position.x for point in base_path]
        base_path_y = [point.position.y for point in base_path]
        base_path_z = [point.position.z for point in base_path]
        base_path_s = [0.0] * len(base_path)
        for i in range(1, len(base_path)):
            base_path_s[i] = base_path_s[i - 1] + self.tu.calcDistance2d(
                base_path[i - 1], base_path[i]
            )

        resampled_s = []
        for point in frenet_predicted_path:
            resampled_s.append(point.s)
        if resampled_s[0] > resampled_s[-1]:
            resampled_s.reverse()

        # Spline Interpolation
        spline_ref_path_x = self.interpolation.spline(base_path_s, base_path_x, resampled_s)
        spline_ref_path_y = self.interpolation.spline(base_path_s, base_path_y, resampled_s)
        spline_ref_path_z = self.interpolation.spline(base_path_s, base_path_z, resampled_s)

        for i in range(interpolate_num - 1):
            interpolated_pose = gmsgs.Pose()
            current_point = self.tu.createPoint(spline_ref_path_x[i], spline_ref_path_y[i], 0.0)
            next_point = self.tu.createPoint(
                spline_ref_path_x[i + 1], spline_ref_path_y[i + 1], 0.0
            )
            yaw = self.tu.calcAzimuthAngle(current_point, next_point)
            interpolated_pose.position = self.tu.createPoint(
                spline_ref_path_x[i], spline_ref_path_y[i], spline_ref_path_z[i]
            )
            interpolated_pose.orientation = self.tu.createQuaternionFromYaw(yaw)
            interpolated_path.append(interpolated_pose)

        interpolated_path[-1].position = self.tu.createPoint(
            spline_ref_path_x[-1], spline_ref_path_y[-1], spline_ref_path_z[-1]
        )
        interpolated_path[-1].orientation = interpolated_path[interpolate_num - 2].orientation

        return interpolated_path

    def _generateFrenetPath(
        self, current_point: FrenetPoint, target_point: FrenetPoint, max_length: float
    ) -> FrenetPath:
        path: FrenetPath = []
        duration = self.time_horizon

        # Compute Lateral and Longitudinal Coefficients to generate the trajectory
        lat_coeff = self._calcLatCoefficients(current_point, target_point, duration)
        lon_coeff = self._calcLonCoefficients(current_point, target_point, duration)

        t = 0.0
        while t <= duration:
            d_next = (
                current_point.d
                + current_point.d_vel * t
                + 0 * 2 * t**2
                + lat_coeff[0] * t**3
                + lat_coeff[1] * t**4
                + lat_coeff[2] * t**5
            )
            s_next = (
                current_point.s
                + current_point.s_vel * t
                + 0 * 2 * t**2
                + lon_coeff[0] * t**3
                + lon_coeff[1] * t**4
            )
            if s_next > max_length:
                break

            # We assume the object is traveling at a constant speed along s direction
            point = FrenetPoint()
            point.s = max(s_next, 0.0)
            point.s_vel = current_point.s_vel
            point.s_acc = current_point.s_acc
            point.d = d_next
            point.d_vel = current_point.d_vel
            point.d_acc = current_point.d_acc
            path.append(point)

            t += self.sampling_time_interval

        return path

    def _calcLatCoefficients(
        self, current_point: FrenetPoint, target_point: FrenetPoint, T: float
    ) -> List[float]:
        """Calculate lateral Path.

        -------------------------------
        Quintic polynomial for d

             A = np.array([[T**3, T**4, T**5],

                           [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],

                           [6 * T, 12 * T ** 2, 20 * T ** 3]])

             A_inv = np.matrix([[10/(T**3), -4/(T**2), 1/(2*T)],

                                [-15/(T**4), 7/(T**3), -1/(T**2)],

                                [6/(T**5), -3/(T**4),  1/(2*T**3)]])

             b = np.matrix([[xe - self.a0 - self.a1 * T - self.a2 * T**2],

                            [vxe - self.a1 - 2 * self.a2 * T],

                            [axe - 2 * self.a2]])

        Return
        -------------------------------
            3X1 vector -> List[float]
        """
        A_lat_inv = np.matrix(
            [
                [10 / (T**3), -4 / (T**2), 1 / (2 * T)],
                [-15 / (T**4), 7 / (T**3), -1 / (T**2)],
                [6 / (T**5), -3 / (T**4), 1 / (2 * T**3)],
            ]
        )
        b_lat = np.matrix(
            [
                [target_point.d - current_point.d - current_point.d_vel * T],
                [target_point.d_vel - current_point.d_vel],
                [target_point.d_acc],
            ]
        )
        result = A_lat_inv * b_lat

        return result.squeeze().tolist()[0]

    def _calcLonCoefficients(
        self, current_point: FrenetPoint, target_point: FrenetPoint, T: float
    ) -> List[float]:
        """Longitudinal Path Calculation.

        -------------------------------
        Quadric polynomial
            A_inv = np.matrix(  [[1/(T**2),   -1/(3*T)],
                                [-1/(2*T**3), 1/(4*T**2)]])
            b = np.matrix( [[vxe - self.a1 - 2 * self.a2 * T],
                        [axe - 2 * self.a2]])

        Return
        -------------------------------
            2X1 Vector -> List[float]
        """
        A_lon_inv = np.matrix([[1 / (T**2), -1 / (3 * T)], [-1 / (2 * T**3), 1 / (4 * T**2)]])
        b_lon = np.matrix([[target_point.s_vel - current_point.s_vel], [0.0]])
        result = A_lon_inv * b_lon

        return result.squeeze().tolist()[0]

    def _convertToPredictedPath(
        self, Tobject: TrackedObject, frenet_predicted_path: FrenetPath, ref_path: PosePath
    ) -> PredictedPath:
        predicted_path = PredictedPath()
        predicted_path.time_step = Duration.to_msg(Duration(seconds=self.sampling_time_interval))
        predicted_path.path = []
        for i in range(len(ref_path)):
            # Reference Point from interpolated reference path
            ref_pose = ref_path[i]

            # Frenet Point from frenet predicted path
            frenet_point = frenet_predicted_path[i]

            # Converted Pose
            predicted_pose = self.tu.calcoffsetpose_np(ref_pose, 0.0, frenet_point.d, 0.0)
            predicted_pose.position.z = Tobject.kinematics.pose_with_covariance.pose.position.z
            if i == 0:
                predicted_pose.orientation = (
                    Tobject.kinematics.pose_with_covariance.pose.orientation
                )
            else:
                yaw = self.tu.calcAzimuthAngle(
                    predicted_path.path[i - 1].position, predicted_pose.position
                )
                predicted_pose.orientation = self.tu.createQuaternionFromYaw(yaw)

            predicted_path.path.append(predicted_pose)

        return predicted_path

    def _getFrenetPoint(self, Tobject: TrackedObject, ref_path: PosePath) -> FrenetPoint:
        frenet_point = FrenetPoint()
        obj_point = Tobject.kinematics.pose_with_covariance.pose.position

        nearest_segment_idx = self.tu.findNearestSegmentIndex(ref_path, obj_point)
        lonn = self.tu.calcLongitudinalOffsetToSegment(ref_path, nearest_segment_idx, obj_point)
        vx = Tobject.kinematics.twist_with_covariance.twist.linear.x
        vy = Tobject.kinematics.twist_with_covariance.twist.linear.y
        obj_yaw = self.tu.getYawFromQuaternion(
            Tobject.kinematics.pose_with_covariance.pose.orientation
        )
        lane_yaw = self.tu.getYawFromQuaternion(ref_path[nearest_segment_idx].orientation)
        delta_yaw = obj_yaw - lane_yaw

        frenet_point.s = self.tu.calcSignedArcLength(ref_path, 0, nearest_segment_idx) + lonn
        frenet_point.d = self.tu.calcLateralOffset(ref_path, obj_point)
        frenet_point.s_vel = vx * np.cos(delta_yaw) - vy * np.sin(delta_yaw)
        frenet_point.d_vel = vx * np.sin(delta_yaw) + vy * np.cos(delta_yaw)
        frenet_point.s_acc = 0.0
        frenet_point.d_acc = 0.0

        return frenet_point
