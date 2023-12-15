# From system and projects imports
from enum import Enum

# Outside imports
import math
import time
from typing import List

# Import message types
from autoware_auto_perception_msgs.msg import DetectedObjectKinematics
from autoware_auto_perception_msgs.msg import ObjectClassification
from autoware_auto_perception_msgs.msg import PredictedObject
from autoware_auto_perception_msgs.msg import PredictedObjectKinematics
from autoware_auto_perception_msgs.msg import PredictedObjects
from autoware_auto_perception_msgs.msg import TrackedObject
from autoware_auto_perception_msgs.msg import TrackedObjectKinematics
from autoware_auto_perception_msgs.msg import TrackedObjects
import geometry_msgs.msg as gmsgs

# From lanelet2 imports
from lanelet2.core import BasicPoint2d
from lanelet2.core import ConstLanelet
from lanelet2.core import ConstLineString2d
from lanelet2.core import Lanelet
from lanelet2.core import LaneletMap
from lanelet2.core import registerId
import lanelet2.geometry as l2_geom
from lanelet2.routing import LaneletPath
from lanelet2.routing import PossiblePathsParams
from lanelet2.routing import RelationType
from lanelet2.routing import RoutingGraph
import lanelet2.traffic_rules as traffic_rules
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image as ImageMsg
import std_msgs.msg as smsgs
import tf2_geometry_msgs as tf2_gmsgs
from tf2_ros.buffer import Buffer
from tum_prediction.utils_tier4 import Tier4Utils
from tum_prediction.map_loader import MapLoader

# Local imports
from tum_prediction.original_path_generator import PathGenerator
from tum_prediction.image_path_generator_pil import PILImagePathGenerator


# Define data structures or new types
class LaneletData:
    def __init__(self):
        self.lanelet: Lanelet()
        self.probability: float


class LaneletsInfoData:
    def __init__(self):
        self.possible_lanelets: List = []
        self.surrounding_lanelets: List = []
        self.current_lanelets: List = []
        self.possible_ids: List = []
        self.surrounding_ids: List = []

class LateralKinematicsToLanelet:
    def __init__(self):
        self.dist_from_left_boundary = 0.0
        self.dist_from_right_boundary = 0.0
        self.left_lateral_velocity = 0.0
        self.right_lateral_velocity = 0.0
        self.filtered_left_lateral_velocity = 0.0
        self.filtered_right_lateral_velocity = 0.0


class ObjectData:
    def __init__(self):
        self.header: smsgs.Header
        self.current_lanelets: ConstLanelets
        self.future_possible_lanelets: ConstLanelets
        self.pose: gmsgs.Pose
        self.twist: gmsgs.Twist
        self.time_delay: float
        self.lateral_kinematics_set = {}  # key: ConstLanelet, value: LateralKinematicsToLanelet
        self.one_shot_maneuver: Maneuver = Maneuver.UNINITIALIZED
        self.output_maneuver: Maneuver = Maneuver.UNINITIALIZED


class Maneuver(Enum):
    UNINITIALIZED = 0
    LANE_FOLLOW = 1
    LEFT_LANE_CHANGE = 2
    RIGHT_LANE_CHANGE = 3


PosePath = List[gmsgs.Pose]


class PredictedRefPath:
    def __init__(self):
        self.probability: float
        self.path: PosePath
        self.maneuver: Maneuver


ConstLanelets = List[ConstLanelet]
Lanelets = List[Lanelet]
LaneletsData = List[LaneletData]
LaneletPaths = List[LaneletPath]


# Input and output topics
input_topic_objects = "/perception/object_recognition/tracking/objects"
original_output_topic = "/perception/object_recognition/objects"
pareller_output_topic = "/parallel/objects"
image_topic = "/parallel/surrounding_image"
runtime_topic = "/parallel/py_runtime"

# This map path is for microservice
# Map_Path = "/autoware/src/universe/autoware.universe/perception/tum_prediction/maps/edgar_map/DEU_GarchingCampus-1/lanelet2_map.osm"


class ParallelPathGeneratorNode(Node):
    """Node for generating path for other vehicles and crosswalk users.

    Use this for final testing!
    --------------------

    ros2 launch autoware_launch planning_simulator.launch.xml map_path:=$HOME/autoware-integration/src/tum_prediction/sample_map/DEU_GarchingCampus-1 vehicle_model:=sample_vehicle sensor_model:=sample_sensor_kit

    ros2 launch tum_prediction parallel_map_based_prediction.launch.xml

    Topics:
    --------------------

    Input topics  :     /perception/object_recognition/tracking/objects : autoware_auto_perception_msgs/TrackedObjects

    Output topics :     /perception/object_recognition/objects : autoware_auto_perception_msgs/PredictedObjects

                        /parallel/objects (Use this for testing) TODO: write how to pub to autoware by switching topic name

                        /parallel/surrounding_image : sensor_msgs/Image

                        /parallel/original_runtime : std_msgs/Float32

    --------------------
    """

    def __init__(self):
        super().__init__("parallel_path_generator_node")

        # declare paremeters from yaml file
        self.Map_Path_ = self.declare_parameter("Map_Path").value
        self.map_origin_latitude_ = self.declare_parameter(
            "Map_Config.map_origin.latitude", 48.262685577935336
        ).value
        self.map_origin_longitude_ = self.declare_parameter(
            "Map_Config.map_origin.longitude", 11.6639997281684
        ).value

        self.enable_delay_compensation_ = self.declare_parameter(
            "enable_delay_compensation", True
        ).value
        self.prediction_time_horizon_ = self.declare_parameter(
            "prediction_time_horizon", 7.8
        ).value  # changed to 7.8 [s]
        self.prediction_sampling_time_interval_ = self.declare_parameter(
            "prediction_sampling_delta_time", 0.5
        ).value
        self.min_velocity_for_map_based_prediction_ = self.declare_parameter(
            "min_velocity_for_map_based_prediction", 1.0
        ).value
        self.min_crosswalk_user_velocity_ = self.declare_parameter(
            "min_crosswalk_user_velocity", 1.0
        ).value
        self.dist_threshold_for_searching_lanelet_ = self.declare_parameter(
            "dist_threshold_for_searching_lanelet", 3.0
        ).value
        self.delta_yaw_threshold_for_searching_lanelet_ = self.declare_parameter(
            "delta_yaw_threshold_for_searching_lanelet", 0.785
        ).value
        self.sigma_lateral_offset_ = self.declare_parameter("sigma_lateral_offset", 0.5).value
        self.sigma_yaw_angle_deg_ = self.declare_parameter("sigma_yaw_angle_deg", 5.0).value
        self.object_buffer_time_length_ = self.declare_parameter(
            "object_buffer_time_length", 2.0
        ).value
        self.history_time_length_ = self.declare_parameter("history_time_length", 1.0).value

        # lane change detection
        self.lane_change_detection_method_ = self.declare_parameter(
            "lane_change_detection.method", "lat_diff_distance"
        ).value
        # self.get_logger().info('Lane change detection method is: ' + str(self.lane_change_detection_method_))

        # lane change detection by time_to_change_lane
        self.dist_threshold_to_bound_ = self.declare_parameter(
            "lane_change_detection.time_to_change_lane.dist_threshold_for_lane_change_detection",
            1.0,
        ).value
        self.time_threshold_to_bound_ = self.declare_parameter(
            "lane_change_detection.time_to_change_lane.time_threshold_for_lane_change_detection",
            5.0,
        ).value
        self.cutoff_freq_of_velocity_lpf_ = self.declare_parameter(
            "lane_change_detection.time_to_change_lane.cutoff_freq_of_velocity_for_lane_change_detection",
            0.1,
        ).value

        # lane change detection by lat_diff_distance
        self.dist_ratio_threshold_to_left_bound_ = self.declare_parameter(
            "lane_change_detection.lat_diff_distance.dist_ratio_threshold_to_left_bound", -0.5
        ).value
        self.dist_ratio_threshold_to_right_bound_ = self.declare_parameter(
            "lane_change_detection.lat_diff_distance.dist_ratio_threshold_to_right_bound", 0.5
        ).value
        self.diff_dist_threshold_to_left_bound_ = self.declare_parameter(
            "lane_change_detection.lat_diff_distance.diff_dist_threshold_to_left_bound", 0.29
        ).value
        self.diff_dist_threshold_to_right_bound_ = self.declare_parameter(
            "lane_change_detection.lat_diff_distance.diff_dist_threshold_to_right_bound", -0.29
        ).value

        self.num_continuous_state_transition_ = self.declare_parameter(
            "lane_change_detection.num_continuous_state_transition", 3
        ).value

        self.reference_path_resolution_ = self.declare_parameter(
            "reference_path_resolution", 0.5
        ).value
        # prediction path will be disabled when the estimated path length exceeds lanelet length.
        # This parameter control the estimated path length = vx * th * (rate)
        self.prediction_time_horizon_rate_for_validate_lane_length_ = self.declare_parameter(
            "prediction_time_horizon_rate_for_validate_lane_length", 0.8
        ).value
        self.image_size_ = self.declare_parameter("image_size", 39).value
        self.map_to_image_rate_ = self.declare_parameter("map_to_image_rate", 2.5).value
        self.lanelet_search_amount_ = self.declare_parameter("lanelet_search_amount", 20).value

        # Instantiate PathGenerator and Tier4Utils classes
        self.pg = PathGenerator(
            self.prediction_time_horizon_,
            self.prediction_sampling_time_interval_,
            self.min_crosswalk_user_velocity_,
        )
        self.tu = Tier4Utils()

        # Declare subscribers and publishers
        self.object_sub = self.create_subscription(
            TrackedObjects, input_topic_objects, self.object_callback, 10
        )
        self.pred_objects_pub = self.create_publisher(PredictedObjects, pareller_output_topic, 10)

        # Lanelet related objects for lanelet map
        self.lanelet_map = LaneletMap()
        self.all_lanelets = None
        self.traffic_rules = None
        self.routing_graph = None
        self.crosswalks_ = None
        self.objects_history_ = {}
        self.tf_buffer = Buffer()

        # load lanelet map for prediction from osm map, rather than subscribing /vector_map topic
        self.get_logger().info("[Node]: Start loading lanelet")
        self.ml = MapLoader(self.Map_Path_, self.map_origin_latitude_, self.map_origin_longitude_)
        self.lanelet_map = self.ml.load_map_for_prediction()
        self.get_logger().info("[Node]: Map is loaded")

        # query lanelets and crosswalks
        self.all_lanelets = self.query_laneletLayer(
            self.lanelet_map
        )  # also rigister Id in this method
        self.get_logger().info("[Node]: Id is registered!")
        crosswalks = self.query_crosswalkLanelets(self.all_lanelets)
        walkways = self.query_walkwayLanelets(self.all_lanelets)
        self.crosswalks_ = crosswalks
        self.crosswalks_.extend(walkways)

        # Create routing graph
        self.traffic_rules = traffic_rules.create(
            traffic_rules.Locations.Germany, traffic_rules.Participants.Vehicle
        )
        self.routing_graph = RoutingGraph(self.lanelet_map, self.traffic_rules)
        self.get_logger().info("[Node]: Routing graph is created!")

        # Instantiate ImageGenerator class
        self.image_pub = self.create_publisher(ImageMsg, image_topic, 10)
        self.search_dist = 0.0
        self.pipg = PILImagePathGenerator(image_size_=self.image_size_)

        self.lanelets_data = None
        self.runtime_pub = self.create_publisher(smsgs.Float32, runtime_topic, 10)
        self.max_object_runtime = 0.001
        self.max_gen_image_runtime = 0.0001

    def object_callback(self, in_objects: TrackedObjects):
        start_time = time.time()
        # grid_image = None

        # Guard for map pointer and frame transformation
        if self.lanelet_map is None:
            return

        world2map_transform = self.tf_buffer.lookup_transform(
            "map",
            in_objects.header.frame_id,
            in_objects.header.stamp,
            rclpy.duration.Duration(seconds=1.0),
        )
        map2world_transform = self.tf_buffer.lookup_transform(
            in_objects.header.frame_id,
            "map",
            in_objects.header.stamp,
            rclpy.duration.Duration(seconds=1.0),
        )

        if world2map_transform is None or map2world_transform is None:
            return

        # Remove old objects information in object history
        objects_detected_time = self.tu.to_cpp_seconds(
            Time.from_msg(in_objects.header.stamp).seconds_nanoseconds()
        )
        self.removeOldObjectsHistory(objects_detected_time)

        # result output
        output = PredictedObjects()
        output.header = in_objects.header
        output.header.frame_id = "map"

        for oobject in in_objects.objects:
            # object_id = self.tu.toHexString(oobject.object_id)
            self.lanelets_data = LaneletsInfoData()
            transformed_object: TrackedObject = oobject

            # transform object frame if it's not based on map frame
            if in_objects.header.frame_id != "map":
                pose_in_map = gmsgs.PoseStamped()
                pose_orig = gmsgs.PoseStamped()
                pose_orig.pose = oobject.kinematics.pose_with_covariance.pose
                pose_in_map = tf2_gmsgs.do_transform_pose(pose_orig, world2map_transform)
                transformed_object.kinematics.pose_with_covariance.pose = pose_in_map.pose

            # get tracking label and update it for the prediction
            tracking_label = transformed_object.classification[0].label
            label = self.changeLabelForPrediction(tracking_label, oobject, self.lanelet_map)

            # TODO: For crosswalk user, use linear prediction for now. Test with another prediction method later
            if label == ObjectClassification.PEDESTRIAN or label == ObjectClassification.BICYCLE:
                predicted_object = self.getPredictedObjectAsCrosswalkUser(transformed_object)
                output.objects.append(predicted_object)

            # For road user
            elif (
                label == ObjectClassification.CAR
                or label == ObjectClassification.BUS
                or label == ObjectClassification.TRAILER
                or label == ObjectClassification.MOTORCYCLE
                or label == ObjectClassification.TRUCK
            ):
                # Update object yaw and velocity
                self.updateObjectData(transformed_object)

                # Get Closest Lanelet
                current_lanelets = self.getCurrentLanelets(transformed_object)
                for lanelet in current_lanelets:
                    self.lanelets_data.current_lanelets.append(lanelet.lanelet)
                
                # Update Objects History.
                self.updateObjectsHistory(output.header, transformed_object, current_lanelets)

                # For off lane obstacles
                if len(current_lanelets) == 0:
                    predicted_path = self.pg.generatePathForOffLaneVehicle(transformed_object)
                    predicted_path.confidence = 1.0
                    if len(predicted_path.path) == 0:
                        continue

                    predicted_object = self.convertToPredictedObject(transformed_object)
                    predicted_object.kinematics.predicted_paths.append(predicted_path)
                    output.objects.append(predicted_object)
                    continue

                # For too-slow vehicle
                if (
                    abs(transformed_object.kinematics.twist_with_covariance.twist.linear.x)
                    < self.min_velocity_for_map_based_prediction_
                ):
                    predicted_path = self.pg.generatePathForLowSpeedVehicle(transformed_object)
                    predicted_path.confidence = 1.0
                    if len(predicted_path.path) == 0:
                        continue

                    predicted_object = self.convertToPredictedObject(transformed_object)
                    predicted_object.kinematics.predicted_paths.append(predicted_path)
                    output.objects.append(predicted_object)
                    continue

                # Get Predicted Reference Path for Each Maneuver and current lanelets.
                ref_paths = self.getPredictedReferencePath(
                    transformed_object, current_lanelets, objects_detected_time
                )

                # If predicted reference path is empty, assume this object is out of the lane
                if len(ref_paths) == 0:
                    predicted_path = self.pg.generatePathForLowSpeedVehicle(transformed_object)
                    predicted_path.confidence = 1.0
                    if len(predicted_path.path) == 0:
                        continue

                    predicted_object = self.convertToPredictedObject(transformed_object)
                    predicted_object.kinematics.predicted_paths.append(predicted_path)
                    output.objects.append(predicted_object)
                    continue

                predicted_paths = []  # List(PredictedPath)
                for ref_path in ref_paths:
                    predicted_path = self.pg.generatePathForOnLaneVehicle(
                        transformed_object, ref_path.path
                    )
                    if len(predicted_path.path) == 0:
                        continue

                    predicted_path.confidence = ref_path.probability
                    predicted_paths.append(predicted_path)

                # Normalize Path Confidence and output the predicted object
                sum_confidence = 0.0
                for predicted_path in predicted_paths:
                    sum_confidence += predicted_path.confidence
                min_sum_confidence_value = 1e-3
                sum_confidence = max(sum_confidence, min_sum_confidence_value)

                for predicted_path in predicted_paths:
                    predicted_path.confidence = predicted_path.confidence / sum_confidence

                predicted_object = self.convertToPredictedObject(transformed_object)
                for predicted_path in predicted_paths:
                    predicted_object.kinematics.predicted_paths.append(predicted_path)

                output.objects.append(predicted_object)

                # Generate grid map for on lane vehicles
                # s_grid = time.time()
                # grid_image = self.pipg.generateImageForOnLaneObject(transformed_object, self.search_dist, self.lanelets_data)
                # e_grid = time.time()
                # if e_grid - s_grid > self.max_gen_image_runtime:
                #     self.max_gen_image_runtime = e_grid - s_grid
                #     self.get_logger().info("new max grid image runtime: '%f'" % self.max_gen_image_runtime)
            # For unknown object
            else:
                predicted_object = self.convertToPredictedObject(transformed_object)

                predicted_path = self.pg.generatePathForNonVehicleObject(transformed_object)
                predicted_path.confidence = 1.0

                predicted_object.kinematics.predicted_paths.append(predicted_path)
                output.objects.append(predicted_object)

        # Publish results
        self.pred_objects_pub.publish(output)
        # Uncomment this and also the lines upper if you want to see the image in Rviz2
        # if grid_image is not None:
        #     self.image_pub.publish(grid_image)

        end_time = time.time()
        self.runtime_pub.publish(smsgs.Float32(data=end_time - start_time))

        if end_time - start_time > self.max_object_runtime:
            self.max_object_runtime = end_time - start_time
            self.get_logger().info("max objects runtime: '%f'" % self.max_object_runtime)

    def getPredictedReferencePath(
        self,
        Tobject: TrackedObject,
        current_lanelets_data: LaneletsData,
        object_detected_time: float,
    ) -> List[PredictedRefPath]:
        obj_vel = abs(Tobject.kinematics.twist_with_covariance.twist.linear.x)

        all_ref_paths = []  # List(PredictedRefPath)

        for current_lanelet_data in current_lanelets_data:
            # parameter for lanelet::routing::PossiblePathsParams
            search_dist = self.prediction_time_horizon_ * obj_vel + self.tu.getLaneletLength3d(
                current_lanelet_data.lanelet
            )
            self.search_dist = search_dist / self.map_to_image_rate_
            possible_params = PossiblePathsParams(
                search_dist, 6, 0, False, True
            )  # 2nd. input is setted by test with original code
            validate_time_horizon = (
                self.prediction_time_horizon_
                * self.prediction_time_horizon_rate_for_validate_lane_length_
            )

            # Step1. Gen the path
            # Step1.1 Get the left lanelet
            left_paths: LaneletPaths = []
            # # old way
            # opt_left = self.routing_graph.left(current_lanelet_data.lanelet)
            # if opt_left is not None:
            #     left_paths = self.routing_graph.possiblePaths(opt_left, possible_params)
            left_lanelet = self.getLeftOrRightLanelets(current_lanelet_data.lanelet, True)
            if left_lanelet is not None:
                left_paths = self.getPathsForNormalOrIsolatedLanelet(
                    left_lanelet, possible_params, Tobject, validate_time_horizon
                )

            # Step1.2 Get the right lanelet
            right_paths: LaneletPaths = []
            # # old way
            # opt_right = self.routing_graph.right(current_lanelet_data.lanelet)
            # if opt_right is not None:
            #     right_paths = self.routing_graph.possiblePaths(opt_right, possible_params)
            right_lanelet = self.getLeftOrRightLanelets(current_lanelet_data.lanelet, False)
            if right_lanelet is not None:
                right_paths = self.getPathsForNormalOrIsolatedLanelet(
                    right_lanelet, possible_params, Tobject, validate_time_horizon
                )

            # Step1.3 Get the centerline
            # # old way
            # center_paths = self.routing_graph.possiblePaths(current_lanelet_data.lanelet, possible_params)
            center_paths = self.getPathsForNormalOrIsolatedLanelet(
                current_lanelet_data.lanelet, possible_params, Tobject, validate_time_horizon
            )

            # Skip calculations if all paths are empty
            if len(left_paths) == 0 and len(right_paths) == 0 and len(center_paths) == 0:
                continue

            # Step2. Predict Object Maneuver
            predicted_maneuver = self.predictObjectManeuver(
                Tobject, current_lanelet_data, object_detected_time
            )

            # Step3. Allocate probability for each predicted maneuver
            maneuver_prob = self.calculateManeuverProbability(
                predicted_maneuver, left_paths, right_paths, center_paths
            )

            # Step4. add candidate reference paths to the all_ref_paths
            path_prob = current_lanelet_data.probability
            self.addReferencePaths(
                Tobject,
                left_paths,
                path_prob,
                maneuver_prob,
                Maneuver.LEFT_LANE_CHANGE,
                all_ref_paths,
            )
            self.addReferencePaths(
                Tobject,
                right_paths,
                path_prob,
                maneuver_prob,
                Maneuver.RIGHT_LANE_CHANGE,
                all_ref_paths,
            )
            self.addReferencePaths(
                Tobject, center_paths, path_prob, maneuver_prob, Maneuver.LANE_FOLLOW, all_ref_paths
            )

        return all_ref_paths

    def getPathsForNormalOrIsolatedLanelet(
        self, lanelet, possible_params, Tobject, validate_time_horizon
    ):
        """Lambda function to get possible paths for isolated lanelet.

        Isolated is often caused by lanelet with no connection e.g. shoulder-lane.
        """
        # if lanelet is not isolated, return normal possible paths
        if not self.isIsolatedLanelet(lanelet, self.routing_graph):
            return self.routing_graph.possiblePaths(lanelet, possible_params)

        # if lanelet is isolated, check if it has enough length
        if not self.validateIsolatedLaneletLength(lanelet, Tobject, validate_time_horizon):
            return []
        else:
            return self.getPossiblePathsForIsolatedLanelet(lanelet)

    def isIsolatedLanelet(self, lanelet: ConstLanelet, graph: RoutingGraph) -> bool:
        """Check if the lanelet is isolated in routing graph."""
        following_lanelets = graph.following(lanelet)
        left_lanelets = graph.lefts(lanelet)
        right_lanelets = graph.rights(lanelet)

        return len(following_lanelets) == 0 and len(left_lanelets) == 0 and len(right_lanelets) == 0

    def validateIsolatedLaneletLength(
        self, lanelet: ConstLanelet, Tobject: TrackedObject, prediction_time: float
    ) -> bool:
        """Validate isolated lanelet length has enough length for prediction."""
        # get closest center line point to object
        center_line = lanelet.centerline
        obj_pose = Tobject.kinematics.pose_with_covariance.pose.position
        obj_point = BasicPoint2d(obj_pose.x, obj_pose.y)

        # get end point of the center line
        end_p = center_line[-1]
        end_point = BasicPoint2d(end_p.x, end_p.y)

        # calc approx distance between closest point and end point
        approx_distance = l2_geom.distance(obj_point, end_point)
        min_length = Tobject.kinematics.twist_with_covariance.twist.linear.x * prediction_time

        return approx_distance > min_length

    def getPossiblePathsForIsolatedLanelet(self, lanelet: ConstLanelet) -> LaneletPaths:
        """Get the Possible Paths For Isolated Lanelet object."""
        possible_lanelets: ConstLanelets = []
        possible_lanelets.append(lanelet)
        possible_paths: LaneletPaths = []
        # need to initialize path with constant lanelets
        possible_paths.append(LaneletPath(possible_lanelets))

        return possible_paths

    def getLeftOrRightLanelets(self, lanelet: ConstLanelet, get_left: bool):
        """Lambda function to extract left/right lanelets."""
        opt = self.routing_graph.left(lanelet) if get_left else self.routing_graph.right(lanelet)
        if opt:
            return opt

        adjacent = (
            self.routing_graph.adjacentLeft(lanelet)
            if get_left
            else self.routing_graph.adjacentRight(lanelet)
        )
        if adjacent:
            return adjacent

        # search for unconnected lanelet
        unconnected_lanelets = (
            self.getLeftLineSharingLanelets(lanelet, self.lanelet_map)
            if get_left
            else self.getRightLineSharingLanelets(lanelet, self.lanelet_map)
        )
        # just return first candidate of unconnected lanelet for now
        if len(unconnected_lanelets) > 0:
            return unconnected_lanelets[0]

        # if no candidate lanelet found, return empty
        return None

    def getLeftLineSharingLanelets(
        self, current_lanelet: ConstLanelet, lanelet_map
    ) -> List[ConstLanelet]:
        """Get the Left LineSharing Lanelets object."""
        output_lanelets: ConstLanelets = []

        # step1: look for lane sharing current left bound
        left_lane_candidates = lanelet_map.laneletLayer.findUsages(current_lanelet.leftBound)
        for candidate in left_lane_candidates:
            # exclude self lanelet
            if candidate == current_lanelet:
                continue

            # if candidate has linestring as right bound, assign it to output
            if candidate.rightBound == current_lanelet.leftBound:
                output_lanelets.append(candidate)

        return output_lanelets

    def getRightLineSharingLanelets(
        self, current_lanelet: ConstLanelet, lanelet_map
    ) -> List[ConstLanelet]:
        """Get the Right LineSharing Lanelets object."""
        output_lanelets: ConstLanelets = []

        # step1: look for lane sharing current right bound
        right_lane_candidates = lanelet_map.laneletLayer.findUsages(current_lanelet.rightBound)
        for candidate in right_lane_candidates:
            # exclude self lanelet
            if candidate == current_lanelet:
                continue

            # if candidate has linestring as left bound, assign it to output
            if candidate.leftBound == current_lanelet.rightBound:
                output_lanelets.append(candidate)

        return output_lanelets

    def addReferencePaths(
        self,
        Tobject: TrackedObject,
        candidate_paths: LaneletPaths,
        path_probability: float,
        maneuver_probability: dict,
        maneuver: Maneuver,
        reference_paths: List,
    ):
        """Self-defined method, was lambda function in the original code."""
        if len(candidate_paths) != 0:
            self.updateFuturePossibleLanelets(Tobject, candidate_paths)

            # Here is the final lanelet path, and this is for image generation
            for candidate_path in candidate_paths:
                for lanelet in candidate_path:
                    if lanelet.id not in self.lanelets_data.possible_ids:
                        self.lanelets_data.possible_ids.append(lanelet.id)
                        self.lanelets_data.possible_lanelets.append(lanelet)

            converted_paths = self.convertPathType(candidate_paths)
            for converted_path in converted_paths:
                predicted_path = PredictedRefPath()
                predicted_path.probability = maneuver_probability[maneuver] * path_probability
                predicted_path.path = converted_path
                predicted_path.maneuver = maneuver
                reference_paths.append(predicted_path)

    def updateFuturePossibleLanelets(self, Tobject: TrackedObject, paths: LaneletPaths):
        object_id = self.tu.toHexString(Tobject.object_id)
        if object_id not in self.objects_history_:
            return

        possible_lanelets = self.objects_history_[object_id][
            -1
        ].future_possible_lanelets  # List(ConstLanelet)
        for path in paths:
            for lanelet in path:
                for p_lanelet in possible_lanelets:
                    if lanelet.id == p_lanelet.id:
                        possible_lanelets.append(lanelet)
                        break

    def convertPathType(self, paths: LaneletPaths) -> List[PosePath]:
        converted_paths = []
        for path in paths:
            converted_path: PosePath = []

            # Insert Positions. Note that we start inserting points from previous lanelet
            if len(path) != 0:
                prev_lanelets = self.routing_graph.previous(path[0])
                if len(prev_lanelets) != 0:
                    prev_lanelet = prev_lanelets[0]
                    init_flag = True
                    prev_p = gmsgs.Pose()
                    for lanelet_p in prev_lanelet.centerline:
                        current_p = gmsgs.Pose()
                        current_p.position = self.tu.toGeomMsgPt(lanelet_p)
                        if init_flag:
                            init_flag = False
                            prev_p = current_p
                            continue

                        lane_yaw = math.atan2(
                            current_p.position.y - prev_p.position.y,
                            current_p.position.x - prev_p.position.x,
                        )
                        current_p.orientation = self.tu.createQuaternionFromYaw(lane_yaw)
                        converted_path.append(current_p)
                        prev_p = current_p

            for lanelet in path:
                init_flag = True
                prev_p = gmsgs.Pose()
                for lanelet_p in lanelet.centerline:
                    current_p = gmsgs.Pose()
                    current_p.position = self.tu.toGeomMsgPt(lanelet_p)
                    if init_flag:
                        init_flag = False
                        prev_p = current_p
                        continue

                    # Prevent from inserting same points
                    if len(converted_path) != 0:
                        last_p = converted_path[-1]
                        tmp_dist = self.tu.calcDistance2d(last_p, current_p)
                        if tmp_dist < 1e-6:
                            prev_p = current_p
                            continue

                    lane_yaw = math.atan2(
                        current_p.position.y - prev_p.position.y,
                        current_p.position.x - prev_p.position.x,
                    )
                    current_p.orientation = self.tu.createQuaternionFromYaw(lane_yaw)
                    converted_path.append(current_p)
                    prev_p = current_p

            # Resample Path
            resampled_converted_path = self.tu.resamplePoseVector(
                converted_path, self.reference_path_resolution_
            )
            converted_paths.append(resampled_converted_path)

        return converted_paths

    def calculateManeuverProbability(
        self,
        predicted_maneuver: Maneuver,
        left_paths: LaneletPaths,
        right_paths: LaneletPaths,
        center_paths: LaneletPaths,
    ) -> dict:
        # instead using ManeuverProbability as a type, use a dict here
        left_lane_change_probability = 0.0
        right_lane_change_probability = 0.0
        lane_follow_probability = 0.0
        if len(left_paths) != 0 and predicted_maneuver == Maneuver.LEFT_LANE_CHANGE:
            LF_PROB_WHEN_LC = 0.9  # probability for lane follow during lane change
            LC_PROB_WHEN_LC = 1.0  # probability for left lane change
            left_lane_change_probability = LC_PROB_WHEN_LC
            right_lane_change_probability = 0.0
            lane_follow_probability = LF_PROB_WHEN_LC
        elif len(right_paths) != 0 and predicted_maneuver == Maneuver.RIGHT_LANE_CHANGE:
            LF_PROB_WHEN_LC = 0.9  # probability for lane follow during lane change
            RC_PROB_WHEN_LC = 1.0  # probability for right lane change
            left_lane_change_probability = 0.0
            right_lane_change_probability = RC_PROB_WHEN_LC
            lane_follow_probability = LF_PROB_WHEN_LC
        elif len(center_paths) != 0:
            LF_PROB = 1.0  # probability for lane follow
            LC_PROB = 0.3  # probability for left lane change
            RC_PROB = 0.3  # probability for right lane change
            if predicted_maneuver == Maneuver.LEFT_LANE_CHANGE:
                # If prediction says left change, but left lane is empty, assume lane follow
                left_lane_change_probability = 0.0
                right_lane_change_probability = RC_PROB if right_paths else 0.0
            elif predicted_maneuver == Maneuver.RIGHT_LANE_CHANGE:
                # If prediction says right change, but right lane is empty, assume lane follow
                left_lane_change_probability = LC_PROB if left_paths else 0.0
                right_lane_change_probability = 0.0
            else:
                # Predicted Maneuver is Lane Follow
                left_lane_change_probability = LC_PROB
                right_lane_change_probability = RC_PROB

            lane_follow_probability = LF_PROB
        else:
            # Center path is empty
            LC_PROB = 1.0  # probability for left lane change
            RC_PROB = 1.0  # probability for right lane change
            lane_follow_probability = 0.0
            if len(left_paths) != 0 and len(right_paths) == 0:
                left_lane_change_probability = LC_PROB
                right_lane_change_probability = 0.0
            elif len(left_paths) == 0 and len(right_paths) != 0:
                left_lane_change_probability = 0.0
                right_lane_change_probability = RC_PROB
            else:
                left_lane_change_probability = LC_PROB
                right_lane_change_probability = RC_PROB

        MIN_PROBABILITY = 1e-3
        max_prob = max(
            MIN_PROBABILITY,
            max(
                lane_follow_probability,
                max(left_lane_change_probability, right_lane_change_probability),
            ),
        )

        # Insert Normalized Probability
        maneuver_prob = {}
        maneuver_prob[Maneuver.LEFT_LANE_CHANGE] = left_lane_change_probability / max_prob
        maneuver_prob[Maneuver.RIGHT_LANE_CHANGE] = right_lane_change_probability / max_prob
        maneuver_prob[Maneuver.LANE_FOLLOW] = lane_follow_probability / max_prob

        return maneuver_prob

    # TODO: finish this method, since time_to_change_lane is not finished.
    def predictObjectManeuver(
        self, Tobject: TrackedObject, current_lanelet_data: LaneletData, object_detected_time: float
    ) -> Maneuver:
        """Do lane change prediction.

        return:
        ------------
        predicted manuever (lane follow, left/right lane change)
        """
        # calculate maneuver
        try:
            if self.lane_change_detection_method_ == "time_to_change_lane":
                current_maneuver = self.predictObjectManeuverByTimeToLaneChange(
                    Tobject, current_lanelet_data, object_detected_time
                )
            elif self.lane_change_detection_method_ == "lat_diff_distance":
                current_maneuver = self.predictObjectManeuverByLatDiffDistance(
                    Tobject, current_lanelet_data, object_detected_time
                )
        except Exception as e:
            raise ValueError("Lane change detection method is invalid." + e)

        object_id = self.tu.toHexString(Tobject.object_id)
        if object_id not in self.objects_history_:
            return current_maneuver

        object_info = self.objects_history_[object_id]

        # update maneuver in object history
        if len(object_info) != 0:
            object_info[-1].one_shot_maneuver = current_maneuver

        # Decide maneuver considering previous results
        if len(object_info) < 2:
            object_info[-1].output_maneuver = current_maneuver
            return current_maneuver

        prev_output_maneuver = object_info[-2].output_maneuver

        for i in range(min(self.num_continuous_state_transition_, len(object_info))):
            tmp_maneuver = object_info[-1 - i].one_shot_maneuver
            if tmp_maneuver != current_maneuver:
                object_info[-1].output_maneuver = prev_output_maneuver
                return prev_output_maneuver

        object_info[-1].output_maneuver = current_maneuver

        return current_maneuver

    def predictObjectManeuverByLatDiffDistance(
        self,
        Tobject: TrackedObject,
        current_lanelet_data: LaneletData,
        _object_detected_time: float,
    ) -> Maneuver:
        # Step1. Check if we have the object in the buffer
        object_id = self.tu.toHexString(Tobject.object_id)
        if object_id not in self.objects_history_:
            return Maneuver.LANE_FOLLOW

        object_info = self.objects_history_[object_id]
        current_time = self.tu.to_cpp_seconds(self.get_clock().now().seconds_nanoseconds())

        # Step2. Get the previous id
        prev_id = len(object_info) - 1
        while prev_id >= 0:
            prev_time_delay = object_info[prev_id].time_delay
            prev_time = (
                self.tu.to_cpp_seconds(
                    Time.from_msg(object_info[prev_id].header.stamp).seconds_nanoseconds()
                )
                + prev_time_delay
            )
            # see if this method will be changed in the future or not, since it used object_detected_time before but commented out
            if current_time - prev_time > self.history_time_length_:
                break
            prev_id -= 1

        if prev_id < 0:
            return Maneuver.LANE_FOLLOW

        # Step3. Get closest previous lanelet ID
        prev_info = object_info[prev_id]
        prev_pose = prev_info.pose
        prev_lanelets = object_info[prev_id].current_lanelets
        if len(prev_lanelets) == 0:
            return Maneuver.LANE_FOLLOW
        prev_lanelet = prev_lanelets[0]
        closest_prev_yaw = float("inf")
        for lanelet in prev_lanelets:
            lane_yaw = self.tu.getLaneletAngle(lanelet, prev_pose.position)
            delta_yaw = self.tu.getYawFromQuaternion(prev_pose.orientation) - lane_yaw
            normalized_delta_yaw = self.tu.normalizeRadian(delta_yaw)
            if normalized_delta_yaw < closest_prev_yaw:
                closest_prev_yaw = normalized_delta_yaw
                prev_lanelet = lanelet

        # Step4. Check if the vehicle has changed lane
        current_lanelet = current_lanelet_data.lanelet
        current_pose = Tobject.kinematics.pose_with_covariance.pose
        dist = self.tu.calcDistance2d(prev_pose, current_pose)
        possible_paths = self.routing_graph.possiblePaths(prev_lanelet, dist + 2.0, 0, False)
        has_lane_changed = True
        if prev_lanelet == current_lanelet:
            has_lane_changed = False
        else:
            for path in possible_paths:
                for lanelet in path:
                    if lanelet == current_lanelet:
                        has_lane_changed = False
                        break

        if has_lane_changed:
            return Maneuver.LANE_FOLLOW

        # Step5. Lane Change Detection
        prev_left_bound = prev_lanelet.leftBound
        prev_right_bound = prev_lanelet.rightBound
        current_left_bound = current_lanelet.leftBound
        current_right_bound = current_lanelet.rightBound
        prev_left_dist = self.calcLeftLateralOffset(prev_left_bound, prev_pose)
        prev_right_dist = self.calcRightLateralOffset(prev_right_bound, prev_pose)
        current_left_dist = self.calcLeftLateralOffset(current_left_bound, current_pose)
        current_right_dist = self.calcRightLateralOffset(current_right_bound, current_pose)
        prev_lane_width = abs(prev_left_dist) + abs(prev_right_dist)
        current_lane_width = abs(current_left_dist) + abs(current_right_dist)
        if prev_lane_width < 1e-3 or current_lane_width < 1e-3:
            self.get_logger().error("[Parallel Map Based Prediction]: Lane width is too small")
            return Maneuver.LANE_FOLLOW

        current_left_dist_ratio = current_left_dist / current_lane_width
        current_right_dist_ratio = current_right_dist / current_lane_width
        diff_left_current_prev = current_left_dist - prev_left_dist
        diff_right_current_prev = current_right_dist - prev_right_dist

        if (
            current_left_dist_ratio > self.dist_ratio_threshold_to_left_bound_
            and diff_left_current_prev > self.diff_dist_threshold_to_left_bound_
        ):
            return Maneuver.LEFT_LANE_CHANGE
        elif (
            current_right_dist_ratio < self.dist_ratio_threshold_to_right_bound_
            and diff_right_current_prev < self.diff_dist_threshold_to_right_bound_
        ):
            return Maneuver.RIGHT_LANE_CHANGE

        return Maneuver.LANE_FOLLOW

    def calcLeftLateralOffset(
        self, boundary_line: ConstLineString2d, search_pose: gmsgs.Pose
    ) -> float:
        return -self.calcRightLateralOffset(boundary_line, search_pose)

    def calcRightLateralOffset(
        self, boundary_line: ConstLineString2d, search_pose: gmsgs.Pose
    ) -> float:
        boundary_path = []  # List(gmsgs.Point)
        for i in range(len(boundary_line)):
            x = boundary_line[i].x
            y = boundary_line[i].y
            boundary_path.append(self.tu.createPoint(x, y, 0.0))

        return abs(self.tu.calcLateralOffset(boundary_path, search_pose.position))

    def convertToPredictedKinematics(
        self, tracked_object: TrackedObjectKinematics
    ) -> PredictedObjectKinematics:
        output = PredictedObjectKinematics()
        output.initial_pose_with_covariance = tracked_object.pose_with_covariance
        output.initial_twist_with_covariance = tracked_object.twist_with_covariance
        output.initial_acceleration_with_covariance = tracked_object.acceleration_with_covariance

        return output

    def convertToPredictedObject(self, tracked_object: TrackedObject) -> PredictedObject:
        predicted_object = PredictedObject()
        predicted_object.kinematics = self.convertToPredictedKinematics(tracked_object.kinematics)
        predicted_object.classification = tracked_object.classification
        predicted_object.object_id = tracked_object.object_id
        predicted_object.shape = tracked_object.shape
        predicted_object.existence_probability = tracked_object.existence_probability

        return predicted_object

    def updateObjectData(self, Tobject: TrackedObject):
        if Tobject.kinematics.orientation_availability == DetectedObjectKinematics.AVAILABLE:
            return

        # Compute yaw angle from the velocity and position of the object
        object_pose = Tobject.kinematics.pose_with_covariance.pose
        object_twist = Tobject.kinematics.twist_with_covariance.twist
        future_object_pose = self.tu.calcoffsetpose_np(
            object_pose, object_twist.linear.x * 0.1, object_twist.linear.y * 0.1, 0.0
        )

        if Tobject.kinematics.twist_with_covariance.twist.linear.x < 0.0:
            if Tobject.kinematics.orientation_availability == DetectedObjectKinematics.SIGN_UNKNOWN:
                original_yaw = self.tu.getYawFromQuaternion(
                    Tobject.kinematics.pose_with_covariance.pose.orientation
                )
                # flip the angle
                Tobject.kinematics.pose_with_covariance.pose.orientation = (
                    self.tu.createQuaternionFromYaw(original_yaw + math.pi)
                )
            else:
                update_object_yaw = self.tu.calcAzimuthAngle(
                    object_pose.position, future_object_pose.position
                )
                Tobject.kinematics.pose_with_covariance.pose.orientation = (
                    self.tu.createQuaternionFromYaw(update_object_yaw)
                )

            Tobject.kinematics.twist_with_covariance.twist.linear.x *= -1.0

        return

    def removeOldObjectsHistory(self, current_time: float):
        invalid_object_id = []
        for object_id, object_data in self.objects_history_.items():
            # If object data is empty, we are going to delete the buffer for the obstacle
            if len(object_data) == 0:
                invalid_object_id.append(object_id)
                continue

            latest_object_time = self.tu.to_cpp_seconds(
                Time.from_msg(object_data[-1].header.stamp).seconds_nanoseconds()
            )

            # Delete Old Objects
            if current_time - latest_object_time > 2.0:
                invalid_object_id.append(object_id)
                continue

            # Delete old information
            while len(object_data) != 0:
                post_object_time = self.tu.to_cpp_seconds(
                    Time.from_msg(object_data[0].header.stamp).seconds_nanoseconds()
                )
                if current_time - post_object_time > 2.0:
                    # Delete Old Position
                    del object_data[0]
                else:
                    break

            if len(object_data) == 0:
                invalid_object_id.append(object_id)
                continue

        for key in invalid_object_id:
            del self.objects_history_[key]

    def updateObjectsHistory(
        self, header: smsgs.Header, Tobject: TrackedObject, current_lanelets_data: LaneletsData
    ):
        object_id = self.tu.toHexString(Tobject.object_id)
        current_lanelets = self.getLanelets(current_lanelets_data)

        single_object_data = ObjectData()
        single_object_data.header = header
        single_object_data.current_lanelets = current_lanelets
        single_object_data.future_possible_lanelets = current_lanelets
        single_object_data.pose = Tobject.kinematics.pose_with_covariance.pose

        object_yaw = self.tu.getYawFromQuaternion(
            Tobject.kinematics.pose_with_covariance.pose.orientation
        )
        single_object_data.pose.orientation = self.tu.createQuaternionFromYaw(object_yaw)
        time_now_in_seconds = self.tu.to_cpp_seconds(self.get_clock().now().seconds_nanoseconds())
        time_header_in_seconds = self.tu.to_cpp_seconds(
            Time.from_msg(header.stamp).seconds_nanoseconds()
        )
        single_object_data.time_delay = abs(time_now_in_seconds - time_header_in_seconds)

        single_object_data.twist = Tobject.kinematics.twist_with_covariance.twist

        # Init lateral kinematics
        for current_lane in current_lanelets:
            lateral_kinematics: LateralKinematicsToLanelet = self.initLateralKinematics(
                current_lane, single_object_data.pose
            )
            single_object_data.lateral_kinematics_set[current_lane] = lateral_kinematics

        if object_id not in self.objects_history_:
            # New Object(Create a new object in object histories)
            object_data = [single_object_data]
            self.objects_history_[object_id] = object_data
        else:
            # Object that is already in the object buffer
            object_data = self.objects_history_[object_id]

            # get previous object data and update
            prev_object_data = object_data[-1]
            self.updateLateralKinematicsVector(
                prev_object_data,
                single_object_data,
                self.routing_graph,
                self.cutoff_freq_of_velocity_lpf_,
            )

            object_data.append(single_object_data)
            self.objects_history_[object_id] = object_data

    def updateLateralKinematicsVector(
        self,
        prev_obj: ObjectData,
        current_obj: ObjectData,
        routing_graph: RoutingGraph,
        lowpass_cutoff: float,
    ):
        """Look for matching lanelet between current/previous object state and calculate velocity.

        Params
        ------------
        prev_obj: previous ObjectData

        current_obj: current ObjectData to be updated

        routing_graph: routing graph object, self.routing_graph.
        """
        current_stamp_time = self.tu.to_cpp_seconds(
            Time.from_msg(current_obj.header.stamp).seconds_nanoseconds()
        )
        prev_stamp_time = self.tu.to_cpp_seconds(
            Time.from_msg(prev_obj.header.stamp).seconds_nanoseconds()
        )
        dt = current_stamp_time - prev_stamp_time

        if dt < 1e-6:
            return  # do not update

        for current_lane, current_lateral_kinematics in current_obj.lateral_kinematics_set.items():
            # 1. has same lanelet
            if current_lane in prev_obj.lateral_kinematics_set:
                prev_lateral_kinematics = prev_obj.lateral_kinematics_set[current_lane]
                self.calcLateralKinematics(
                    prev_lateral_kinematics, current_lateral_kinematics, dt, lowpass_cutoff
                )
                break

            # 2. successive lanelet
            for prev_lane, prev_lateral_kinematics in prev_obj.lateral_kinematics_set.items():
                # The usage of routingRelation is different from the cpp file
                successive_lanelet = (
                    routing_graph.routingRelation(prev_lane, current_lane, False)
                    == RelationType.Successor
                )

                if successive_lanelet:  # lanelet can be connected
                    self.calcLateralKinematics(
                        prev_lateral_kinematics, current_lateral_kinematics, dt, lowpass_cutoff
                    )  # calc velocity
                    break

    def calcLateralKinematics(
        self,
        prev_lateral_kinematics: LateralKinematicsToLanelet,
        current_lateral_kinematics: LateralKinematicsToLanelet,
        dt: float,
        cutoff: float,
    ):
        """Calc lateral velocity and filtered velocity of object in a lanelet.

        Params
        ---------------
        prev_lateral_kinematics: previous lateral lanelet kinematics

        current_lateral_kinematics: current lateral lanelet kinematics

        dt: sampling time [s]
        """
        # calc velocity via backward difference
        current_lateral_kinematics.left_lateral_velocity = (
            current_lateral_kinematics.dist_from_left_boundary
            - prev_lateral_kinematics.dist_from_left_boundary
        ) / dt
        current_lateral_kinematics.right_lateral_velocity = (
            current_lateral_kinematics.dist_from_right_boundary
            - prev_lateral_kinematics.dist_from_right_boundary
        ) / dt

        # low pass filtering left velocity: default cut_off is 0.6 Hz
        current_lateral_kinematics.filtered_left_lateral_velocity = self.FirstOrderLowpassFilter(
            prev_lateral_kinematics.filtered_left_lateral_velocity,
            prev_lateral_kinematics.left_lateral_velocity,
            current_lateral_kinematics.left_lateral_velocity,
            dt,
            cutoff,
        )
        current_lateral_kinematics.filtered_right_lateral_velocity = self.FirstOrderLowpassFilter(
            prev_lateral_kinematics.filtered_right_lateral_velocity,
            prev_lateral_kinematics.right_lateral_velocity,
            current_lateral_kinematics.right_lateral_velocity,
            dt,
            cutoff,
        )

    def FirstOrderLowpassFilter(
        self,
        prev_y: float,
        prev_x: float,
        x: float,
        sampling_time: float = 0.1,
        cutoff_freq: float = 0.1,
    ) -> float:
        """First order Low pass filtering.

        Params
        ---------------
        prev_y: previous filtered value

        prev_x: previous input value

        x current: input value

        cutoff_freq  cutoff frequency in Hz not rad/s (1/s)

        sampling_time  sampling time of discrete system (s)

        Return
        ---------------
        current filtered value: float
        """
        # Eq:  yn = a yn-1 + b (xn-1 + xn)
        wt = 2.0 * math.pi * cutoff_freq * sampling_time
        a = (2.0 - wt) / (2.0 + wt)
        b = wt / (2.0 + wt)

        return a * prev_y + b * (prev_x + x)

    def getLanelets(self, data: LaneletsData) -> ConstLanelets:
        lanelets: ConstLanelets = []
        for lanelet_data in data:
            lanelets.append(lanelet_data.lanelet)

        return lanelets

    def initLateralKinematics(
        self, lanelet: ConstLanelet, pose: gmsgs.Pose
    ) -> LateralKinematicsToLanelet:
        """Init lateral kinematics struct.

        Params
        ---------------
        lanelet: closest lanelet

        pose: search pose

        Return
        ---------------
        lateral kinematics data struct
        """
        lateral_kinematics = LateralKinematicsToLanelet()
        left_bound = lanelet.leftBound
        right_bound = lanelet.rightBound
        left_dist = self.calcAbsLateralOffset(left_bound, pose)
        right_dist = self.calcAbsLateralOffset(right_bound, pose)

        # calc boundary distance
        lateral_kinematics.dist_from_left_boundary = left_dist
        lateral_kinematics.dist_from_right_boundary = right_dist

        # velocities are not init in the first step
        lateral_kinematics.left_lateral_velocity = 0
        lateral_kinematics.right_lateral_velocity = 0
        lateral_kinematics.filtered_left_lateral_velocity = 0
        lateral_kinematics.filtered_right_lateral_velocity = 0

        return lateral_kinematics

    def calcAbsLateralOffset(
        self, boundary_line: ConstLineString2d, search_pose: gmsgs.Pose
    ) -> float:
        """Calc lateral offset from pose to linestring.

        Params
        ---------------
        boundary_line: 2d line strings

        search_pose: search point

        Return
        ---------------
        float
        """
        boundary_path = []
        for point in boundary_line:
            x = point.x
            y = point.y
            boundary_path.append(self.tu.createPoint(x, y, 0.0))

        return abs(self.tu.calcLateralOffset(boundary_path, search_pose.position))

    def changeLabelForPrediction(
        self, label: ObjectClassification.label, Tobject: TrackedObject, lanelet_map: LaneletMap
    ) -> ObjectClassification.label:
        """Change label for prediction.

        Cases
        ---------------
        Case 1: For CAR, BUS, TRUCK, TRAILER, UNKNOWN, do not change label.

        Case 2: For BICYCLE and MOTORCYCLE

        Case 3: For PEDESTRIAN, don't change label for now.

        Return
        ---------------
        label: ObjectClassification.label
        """
        # for car like vehicle do not change labels
        if (
            label == ObjectClassification.CAR
            or label == ObjectClassification.BUS
            or label == ObjectClassification.TRUCK
            or label == ObjectClassification.TRAILER
            or label == ObjectClassification.UNKNOWN
        ):
            return label

        # for bicycle and motorcycle
        elif label == ObjectClassification.MOTORCYCLE or label == ObjectClassification.BICYCLE:
            # if object is within road lanelet and satisfies yaw constraints
            within_road_lanelet = self.withinRoadLanelet(Tobject, lanelet_map, True)
            high_speed_threshold = 25.0 / 18.0 * 5.0  # High speed bycicle 25 km/h
            high_speed_object = (
                Tobject.kinematics.twist_with_covariance.twist.linear.x > high_speed_threshold
            )

            # if the object is within lanelet, do the same estimation with vehicle
            if within_road_lanelet:
                return ObjectClassification.MOTORCYCLE
            elif high_speed_object:
                return label
            else:
                return ObjectClassification.BICYCLE

        # for pedestrian
        elif label == ObjectClassification.PEDESTRIAN:
            # Since so far the program don't change label for pedestrian, the following code won't be used.
            """within_road_lanelet = self.withinRoadLanelet(object, lanelet_map, True)
            max_velocity_for_human_mps = 25.0 / 18.0 * 5.0  # Max human being motion speed is 25km/h
            high_speed_object = object.kinematics.twist_with_covariance.twist.linear.x > max_velocity_for_human_mps

            # fast, human-like object: like segway
            if within_road_lanelet and high_speed_object:
                return label
            elif high_speed_object:
                return label # currently do nothing
            else:
                return label"""
            return label
        else:
            return label

    def getPredictedObjectAsCrosswalkUser(self, Tobject: TrackedObject) -> PredictedObject:
        """Get predicted object as crosswalk user.

        For now do linear prediction.
        """
        predicted_path = self.pg.generatePathForOffLaneVehicle(Tobject)
        predicted_path.confidence = 1.0

        predicted_object = self.convertToPredictedObject(Tobject)
        predicted_object.kinematics.predicted_paths.append(predicted_path)

        return predicted_object

    def getCurrentLanelets(self, Tobject: TrackedObject) -> LaneletsData:
        # obstacle point
        search_point = BasicPoint2d(
            Tobject.kinematics.pose_with_covariance.pose.position.x,
            Tobject.kinematics.pose_with_covariance.pose.position.y,
        )

        # nearest lanelet
        surrounding_lanelets = l2_geom.findNearest(self.lanelet_map.laneletLayer, search_point, 10)
        sls = l2_geom.findNearest(self.lanelet_map.laneletLayer, search_point, self.lanelet_search_amount_)
        for sl in sls:
            self.lanelets_data.surrounding_lanelets.append(sl[1])
            self.lanelets_data.surrounding_ids.append(sl[1].id)

        # No Closest Lanelets
        if len(surrounding_lanelets) == 0:
            return []

        closest_lanelets: LaneletsData = []
        for lanelet in surrounding_lanelets:
            # Check if the close lanelets meet the necessary condition for start lanelets and
            # check if similar lanelet is inside the closest lanelet
            if not self.checkCloseLaneletCondition(
                lanelet, Tobject, search_point
            ) or self.isDuplicated(lanelet, closest_lanelets):
                continue

            closest_lanelet = LaneletData()
            closest_lanelet.lanelet = lanelet[1]
            closest_lanelet.probability = self.calculateLocalLikelihood(lanelet[1], Tobject)
            closest_lanelets.append(closest_lanelet)

        return closest_lanelets

    def checkCloseLaneletCondition(
        self, lanelet, Tobject: TrackedObject, search_point: BasicPoint2d
    ) -> bool:
        # Step1. If we only have one point in the centerline, we will ignore the lanelet
        if len(lanelet[1].centerline) <= 1:
            return False

        # Step2. Check if the obstacle is inside of this lanelet
        if not l2_geom.inside(lanelet[1], search_point):
            return False

        # If the object is in the objects history, we check if the target lanelet is
        # inside the current lanelets id or following lanelets
        object_id = self.tu.toHexString(Tobject.object_id)
        if object_id in self.objects_history_:
            possible_lanelet = self.objects_history_[object_id][-1].future_possible_lanelets
            not_in_possible_lanelet = True
            for (
                p_lanelet
            ) in (
                possible_lanelet
            ):  # different from original code, since the serach strategy is different
                if p_lanelet.id == lanelet[1].id:
                    not_in_possible_lanelet = False
                    break
            if len(possible_lanelet) != 0 and not_in_possible_lanelet:
                return False

        # Step3. Calculate the angle difference between the lane angle and obstacle angle
        object_yaw = self.tu.getYawFromQuaternion(
            Tobject.kinematics.pose_with_covariance.pose.orientation
        )
        lane_yaw = self.tu.getLaneletAngle(
            lanelet[1], Tobject.kinematics.pose_with_covariance.pose.position
        )

        delta_yaw = object_yaw - lane_yaw
        normalized_delta_yaw = self.tu.normalizeRadian(delta_yaw)
        abs_norm_delta = abs(normalized_delta_yaw)

        # Step4. Check if the closest lanelet is valid, and add all
        # of the lanelets that are below max_dist and max_delta_yaw
        object_vel = Tobject.kinematics.twist_with_covariance.twist.linear.x
        is_yaw_reversed = (
            math.pi - self.delta_yaw_threshold_for_searching_lanelet_ < abs_norm_delta
            and object_vel < 0.0
        )
        if lanelet[0] < self.dist_threshold_for_searching_lanelet_ and (
            is_yaw_reversed or abs_norm_delta < self.delta_yaw_threshold_for_searching_lanelet_
        ):
            return True

        return False

    def isDuplicated(self, target_lanelet, lanelets_data: LaneletsData) -> bool:
        CLOSE_LANELET_THRESHOLD = 0.1
        for lanelet_data in lanelets_data:
            target_lanelet_end_p = target_lanelet[1].centerline[-1]
            lanelet_end_p = lanelet_data.lanelet.centerline[-1]
            dist = np.hypot(
                target_lanelet_end_p.x - lanelet_end_p.x, target_lanelet_end_p.y - lanelet_end_p.y
            )

            if dist < CLOSE_LANELET_THRESHOLD:
                return True

        return False

    def calculateLocalLikelihood(self, current_lanelet: Lanelet, Tobject: TrackedObject) -> float:
        obj_point = Tobject.kinematics.pose_with_covariance.pose.position

        # compute yaw difference between the object and lane
        obj_yaw = self.tu.getYawFromQuaternion(
            Tobject.kinematics.pose_with_covariance.pose.orientation
        )
        lane_yaw = self.tu.getLaneletAngle(current_lanelet, obj_point)
        delta_yaw = obj_yaw - lane_yaw
        abs_norm_delta_yaw = abs(self.tu.normalizeRadian(delta_yaw))

        # compute lateral distance
        centerline = current_lanelet.centerline
        converted_centerline = []
        for p in centerline:
            converted_p = self.tu.toGeomMsgPt(p)
            converted_centerline.append(converted_p)

        lat_dist = abs(self.tu.calcLateralOffset(converted_centerline, obj_point))

        # Compute Chi-squared distributed (Equation (8) in the paper)
        sigma_d = self.sigma_lateral_offset_
        sigma_yaw = math.pi * self.sigma_yaw_angle_deg_ / 180.0
        delta = np.array([lat_dist, abs_norm_delta_yaw])
        P_inv = np.array([[1.0 / (sigma_d * sigma_d), 0.0], [0.0, 1.0 / (sigma_yaw * sigma_yaw)]])
        MINIMUM_DISTANCE = 1e-6
        dist = np.maximum(np.dot(delta, np.dot(P_inv, delta)), MINIMUM_DISTANCE)

        return np.float32(1.0 / dist)

    # Methods from lanelet2_extension #
    def query_subtypeLanelets(self, lls: ConstLanelets, subtype) -> ConstLanelets:
        subtype_lanelets: ConstLanelets() = []
        for ll in lls:
            if ll.attributes["subtype"] == subtype:
                subtype_lanelets.append(ll)

        return subtype_lanelets

    def query_laneletLayer(self, ll_map: LaneletMap) -> ConstLanelets:
        lanelets: ConstLanelets() = []
        if ll_map is None:
            print("No map received!")
            return lanelets

        for ll in ll_map.laneletLayer:
            # Try to register the lanelet id here
            registerId(ll.id)
            lanelets.append(ll)

        return lanelets

    def query_crosswalkLanelets(self, lls: ConstLanelets) -> ConstLanelets:
        return self.query_subtypeLanelets(lls, "crosswalk")

    def query_walkwayLanelets(self, lls: ConstLanelets) -> ConstLanelets:
        return self.query_subtypeLanelets(lls, "walkway")

    # Methods that unused
    """
    def predictObjectManeuverByTimeToLaneChange(
        self, object: TrackedObject, current_lanelet_data: LaneletData, _object_detected_time: float
    ) -> Maneuver:
        pass

    def withinRoadLanelet(self, object: TrackedObject, lanelet_map: LaneletMap, use_yaw_information: bool = False) -> bool:

        obj_pos = object.kinematics.pose_with_covariance.pose.position

        search_point = BasicPoint2d(obj_pos.x, obj_pos.y)
        # nearest lanelet
        search_radius = 10.0 # [m]
        surrounding_lanelets = l2_geom.findNearest(self.lanelet_map.laneletLayer, search_point, search_radius)

        for lanelet in surrounding_lanelets:
            if lanelet.attributes["subtype"] != None:
                attr = lanelet.attributes["subtype"]
                if attr == "crosswalk" or attr == "walkway":
                    continue

            if self.withLanelet(object, lanelet[1], use_yaw_information):
                return True

        return False

    def withLanelet(self, object: TrackedObject, lanelet: ConstLanelet, use_yaw_information: bool = False, yaw_threshold: float = 0.6):
        obj_pos = object.kinematics.pose_with_covariance.pose.position
        polygon = lanelet.polygon2d.lineStrings # different from cpp """


def main(args=None):
    rclpy.init(args=args)

    ppgn = ParallelPathGeneratorNode()

    rclpy.spin(ppgn)


if __name__ == "__main__":
    main()
