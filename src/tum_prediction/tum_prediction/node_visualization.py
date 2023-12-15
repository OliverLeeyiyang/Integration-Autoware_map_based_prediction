from autoware_auto_perception_msgs.msg import PredictedObjects
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

# Followed topics
orinigal_objects_topic = ("/perception/object_recognition/objects") 
py_pred_objects_topic = "/parallel/objects"  # output topic of node_map_based_prediction.py
nn_pred_objects_topic = "/parallel/nn_pred_objects"  # output topic of node_routing_based_prediction.py

# Topics to publish
py_pred_path_marker_publisher = "/visualization/py_pred_path_marker"
nn_pred_path_marker_publisher = "/visualization/nn_pred_path_marker"


class Visualization(Node):
    def __init__(self):
        super().__init__("path_visualization_node")
        self.py_marker_publisher = self.create_publisher(MarkerArray, py_pred_path_marker_publisher, 10)
        self.nn_marker_publisher = self.create_publisher(MarkerArray, nn_pred_path_marker_publisher, 10)

        self.subscription1 = self.create_subscription(
            PredictedObjects, py_pred_objects_topic, self.callback1, 10
        )
        self.subscription2 = self.create_subscription(
            PredictedObjects, nn_pred_objects_topic, self.callback2, 10
        )

        self.frame_id = None

    def callback1(self, msg: PredictedObjects):
        self.frame_id = msg.header.frame_id
        marker_array = MarkerArray()
        ID = 0

        for obj in msg.objects:
            for pre_path in obj.kinematics.predicted_paths:
                # create a marker for each path with unique ID
                if len(pre_path.path) == 0:
                    print("path is empty")
                    continue
                else:
                    marker = Marker()
                    marker.header.frame_id = self.frame_id
                    marker.type = Marker.LINE_STRIP
                    marker.id = ID
                    ID += 1
                    marker.action = Marker.ADD
                    marker.lifetime = rclpy.duration.Duration(seconds=0.1).to_msg()
                    marker.scale.x = 0.4  # line width
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    marker.color.a = 1.0  # transparency

                    for pose in pre_path.path:
                        marker.points.append(pose.position)

                    marker_array.markers.append(marker)

        self.py_marker_publisher.publish(marker_array)
    
    def callback2(self, msg: PredictedObjects):
        self.frame_id = msg.header.frame_id
        marker_array = MarkerArray()
        ID = 0

        for obj in msg.objects:
            for pre_path in obj.kinematics.predicted_paths:
                # create a marker for each path with unique ID
                if len(pre_path.path) == 0:
                    print("path is empty")
                    continue
                else:
                    marker = Marker()
                    marker.header.frame_id = self.frame_id
                    marker.type = Marker.LINE_STRIP
                    marker.id = ID
                    ID += 1
                    marker.action = Marker.ADD
                    marker.lifetime = rclpy.duration.Duration(seconds=0.1).to_msg()
                    marker.scale.x = 0.4  # line width
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                    marker.color.a = 1.0  # transparency

                    for pose in pre_path.path:
                        marker.points.append(pose.position)

                    marker_array.markers.append(marker)

        self.nn_marker_publisher.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    visualization = Visualization()
    rclpy.spin(visualization)


if __name__ == "__main__":
    main()
