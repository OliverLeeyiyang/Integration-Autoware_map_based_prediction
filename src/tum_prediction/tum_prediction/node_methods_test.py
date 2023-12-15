from tum_prediction.original_path_generator import PathGenerator
from tum_prediction.node_map_based_prediction import ParallelPathGeneratorNode
from tum_prediction.utils_tier4 import Tier4Utils

import rclpy
from rclpy.node import Node
import geometry_msgs.msg as gmsgs
import numpy as np
from shapely.geometry import Point, LineString, MultiLineString
from array import array
import pickle
from sensor_msgs.msg import Image as ImageMsg
from PIL import Image, ImageDraw, ImagePath
from cv_bridge import CvBridge
import torch.nn as nn

# input topics
input_topic_objects = '/perception/object_recognition/tracking/objects'
input_topic_map = '/map/vector_map'

# output topics
output_topic_objects = '/perception/object_recognition/objects'

# Parameters
time_horizon = 10.0
sampling_time_interval = 0.5
min_crosswalk_user_velocity = 1.0



class TestClass(Node):
    '''Test methods in SelfUtils class and ParallelPathGeneratorNode class'''

    def __init__(self):
        super().__init__('test_class_node')
        self.pub = self.create_publisher(ImageMsg, 'image_topic', 10)

        # img = self.test_pil()
        # self.pub.publish(img)


    def test_method_in_selfutils(self):
        self.tu = Tier4Utils()

        self.x = 3.041403437917443
        self.y = 0.0
        self.z = 0.0
        self.pose = gmsgs.Pose()

        # Test methods in SelfUtils class
        print("Test methods in SelfUtils class:")

        # Test createQuaternion method
        print('Test createQuaternion method for (0.0, 0.0, 0.0, 1.0):')
        q = self.tu.createQuaternion(0.0, 0.0, 0.0, 1.0)
        print('Quaternion is: ', q)

        # Test createTranslation method
        print('Test createTranslation method for (1.0, 1.0, 1.0):')
        v = self.tu.createTranslation(self.x, self.y, self.z)
        print('Translation is: ', v)
        
        # Test calc_offset_pose method
        print('Test calc_offset_pose method for Pose (1.0, 1.0, 1.0) with Trans and Quat above:')
        self.pose.position.x = 3849.0256916445646
        self.pose.position.y = 73705.4632446311
        self.pose.position.z = 1.3018890619277954
        self.pose.orientation.x = 0.0
        self.pose.orientation.y = 0.0
        self.pose.orientation.z = -0.9288566368032162
        self.pose.orientation.w = 0.37043939891245115
        #self.pose.orientation = self.tu.createQuaternion(0.0, 0.0, 0.0, 1.0)
        # quat = tf_transformations.quaternion_from_euler(0.0, 0.0, 0.0)
        # self.pose.orientation = self.tu.createQuaternion(quat[0], quat[1], quat[2], quat[3])
        print('Original Pose is: ', self.pose)
        new_pose = self.tu.calcoffsetpose(self.pose, self.x, self.y, self.z)
        print('New pose is: ', new_pose)
    

    def test_ppgn(self):
        ''' Test instructions:

        To test this method, we need to run:
        --------------------

        $ ros2 launch autoware_launch planning_simulator.launch.xml map_path:=$HOME/autoware_map/sample-map-planning vehicle_model:=sample_vehicle sensor_model:=sample_sensor_kit
        
        $ ros2 topic echo /perception/object_recognition/tracking/objects
        
        $ ros2 topic echo /test_pred_path
        
        $ ros2 topic echo /perception/object_recognition/objects

        --------------------
        '''
        self.ppgn = ParallelPathGeneratorNode(time_horizon, sampling_time_interval, min_crosswalk_user_velocity)
        print('ParallelPathGeneratorNode class is ready!')
    

    def test_ran(self):
        self.tu = Tier4Utils()
        point1 = gmsgs.Point()
        point1.x = 1.0
        point1.y = 1.0
        point1.z = 1.0
        point2 = gmsgs.Point()
        point2.x = 2.0
        point2.y = 2.0
        point2.z = 2.0
        point3 = gmsgs.Point()
        point3.x = 3.0
        point3.y = 3.0
        point3.z = 3.0

        from typing import List
        PosePath = List[gmsgs.Pose]
        points: PosePath = []
        pose1 = gmsgs.Pose()
        pose1.position = point1
        points.append(pose1)
        pose2 = gmsgs.Pose()
        pose2.position = point2
        points.append(pose2)
        pose3 = gmsgs.Pose()
        pose3.position = point3
        points.append(pose3)
        # print('idx is:', self.tu.findNearestSegmentIndex(points, point3))
        # print('list is:', self.tu.removeOverlapPoints(points))
        # print('dist is:', self.tu.calcLateralOffset(points, point3))
        # print('dist is:', self.tu.calcSignedArcLength(points, 0,2))
        # print('dist is:', self.tu.calcLongitudinalOffsetToSegment(points,1, point3))
    

    def test_ppg(self):
        self.ppg = PathGenerator(time_horizon, sampling_time_interval, min_crosswalk_user_velocity)
        from typing import TypedDict
        FrenetPoint = TypedDict('FrenetPoint', {'s': float, 'd': float, 's_vel': float, 'd_vel': float, 's_acc': float, 'd_acc': float})
        cur = FrenetPoint()
        cur['s_vel'] = 1.0
        cur['d_vel'] = 1.0
        cur['d'] = 1.0
        cur['d_acc'] = 1.0
        tar = FrenetPoint()
        tar['s_vel'] = 2.0
        tar['d_vel'] = 2.0
        tar['d'] = 2.0
        tar['d_acc'] = 2.0
        print(self.ppg.calcLatCoefficients(cur, tar, 1))
    
    def test_map(self):
        aa = array('l', [1, 2, 3, 4, 5])
        print('type aa is:', type(aa))
        ab = pickle.dumps(aa)
        print('ab is:', ab)
        ac = pickle.loads(ab)
        print('type ac is:', type(ac))
        print('ac is:', ac)\
        
    def test_time(self):
        from rclpy.time import Time
        time1 = self.get_clock().now()
        print('time1 is:', time1)
        time2 = time1.seconds_nanoseconds()
        print('time2 is:', time2)
        time3 = time2[0]
        print('time3 is:', time3)
        self.tu = Tier4Utils()
        time4 = self.tu.to_cpp_seconds(time2)
        print('time4 is:', time4)
    
    def test_tf(self):
        self.tu = Tier4Utils()
        transform_homo_matrix = self.tu.create_homo_matrix([3,0,0],[0,0,0,1])
        print(transform_homo_matrix)
        H = transform_homo_matrix @ transform_homo_matrix
        print(H)
        T = np.array([0,0,0]).T
        print(list(T))
        print(type(list(T)))
        transform_homo_matrix[:3,3] = T
        print(transform_homo_matrix)
    
    def test_len(self):
        a = []
        print(len(a))
        print(not a)
        a.append(1)
        print(len(a))
        print(not a)

    def test_find_path(self):
        import os
        # map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parallel_prediction", "lanelet2_map.osm")
        map_path = os.path.abspath("src/maps/sample-map-planning-modi/lanelet2_map.osm")
        print(map_path)
    
    def test_interpolation(self):
        from tum_prediction.utils_interpolation import Interpolation
        Inp = Interpolation()
        Inp.validateKeysAndValues([1.0], [1.0, 2.0, 3.0, 4.0])
    
    def test_spline(self):
        from tum_prediction.utils_interpolation import Interpolation, SplineInterpolation
        Inp = Interpolation()
        base_keys = [-1.5, 1.0, 5.0, 10.0, 15.0, 20.0]
        base_values = [-1.2, 0.5, 1.0, 1.2, 2.0, 1.0]
        query_keys = [0.0, 8.0, 18.0]
        ans = [-0.075611, 0.997242, 1.573258]

        Spl = SplineInterpolation(base_keys, base_values)
        print(Spl.multi_spline_coef_.a)
        print(Spl.multi_spline_coef_.b)
        print(Spl.multi_spline_coef_.c)
        print(Spl.multi_spline_coef_.d)
        query_values = Spl.getSplineInterpolatedValues(query_keys)
        print(query_values)
    
    def test_getid(self):
        from lanelet2.core import getId, Point3d
        p1 = Point3d(getId(), 1, 2, 3)
        p2 = Point3d(getId(), 1, 2, 3)
        p3 = Point3d(0, 1, 2, 3)
        print(p1.id)
        print(p2.id)
        print(p3.id)

    def test_gridmap(self):
        from tum_prediction.map_loader import MapLoader
        from tum_prediction.image_path_generator import GridMapGenerator

        ml = MapLoader("/home/oliver/ma_prediction_integration/src/maps/sample-map-planning-modi/lanelet2_map.osm", 35.23808753540768, 139.9009591876285)
        lanelet2_map = ml.load_map_for_prediction()
        gmg = GridMapGenerator()

        matrix = np.zeros((39, 39))
        matrix[3,:] = 1
        matrix[:,3] = -1

        image = gmg.matrix_to_gridmap(matrix)
        self.pub.publish(image)
    
    def test_points_matrix(self):
        from tum_prediction.image_path_generator import ImagePathGenerator
        import matplotlib.pyplot as plt

        ipg = ImagePathGenerator(10, 0, 39)
        position = np.array([0, 0])
        yaw = -1.089
        search_dist = 2 * 10 * 3
        points_matrix = ipg.generate_points_matrix(position, yaw, search_dist, 9)
        x = points_matrix[:,:,0]
        y = points_matrix[:,:,1]
        plt.figure(figsize=(8, 8))
        plt.scatter(x, y)
        plt.scatter(x[0, 0], y[0, 0], s=30, label='First Point', color='red', marker='o')
        # Plot the lines x=0 and y=0
        plt.axvline(x=0, color='r', linestyle='--', label='x=0')  # Vertical line at x=0
        plt.axhline(y=0, color='g', linestyle='--', label='y=0')  # Horizontal line at y=0
        plt.scatter(20, 10, s=50, color='green', marker='o', label='Origin (0, 0)')
        # Plot a line with the same direction as the given yaw angle
        slope = np.tan(yaw)  # Compute the slope of the line
        x_vals = np.linspace(-1, 10, 100)  # X values for the line
        y_vals = slope * x_vals  # Compute corresponding Y values
        plt.plot(x_vals, y_vals, color='b', label=f'Yaw Angle: {yaw:.3f}')

        plt.show()

    def test_get_grid_state_matrix(self):
        from tum_prediction.image_path_generator import ImagePathGenerator
        from tum_prediction.image_path_generator import GridMapGenerator

        psm = np.zeros((40, 40))
        psm[0, 0] = 1
        psm[10, :] = 1
        psm[:, 10] = 1
        psm[20, :] = -1

        gmg = GridMapGenerator()

        ipg = ImagePathGenerator(10, 0, 39)
        grid_state_matrix = ipg.get_grid_state_matrix(psm)
        image = gmg.matrix_to_gridmap(grid_state_matrix)
        self.pub.publish(image)
    
    def test_min(self):
        a_list = [-1, 2, 0, -3, 4, -5]
        abs_a_list = [abs(a) for a in a_list]
        index = abs_a_list.index(min(abs_a_list))
        print(index)
        value = a_list[index]
        print(value)
    
    def test_reshape(self):
        a = np.zeros(3)
        print(a)

    def test_methods(self):
        from tum_prediction.image_path_generator import calcoffsetpose
        self.tu = Tier4Utils()
        import time

        x = 3.041403437917443
        y = 0.0
        z = 0.0
        yaw = -1.06
        pose = gmsgs.Pose()

        # Test calc_offset_pose method
        pose.position.x = 3849.0256916445646
        pose.position.y = 73705.4632446311
        pose.position.z = 1.3018890619277954
        pose.orientation = self.tu.createQuaternionFromYaw(-1.06)

        print('Original Pose is: ', pose)
        s1 = time.time()
        new_pose = self.tu.calcoffsetpose_np(pose, x, y, z)
        e1 = time.time()
        print("old time is:", e1-s1)
        print('New pose is: ', new_pose)

        # test new method
        position = np.array([3849.0256916445646, 73705.4632446311], dtype=np.float64)
        s2 = time.time()
        new_position = calcoffsetpose(position, yaw, x, y, z)
        e2 = time.time()
        print("new time is:", e2-s2)
        print('New position is: ', new_position)

        # test new method 2
        s3 = time.time()
        new_pose_2 = self.tu.calcoffsetpose_np(pose, x, y, z)
        e3 = time.time()
        print("old time is:", e3-s3)
        print('New pose is: ', new_pose_2)

    
    def test_cc(self):
        a = np.zeros(10, dtype=float)
        print(a.shape)
        print(len(a))
    
    def test_c2(self):
        ll = [1,2,3,4]
        ll2 = ll[:]
        ll.reverse()
        print(ll)
        ll3 = ll + ll2
        print(ll3)
    
    def test_pil(self):
        # polygon1 = [(20,20),(20,150),(150,20),(20,20)]
        polygon = [(10,10),(20,15),(20,20),(15,20),(10,10)]
        polygon1 = self.rotate_polygon(polygon, [15,15], -np.pi/4)

        # image = ImagePath.Path(polygon1).getbbox()
        size = (39, 39)
        img = Image.new("RGB", size, "grey") 
        img1 = ImageDraw.Draw(img)  
        img1.polygon(polygon1, fill ="black", outline ="black")
        img.show()

        image_msg = self.create_sensor_msg_from_pil_image(img)
        np_image = np.array(img)
        print(np_image.shape)
        print(np_image[0])
        return image_msg
    
    def rotate_polygon(self, polygon, object_pos, lanelet_yaw):
        '''Rotate the polygon by the yaw angle(in radian)'''
        new_polygon = []
        x0 = object_pos[0]
        y0 = object_pos[1]
        for point in polygon:
            x = point[0]
            y = point[1]
            new_point = ((x - x0) * np.cos(lanelet_yaw) - (y - y0) * np.sin(lanelet_yaw) + x0,(x - x0) * np.sin(lanelet_yaw) + (y - y0) * np.cos(lanelet_yaw) + y0)
            new_polygon.append(new_point)

        return new_polygon

    def create_sensor_msg_from_pil_image(self, pil_image):
        # Convert PIL image to NumPy array
        image_np = np.array(pil_image)

        # Create a ROS 2 Image message
        bridge = CvBridge()
        image_msg = bridge.cv2_to_imgmsg(image_np, encoding="rgb8")

        return image_msg
    
    def test_plus(self):
        p1 = (1,2)
        p2 = (3,4)
        p3 = p1 + p2
        print(p3)
    
    def test_dict(self):
        dict = {}
        dict['a'] = 1
        dict['b'] = 2
        dict['c'] = 3
        dict['d'] = 4
        for values in dict.values():
            print(values)
    
    def test_path(self):
        import os
        p = os.path.abspath('autoware-integration/src/tum_prediction/norms')
        print(p)
        p1 = p + '/egoVelocityNorm.pt'
        print(p1)
    
    def test_model(self):
        from tum_prediction.nn_model import Model, ConvAutoencoder
        model = Model()
        model.load()
    
    def test_float_convert(self):
        a = "1.0"
        print(type(a))
        b = float(a)
        print(b)
        c = b / 1000.0
        print(c)
    
    def test_file_path(self):
        import os
        from ament_index_python import get_package_share_directory
        abpath = os.path.join(get_package_share_directory('tum_prediction'), 'norms', 'egoVelocityNorm.pt')
        print(abpath)
        aa = os.path.abspath('') + '/eva.txt'
        print(aa)
    
    def test_image_floor(self):
        matrix = np.zeros((39, 39))
        matrix[5,:] = 255.0
        matrix[10,:] = 255.0
        matrix[15,:] = 255.0
        matrix[20,:] = 255.0
        matrix[25,:] = 255.0
        matrix[7, :] = 128.0
        matrix[12, :] = 128.0
        matrix[17, :] = 128.0
        matrix[22, :] = 128.0
        print(matrix[10:20, 1:3])
        new_matrix = (2 - np.floor(2 * matrix / 255.0)) / 2.0
        print(new_matrix.shape)
        print(new_matrix[10:20, 1:3])

    def test_matrix(self):
        test_matrix = np.zeros((20,4), dtype=np.float64)
        test_matrix[:,1] = 15
        test_matrix[:,3] = 0
        print(test_matrix[:5,:])





def main(args=None):
    rclpy.init(args=args)

    tc = TestClass()
    print('Hi from pp_test.py')
    # tc.test_getid()

    # tc.test_ppg()
    # tc.test_ran()
    
    # tc.test_method_in_selfutils()
    # tc.test_geopandas()
    # tc.test_map()
    # tc.test_time()
    # tc.test_tf()
    # tc.test_len()
    # tc.test_find_path()
    # tc.test_interpolation()
    # tc.test_spline()
    # tc.test_gridmap()
    # tc.test_points_matrix()
    # tc.test_get_grid_state_matrix()
    # tc.test_min()
    # tc.test_reshape()
    # tc.test_methods()
    # tc.test_cc()
    # tc.test_c1()
    # tc.test_plus()
    # tc.test_dict()
    # tc.test_path()
    # img = tc.test_pil()
    # tc.test_model()
    # tc.test_float_convert()
    # tc.test_file_path()
    # tc.test_image_floor()
    tc.test_matrix()
    """ s1 = time.time()
    img = tc.test_pil()
    tc.pub.publish(img)
    e1 = time.time()
    print("time is:", e1-s1)

    rclpy.spin(tc) """



if __name__ == '__main__':
    main()