import rclpy
from rclpy.node import Node
from typing import List
from tier4_debug_msgs.msg import StringStamped
from std_msgs.msg import Float32
import numpy as np
import csv
import os
import matplotlib.pyplot as plt

# Unfollowed topics
orinigal_pred_objects_topic = "/perception/object_recognition/objects"
nn_pred_objects_topic = "/parallel/nn_pred_objects"

# Followed  runtime topics
original_runtime_topic = "/perception/object_recognition/prediction/map_based_prediction/debug/calculation_time"
py_runtime_topic = "/parallel/py_runtime"
nn_runtime_topic = "/parallel/nn_runtime"

# CSV data saving path
Runtime_csv_path = os.path.abspath('') + '/runtime_evaluation_results.csv'
Runtime_png_path = os.path.abspath('') + '/runtime_evaluation.png'

# Command to record the bag file
# ros2 bag record -o drag_situation /perception/object_recognition/tracking/objects /perception/object_recognition/objects /parallel/nn_pred_objects /perception/object_recognition/prediction/map_based_prediction/debug/calculation_time /parallel/nn_runtime /parallel/py_runtime


class EvaluationNode(Node):
    '''So far only designed for single object situation'''
    def __init__(self):
        super().__init__('evaluation_node')
        self.cpp_runtime_sub = self.create_subscription(
            StringStamped,
            original_runtime_topic,
            self.callback3,
            10)
        self.py_runtime_sub = self.create_subscription(
            Float32,
            py_runtime_topic,
            self.callback4,
            10)
        self.nn_runtime_sub = self.create_subscription(
            Float32,
            nn_runtime_topic,
            self.callback5,
            10)
        
        self.cpp_runtime = []
        self.py_runtime = []
        self.nn_runtime = []

    def callback3(self, msg):
        runtime = float(msg.data) / 1000.0 # convert to seconds
        self.cpp_runtime.append(runtime)
    
    def callback4(self, msg):
        runtime = msg.data
        self.py_runtime.append(runtime)
    
    def callback5(self, msg):
        runtime = msg.data
        self.nn_runtime.append(runtime)

    def do_runtime_evaluation(self):
        if len(self.cpp_runtime) != 0 or len(self.py_runtime) != 0 or len(self.nn_runtime) != 0:
            self.runtime_eva = RuntimeEvaluation(self.cpp_runtime, self.py_runtime, self.nn_runtime)
            self.runtime_eva.write_data_to_csv()
            self.runtime_eva.save_data_to_image()


class RuntimeEvaluation():
    def __init__(self, cpp_runtime: List[float], py_runtime: List[float], nn_runtime: List[float]):
        self.temp_dict = {}
        if len(cpp_runtime) != 0:
            self.temp_dict['cpp'] = np.array(cpp_runtime)
        if len(py_runtime) != 0:
            self.temp_dict['py'] = np.array(py_runtime)
        if len(nn_runtime) != 0:
            self.temp_dict['nn'] = np.array(nn_runtime)

        self.max_runtime_dict = self.compute_max_runtime()
        self.runtime_dict = {}
        self.mean_runtime_dict = {}
        self.std_runtime_dict = {}
        self.generate_mean_and_std_dict()
    
    def runtime_high_pass_filter(self, runtime_list, max_runtime):
        '''Delete the runtime data if it is too low.'''
        runtime_list = runtime_list[ runtime_list >= (max_runtime / 100.0)]
        
        return runtime_list, np.mean(runtime_list), np.std(runtime_list)
    
    def generate_mean_and_std_dict(self):
        for key, value in self.temp_dict.items():
            new_value, mean, std = self.runtime_high_pass_filter(value, self.max_runtime_dict[key])
            self.runtime_dict[key] = new_value
            self.mean_runtime_dict[key] = mean
            self.std_runtime_dict[key] = std
    
    def compute_max_runtime(self):
        max_runtime_dict = {}
        for key, value in self.temp_dict.items():
            max_runtime_dict[key] = np.max(value)
        
        return max_runtime_dict
    
    def write_data_to_csv(self):
        saver = SaveDataIntoCSV(Runtime_csv_path)
        saver.save_to_csv(self.mean_runtime_dict, self.max_runtime_dict, self.std_runtime_dict)
    
    def save_data_to_image(self):
        '''Draw all the runtime data into a image.'''
        plt.figure()
        plt.title('Runtime Evaluation')

        plt.subplot(311)
        plt.plot(1000*self.runtime_dict['cpp'], 'r', label='AW-C++ node')
        plt.ylabel('Runtime (ms)')
        plt.legend()

        plt.subplot(312)
        plt.plot(1000*self.runtime_dict['py'], 'r', label='AW-Python node')
        plt.ylabel('Runtime (ms)')
        plt.legend()

        plt.subplot(313)
        plt.plot(1000*self.runtime_dict['nn'], 'b', label='NN-Python node')
        plt.ylabel('Runtime (ms)')
        plt.xlabel('Topic Message Count')
        plt.legend()

        plt.savefig(Runtime_png_path)
        




class SaveDataIntoCSV:
    def __init__(self, file_path):
        self.file_path = file_path

    def save_to_csv(self, mean_data, max_data, std_data):
        with open(self.file_path, 'w', newline='') as csvfile:
            fieldnames = ['Method', 'Mean Runtime', 'Max Runtime', 'Std Runtime']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            for method in mean_data.keys():
                writer.writerow({
                    'Method': method,
                    'Mean Runtime': mean_data[method],
                    'Max Runtime': max_data[method],
                    'Std Runtime': std_data[method]
                })


def main(args=None):
    rclpy.init(args=args)

    eva_node = EvaluationNode()
    rclpy.logging.get_logger("EvaluationNode").info('Please start the rosbag now')

    try:
        rclpy.spin(eva_node)
    except KeyboardInterrupt:
        eva_node.do_runtime_evaluation()
        rclpy.logging.get_logger("Saving results to /home/${user}/runtime_evaluation_results.csv").info('Done')
        rclpy.logging.get_logger("Saving results to /home/${user}/runtime_evaluation.png").info('Done')

    eva_node.destroy_node()



if __name__ == '__main__':
    main()