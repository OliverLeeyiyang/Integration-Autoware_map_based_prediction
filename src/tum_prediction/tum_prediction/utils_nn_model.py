import torch
import numpy as np
from autoware_auto_perception_msgs.msg import PredictedPath
from rclpy.duration import Duration
import geometry_msgs.msg as gmsgs
from tum_prediction.utils_tier4 import Tier4Utils
from ament_index_python import get_package_share_directory


def loadNorms():
    norm_dict = {}
    pt_path = get_package_share_directory('tum_prediction') + '/norms'
    norm_dict['ego_vel_norm'] = torch.load(pt_path + '/egoVelocityNorm.pt')
    norm_dict['social_inf_norm'] = torch.load(pt_path + '/socialInformationOfCarsInViewNorm.pt')
    norm_dict['gt_norm'] = torch.load(pt_path + '/relGroundTruthNorm.pt')
    return norm_dict


class ModelUtils():
    '''Utility class for model related functions, and be used in routing_based_prediction_node.'''
    def __init__(self, sampling_time_interval_):
        self.sampling_time_interval = sampling_time_interval_
        self.tu = Tier4Utils()

    def transferToMapFrame(self, trajectory, Tobject, viewLength):
        pos = Tobject.kinematics.pose_with_covariance.pose.position
        pos_2d = [pos.x, pos.y]
        orientation = Tobject.kinematics.pose_with_covariance.pose.orientation
        ori = self.tu.getYawFromQuaternion(orientation)
        
        global_point_list = self.toGlobalPointList(trajectory, pos_2d, ori, viewLength)
        global_orientation_list = self.getOrientationList(trajectory, ori)
        predicted_path = self.convertToPredictedPath(global_point_list, global_orientation_list)

        return predicted_path
    
    def convertToPredictedPath(self, global_point_list, global_orientation_list) -> PredictedPath:
        path = PredictedPath()
        path.time_step = Duration.to_msg(Duration(seconds=self.sampling_time_interval))
        path.path = []
        for i in range(len(global_point_list)):
            pose = gmsgs.Pose()
            pose.position.x = global_point_list[i][0]
            pose.position.y = global_point_list[i][1]
            pose.position.z = 0.0
            pose.orientation = self.tu.createQuaternionFromYaw(global_orientation_list[i])
            path.path.append(pose)
        
        return path
    
    def toGlobalPointList(self, trajectory, pos_2d, ori, viewLength):
        # localPointList will be in image coordinate system
        localPointList = self.convertRelativeDxDyTensorAbsolutePointList(trajectory)
        localPointList = self.getPointListInMapBoundary(localPointList, viewLength)
        # print(f'localPointList = {localPointList}')
        globalX = pos_2d[0]
        globalY = pos_2d[1]
        globalOrientation = ori
        globalPointList = self.convertLocalPointListToGlobalPointList(localPointList, globalX, globalY, globalOrientation)

        return globalPointList
    
    def getOrientationList(self, vectorList, ori):
        vectorList = vectorList.view(30, 2).tolist()
        globalOrientationList = np.zeros((30))
        relativeAngle = ori
        globalOrientationList[0] = relativeAngle
        # print(vectorList)
        for i in range(len(globalOrientationList)-1):
            dx = vectorList[i+1][0]
            dy = vectorList[i+1][1]
            angle = np.arctan2(dy, dx)
            relativeAngle = relativeAngle + angle
            if relativeAngle > np.pi:
                relativeAngle -= 2 * np.pi
            elif relativeAngle < -np.pi:
                relativeAngle += 2 * np.pi
            globalOrientationList[i+1] = relativeAngle

        return globalOrientationList

    def convertRelativeDxDyTensorAbsolutePointList(self, relativeDxDyList):
        relativeDxDyWithOrientetAnglesList = self.orientVectorList(relativeDxDyList.view(30, 2).tolist())
        orientedPointList = self.convertingToAbsolutePointList(relativeDxDyWithOrientetAnglesList)
        return orientedPointList
    
    def orientVectorList(self, vectorList):
        orientVectorList = np.zeros((30, 2))
        relativeAngle = np.arctan2(vectorList[0][1], vectorList[0][0]) 
        orientVectorList[0][0] = vectorList[0][0] 
        orientVectorList[0][1] = vectorList[0][1] 

        for i in range(len(vectorList)-1):
            dx = vectorList[i+1][0]
            dy = vectorList[i+1][1]
            if np.abs(dx) < 1e-8 and np.abs(dy) < 1e-8:  # Check for very small values
                angle = 0.0  # Treat near-zero displacements as zero angle
            else:
                angle = np.arctan2(dy, dx) # ToDo: check if this is correct
            relativeAngle = angle + relativeAngle
            # relativeAngle = np.arctan2(np.sin(angle + relativeAngle), np.cos(angle + relativeAngle))

            orientVectorList[i+1][0] = dx * np.cos(relativeAngle) - dy * np.sin(relativeAngle)
            orientVectorList[i+1][1] = dx * np.sin(relativeAngle) + dy * np.cos(relativeAngle)

        return orientVectorList
    
    def convertingToAbsolutePointList(self, orientVectorList):
        x_last = 0.0
        y_last = 0.0
        x = x_last
        y = y_last
        pointList = np.zeros((30, 2))

        # Plot each x and y pair on the axis
        for i in range(len(orientVectorList)):
            x_last, y_last = x, y
            x, y = x_last + orientVectorList[i][0], y_last + orientVectorList[i][1]
            pointList[i][0] = x
            pointList[i][1] = y
        
        return pointList
    
    def getPointListInMapBoundary(self, pointList, viewLength):
        """
        Enforce boundary constraints on  point coordinates. If prediction out of mapInView distance or, the last predictions are set to the previous value that is in the dimensions.

        Args:
        pointList (list or array): List of point coordinates [(x, y), ...].
        viewLength (float): mapInView distance.

        Returns:
        list or array: Modified point coordinates respecting boundary.
        """
        index = 0
        # print(f'globalPointListShape0 =  {pointList.shape[0]}')
        for i in range(pointList.shape[0]):
            index = i
            if pointList[i][0] > viewLength or abs(pointList[i][1]) > viewLength/2: break
        while index < pointList.shape[0]:
            pointList[index][0] = pointList[i-1][0]
            pointList[index][1] = pointList[i-1][1]
            index += 1    
        return pointList
    
    def convertLocalPointListToGlobalPointList(self, localPointList, globalX, globalY, globalOrientation):
        '''Transfer the points from image coordinate system to map coordinate system.
        '''
        globalPointList = np.zeros((30, 2))
        for i in range(len(localPointList)):
            globalPointList[i][0] = globalX + localPointList[i][0] * np.cos(globalOrientation) - localPointList[i][1] * np.sin(globalOrientation)
            globalPointList[i][1] = globalY + localPointList[i][0] * np.sin(globalOrientation) + localPointList[i][1] * np.cos(globalOrientation)
        return globalPointList

    def makeInputTensorValidation(self, encodedMap, egoVelocity, socialInformationOfCarsInView):
            # egoVelocity = torch.tensor(egoVelocity)  # Convert egoVelocity to a tensor
            egoVelocity = egoVelocity.clone().detach()
            combined_tensor = torch.cat((encodedMap.view(-1,64), egoVelocity.view(-1,1), socialInformationOfCarsInView.view(-1,4*20)), dim=1) # Concatenate flattened tensors
            # print(combined_tensor.shape)
            inputTensor = combined_tensor
            return inputTensor

    def predict(self, features, norm_dict, map_encoder, pred_cnn):
        """
        Generate a trajectory prediction for the current car using the given input features.
        Args:
            mapInView (numpy.ndarray): 2D array representing the map information in the car's view.
            egoVelocity (float): The velocity of the current car.
            socialInformationOfCarsInView (numpy.ndarray): 2D array representing the social information of cars in the current car's view.
        Returns:
            numpy.ndarray: The output trajectory prediction for the current car.
        """
        input_sizeX = features['mapInView'].shape[0]
        input_sizeY = features['mapInView'].shape[1]
        # image = torch.tensor(features['mapInView']).float().view(-1, input_sizeX , input_sizeY)
        image = features['mapInView'].clone().detach().float().view(-1, input_sizeX , input_sizeY)
        image = image.unsqueeze(1)
        encodedMap = map_encoder.encoder.forward(image).view(-1,64) # Output of encoder dimensionality reduction

        egoVelocity = torch.tensor(features['egoVelocity'])
        # Add 1.0 to denominator to make the prediction better.
        egoVelocity = torch.where(egoVelocity != 0, egoVelocity / (norm_dict['ego_vel_norm'] + 0.2), torch.zeros_like(egoVelocity)) # normalizing the egoVelocity
        # socialInformationOfCarsInView = torch.tensor(features['socialInformationOfCarsInView'])
        socialInformationOfCarsInView = features['socialInformationOfCarsInView'].clone().detach()
        # socialInformationOfCarsInView = torch.where(socialInformationOfCarsInView != 0, socialInformationOfCarsInView / norm_dict['social_inf_norm'].view(20,4), torch.zeros_like(socialInformationOfCarsInView)).view(80) # normalizing the socialInformationOfCarsInView
        inputTensor = self.makeInputTensorValidation(encodedMap, egoVelocity, socialInformationOfCarsInView).view(-1,145).to(torch.float32) # convert to float32
        prediction = pred_cnn.forward(inputTensor)  # Forward pass -> prediction

        ### Postprocessing --> Only positive x-driving direction allowed
        prediction = prediction.view(30,2) # Separating the two columns
        columnx = prediction[:, 0]
        columny = prediction[:, 1]

        abs_columnx = torch.abs(columnx) # Computing the absolute values for x column (only positive driving direction allowed) -> more stable training
        prediction = torch.stack((abs_columnx, columny), dim=1).view(30,2) # Combining the results back into a tensor with shape (30, 2)
        prediction = prediction.view(30*2)

        # denormalizing the output
        prediction = torch.where(prediction != 0, prediction * norm_dict['gt_norm'], torch.zeros_like(prediction))
        return prediction
