import math

from lanelet2.core import BasicPoint3d
from lanelet2.core import ConstLanelet
from lanelet2.core import ConstLineString3d
from lanelet2.core import GPSPoint
from lanelet2.core import LaneletMap
from lanelet2.core import LineString3d
from lanelet2.core import Point3d
from lanelet2.core import getId
from lanelet2.io import Origin
from lanelet2.io import loadRobust
from lanelet2.projection import LocalCartesianProjector
from lanelet2.projection import UtmProjector
import numpy as np
from shapely import LineString
from shapely import Point
from shapely import distance


class MapLoader:
    def __init__(self, map_file_path, lat, lon):
        self.lanelet2_filename = map_file_path
        self.lanelet2_map_projector_type = "UTM"
        self.center_line_resolution = 5.0

        # map origin
        self.lat = lat
        self.lon = lon

        self.lanelet2_map = LaneletMap()

    def load_map_for_prediction(self):
        self.lanelet2_map = self.load_map(self.lanelet2_filename, self.lanelet2_map_projector_type)
        self.overwriteLaneletsCenterline(self.lanelet2_map, self.center_line_resolution, True)
        return self.lanelet2_map

    def load_map(self, lanelet2_filename: str, lanelet2_map_projector_type: str) -> LaneletMap:
        if lanelet2_map_projector_type == "MGRS":
            print("MGRS is not supported")

        elif lanelet2_map_projector_type == "UTM":
            position = GPSPoint(self.lat, self.lon)
            origin = Origin(position)
            projector = UtmProjector(origin)
            lmap, load_errors = loadRobust(lanelet2_filename, projector)
            if len(load_errors) == 0:
                return lmap

        elif lanelet2_map_projector_type == "LocalCartesian":
            origin = Origin(self.lat, self.lon, 520.0)
            projector = LocalCartesianProjector(origin)
            lmap, load_errors = loadRobust(lanelet2_filename, projector)
            if len(load_errors) == 0:
                return lmap

        else:
            print("lanelet2_map_projector_type is not supported")
            return None

        for error in load_errors:
            print(error)
        return None

    def overwriteLaneletsCenterline(
        self, lanelet_map: LaneletMap, resolution: float, force_overwrite: bool
    ):
        """All the Id of the centerline points are set to 0."""
        for lanelet_obj in lanelet_map.laneletLayer:
            if force_overwrite or len(lanelet_obj.centerline) == 0:
                fine_center_line = self.generateFineCenterline(lanelet_obj, resolution)
                lanelet_obj.centerline = fine_center_line

    def generateFineCenterline(self, lanelet_obj: ConstLanelet, resolution: float) -> LineString3d:
        left_length = self.laneletLength(lanelet_obj.leftBound)
        right_length = self.laneletLength(lanelet_obj.rightBound)
        longer_distance = left_length if left_length > right_length else right_length
        num_segments = max(math.ceil(longer_distance / resolution), 1)

        # Resample points
        left_points = self.resamplePoints(lanelet_obj.leftBound, num_segments)
        right_points = self.resamplePoints(lanelet_obj.rightBound, num_segments)

        # Create centerline
        centerline = LineString3d(getId())
        for i in range(num_segments + 1):
            # Add ID for the average point of left and right
            center_basic_point = 0.5 * (right_points[i] + left_points[i])
            center_point = Point3d(
                getId(), center_basic_point.x, center_basic_point.y, center_basic_point.z
            )
            centerline.append(center_point)

        return centerline

    def resamplePoints(self, line_string: ConstLineString3d, num_segments: int):
        # Calculate length
        line_length = self.laneletLength(line_string)

        # Calculate accumulated lengths
        accumulated_lengths = self.calculateAccumulatedLengths(line_string)
        if len(accumulated_lengths) < 2:
            return []

        # Create each segment
        resampled_points = []  # List[BasicPoint3d]
        i = 0
        while i <= num_segments:
            # Find two nearest points
            target_length = (float(i) / num_segments) * line_length
            index_pair = self.find_nearest_index_pair(accumulated_lengths, target_length)

            # Apply linear interpolation
            back_point = line_string[index_pair[0]]  # Point3d
            front_point = line_string[index_pair[1]]
            bp = BasicPoint3d(back_point.x, back_point.y, back_point.z)
            fp = BasicPoint3d(front_point.x, front_point.y, front_point.z)
            direction_vector = fp - bp

            back_length = accumulated_lengths[index_pair[0]]
            front_length = accumulated_lengths[index_pair[1]]
            segment_length = front_length - back_length
            target_point = bp + ((target_length - back_length) / segment_length) * direction_vector

            # Add to list
            resampled_points.append(target_point)
            i += 1

        return resampled_points

    def calculateAccumulatedLengths(self, line_string: ConstLineString3d):
        segment_distances = self.calculateSegmentDistances(line_string)
        accumulated_lengths = [0]
        # accumulated_lengths.extend
        accumulated_lengths += np.cumsum(segment_distances).tolist()

        return accumulated_lengths

    def calculateSegmentDistances(self, line_string: ConstLineString3d):
        segment_distances = []
        for i in range(1, len(line_string)):
            p1 = Point(line_string[i].x, line_string[i].y)
            p2 = Point(line_string[i - 1].x, line_string[i - 1].y)
            lineDistance = distance(p1, p2)
            segment_distances.append(lineDistance)

        return segment_distances

    def find_nearest_index_pair(self, accumulated_lengths: list, target_length: float):
        # List size
        N = len(accumulated_lengths)

        # Front
        if target_length < accumulated_lengths[1]:
            return (0, 1)

        # Back
        if target_length > accumulated_lengths[N - 2]:
            return (N - 2, N - 1)

        # Middle
        for i in range(1, N):
            if accumulated_lengths[i - 1] <= target_length <= accumulated_lengths[i]:
                return (i - 1, i)

        # Throw an exception because this never happens
        raise RuntimeError("No nearest point found.")

    def laneletLength(self, lanelet: ConstLanelet) -> float:
        """Sum of distances between consecutive points."""
        points = []
        for point in lanelet:
            points.append((point.x, point.y))
        return LineString(points).length
