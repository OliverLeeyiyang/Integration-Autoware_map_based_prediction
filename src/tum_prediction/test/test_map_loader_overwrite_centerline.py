from lanelet2.core import Lanelet
from lanelet2.core import LaneletMap
from lanelet2.core import LineString3d
from lanelet2.core import Point3d
from lanelet2.core import getId
import pytest
from tum_prediction.map_loader import MapLoader

ml = MapLoader(" ", 0.0, 0.0)


# Use this to test: $ pytest -rP test_map_loader_overwrite_centerline.py
# Test passed
class TestSuite:
    def setup_method(self):
        self.sample_map_ptr = LaneletMap()

        # create sample lanelets
        p1 = Point3d(getId(), 0.0, 0.0, 0.0)
        p2 = Point3d(getId(), 0.0, 1.0, 0.0)
        p3 = Point3d(getId(), 1.0, 0.0, 0.0)
        p4 = Point3d(getId(), 1.0, 1.0, 0.0)

        ls_left = LineString3d(getId(), [p1, p2])
        ls_right = LineString3d(getId(), [p3, p4])

        p5 = Point3d(getId(), 0.0, 2.0, 0.0)
        p6 = Point3d(getId(), 1.0, 2.0, 0.0)

        ls_left2 = LineString3d(getId(), [p2, p5])
        ls_right2 = LineString3d(getId(), [p4, p6])

        p7 = Point3d(getId(), 0.0, 3.0, 0.0)
        p8 = Point3d(getId(), 1.0, 3.0, 0.0)

        ls_left3 = LineString3d(getId(), [p5, p7])
        ls_right3 = LineString3d(getId(), [p6, p8])

        p9 = Point3d(getId(), 0.0, 1.0, 0.0)
        p10 = Point3d(getId(), 1.0, 1.0, 0.0)

        ls_left4 = LineString3d(getId(), [p9, p5])
        ls_right4 = LineString3d(getId(), [p10, p6])

        self.road_lanelet = Lanelet(getId(), ls_left, ls_right)
        self.road_lanelet.attributes["subtype"] = "road"

        self.next_lanelet = Lanelet(getId(), ls_left2, ls_right2)
        self.next_lanelet.attributes["subtype"] = "road"

        self.next_lanelet2 = Lanelet(getId(), ls_left3, ls_right3)
        self.next_lanelet2.attributes["subtype"] = "road"

        self.merging_lanelet = Lanelet(getId(), ls_left4, ls_right4)
        self.merging_lanelet.attributes["subtype"] = "road"

        self.sample_map_ptr.add(self.road_lanelet)
        self.sample_map_ptr.add(self.next_lanelet)
        self.sample_map_ptr.add(self.next_lanelet2)
        self.sample_map_ptr.add(self.merging_lanelet)

    def test_OverwriteLaneletsCenterline(self):
        resolution = 5.0
        force_overwrite = False
        ml.overwriteLaneletsCenterline(self.sample_map_ptr, resolution, force_overwrite)

        for lanelet in self.sample_map_ptr.laneletLayer:
            print([lanelet.centerline[0], lanelet.centerline[1], lanelet.centerline[2]])
            assert len(lanelet.centerline) != 0


if __name__ == "__main__":
    pytest.main()
