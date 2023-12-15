import math
from typing import List

from autoware_auto_planning_msgs.msg import Trajectory
from autoware_auto_planning_msgs.msg import TrajectoryPoint
import geometry_msgs.msg as gmsgs
from lanelet2.core import Point2d
from lanelet2.core import Point3d
from lanelet2.core import getId
import pytest
from tum_prediction.utils_tier4 import Tier4Utils

TrajectoryPointArray = List[TrajectoryPoint]

tu = Tier4Utils()


epsilon = 1e-6


def createPose(x, y, z, roll, pitch, yaw):
    p = gmsgs.Pose()
    p.position = tu.createPoint(x, y, z)
    p.orientation = tu.createQuaternionFromRPY(roll, pitch, yaw)

    return p


class GenTraj:
    def __init__(self):
        pass

    def generateTestTrajectory(
        self, num_points, point_interval, vel=0.0, init_theta=0.0, delta_theta=0.0
    ) -> Trajectory:
        traj = Trajectory()
        for i in range(num_points):
            theta = init_theta + i * delta_theta
            x = i * point_interval * math.cos(theta)
            y = i * point_interval * math.sin(theta)

            p = TrajectoryPoint()
            p.pose = createPose(x, y, 0.0, 0.0, 0.0, theta)
            p.longitudinal_velocity_mps = vel
            traj.points.append(p)

        return traj

    def generateTestTrajectoryPointArray(
        self, num_points, point_interval, vel=0.0, init_theta=0.0, delta_theta=0.0
    ) -> TrajectoryPointArray:
        traj = TrajectoryPointArray()
        for i in range(num_points):
            theta = init_theta + i * delta_theta
            x = i * point_interval * math.cos(theta)
            y = i * point_interval * math.sin(theta)

            p = TrajectoryPoint()
            p.pose = createPose(x, y, 0.0, 0.0, 0.0, theta)
            p.longitudinal_velocity_mps = vel
            traj.append(p)

        return traj

    def updateTrajectoryVelocityAt(self, points, idx, vel):
        points[idx].longitudinal_velocity_mps = vel


gt = GenTraj()


# Passed
def test_getId():
    p1 = Point3d(getId(), 1, 2, 3)
    p2 = Point3d(getId(), 1, 2, 3)
    p3 = Point3d(0, 1, 2, 3)
    p1_p2 = p1.id == p2.id
    p1_p3 = p1.id == p3.id
    assert not p1_p2
    assert not p1_p3


# ------------------------------------------------------------#
# Tests for class Tier4Utils
# ------------------------------------------------------------#


# Passed
def test_createTranslation():
    x = 1.0
    y = 2.0
    z = 3.0
    p = gmsgs.Vector3()
    p.x = x
    p.y = y
    p.z = z
    assert tu.createTranslation(x, y, z) == p


# Passed
def test_createQuaternion():
    x = 1.0
    y = 2.0
    z = 3.0
    w = 4.0
    q = gmsgs.Quaternion()
    q.x = x
    q.y = y
    q.z = z
    q.w = w
    assert tu.createQuaternion(x, y, z, w) == q


# Passed
def test_createPoint():
    x = 1.0
    y = 2.0
    z = 3.0
    p = gmsgs.Point()
    p.x = x
    p.y = y
    p.z = z
    assert tu.createPoint(x, y, z) == p


# Passed
def test_getYawFromQuaternion():
    q1 = gmsgs.Quaternion()
    q1.x = 0.0
    q1.y = 0.0
    q1.z = 0.0
    q1.w = 1.0
    q2 = gmsgs.Quaternion()
    q2.x = 0.0
    q2.y = 0.0
    q2.z = 0.70710678118
    q2.w = 0.70710678118

    yaw1 = 0.0
    yaw2 = math.pi / 2.0
    assert tu.getYawFromQuaternion(q1) == yaw1
    assert tu.getYawFromQuaternion(q2) == yaw2


# Passed
def test_createQuaternionFromYaw():
    yaw = math.pi / 4.0
    expected_q = gmsgs.Quaternion()
    expected_q.x = 0.0
    expected_q.y = 0.0
    expected_q.z = 0.38268343236
    expected_q.w = 0.92387953251

    cal_q = tu.createQuaternionFromYaw(yaw)
    assert type(cal_q) == gmsgs.Quaternion
    assert cal_q.x == pytest.approx(expected_q.x, abs=epsilon)
    assert cal_q.y == pytest.approx(expected_q.y, abs=epsilon)
    assert cal_q.z == pytest.approx(expected_q.z, abs=epsilon)
    assert cal_q.w == pytest.approx(expected_q.w, abs=epsilon)


# Passed
def test_createQuaternionFromRPY():
    roll = 0.0
    pitch = 0.0
    yaw = math.pi / 4.0
    expected_q = gmsgs.Quaternion()
    expected_q.x = 0.0
    expected_q.y = 0.0
    expected_q.z = 0.38268343236
    expected_q.w = 0.92387953251

    roll2 = 0.0
    pitch2 = 0.1
    yaw2 = 0.7854
    expected_q2 = gmsgs.Quaternion()
    expected_q2.x = -0.019126242445566
    expected_q2.y = 0.046174713977463
    expected_q2.z = 0.382206025062786
    expected_q2.w = 0.922724572689336

    cal_q = tu.createQuaternionFromRPY(roll, pitch, yaw)
    cal_q2 = tu.createQuaternionFromRPY(roll2, pitch2, yaw2)

    assert type(cal_q) == gmsgs.Quaternion
    assert cal_q.x == pytest.approx(expected_q.x, abs=epsilon)
    assert cal_q.y == pytest.approx(expected_q.y, abs=epsilon)
    assert cal_q.z == pytest.approx(expected_q.z, abs=epsilon)
    assert cal_q.w == pytest.approx(expected_q.w, abs=epsilon)
    assert type(cal_q2) == gmsgs.Quaternion
    assert cal_q2.x == pytest.approx(expected_q2.x, abs=epsilon)
    assert cal_q2.y == pytest.approx(expected_q2.y, abs=epsilon)
    assert cal_q2.z == pytest.approx(expected_q2.z, abs=epsilon)
    assert cal_q2.w == pytest.approx(expected_q2.w, abs=epsilon)


# Passed
def test_calcAzimuthAngle():
    p1 = gmsgs.Point()
    p1.x = 0.0
    p1.y = 0.0
    p1.z = 0.0
    p2 = gmsgs.Point()
    p2.x = 1.0
    p2.y = 0.0
    p2.z = 0.0
    p3 = gmsgs.Point()
    p3.x = 0.0
    p3.y = 1.0
    p3.z = 0.0
    p4 = gmsgs.Point()
    p4.x = 1.0
    p4.y = 1.0
    p4.z = 0.0

    assert tu.calcAzimuthAngle(p1, p2) == 0.0
    assert tu.calcAzimuthAngle(p1, p3) == math.pi / 2.0
    assert tu.calcAzimuthAngle(p1, p4) == math.pi / 4.0


# Passed
def test_calcDistance2d():
    p1 = gmsgs.Point()
    p1.x = 10.0
    p1.y = 2.0
    p1.z = 0.0
    p2 = gmsgs.Point()
    p2.x = 2.0
    p2.y = 4.0
    p2.z = 0.0

    assert tu.calcDistance2d(p1, p2) == pytest.approx(8.246211251235321, abs=epsilon)


# Passed
def test_calcDistance3d():
    p1 = gmsgs.Point()
    p1.x = 10.0
    p1.y = 2.0
    p1.z = 9.0
    p2 = gmsgs.Point()
    p2.x = 2.0
    p2.y = 4.0
    p2.z = 20.0

    assert tu.calcDistance3d(p1, p2) == pytest.approx(13.747727084867520, abs=epsilon)


# Passed
def test_calcSquaredDistance2d():
    p1 = gmsgs.Point()
    p1.x = 10.0
    p1.y = 2.0
    p1.z = 0.0
    p2 = gmsgs.Point()
    p2.x = 2.0
    p2.y = 4.0
    p2.z = 0.0

    assert tu.calcSquaredDistance2d(p1, p2) == pytest.approx(68.0, abs=epsilon)


# Passed
def test_calcElevationAngle():
    p1 = gmsgs.Point()
    p1.x = 10.0
    p1.y = 2.0
    p1.z = 9.0
    p2 = gmsgs.Point()
    p2.x = 2.0
    p2.y = 4.0
    p2.z = 20.0

    assert tu.calcElevationAngle(p1, p2) == pytest.approx(0.927515690739948, abs=epsilon)


# Passed
def test_validateNonEmpty():
    empty_traj = Trajectory()
    with pytest.raises(ValueError, match="Points is empty"):
        tu.validateNonEmpty(empty_traj.points)

    non_empty_traj = gt.generateTestTrajectory(10, 1.0)

    try:
        tu.validateNonEmpty(non_empty_traj.points)
    except ValueError:
        pytest.fail("Unexpected ValueError raised")


# Passed
def test_findNearestIndex():
    empty_traj = Trajectory()
    with pytest.raises(ValueError, match="Points is empty"):
        tu.findNearestIndex(empty_traj.points, gmsgs.Point())

    traj = gt.generateTestTrajectory(10, 1.0)

    # Start point
    assert tu.findNearestIndex(traj.points, tu.createPoint(0.0, 0.0, 0.0)) == 0

    # End point
    assert tu.findNearestIndex(traj.points, tu.createPoint(9.0, 0.0, 0.0)) == 9

    # Boundary conditions
    assert tu.findNearestIndex(traj.points, tu.createPoint(0.5, 0.0, 0.0)) == 0
    assert tu.findNearestIndex(traj.points, tu.createPoint(0.51, 0.0, 0.0)) == 1

    # Point before start point
    assert tu.findNearestIndex(traj.points, tu.createPoint(-4.0, 5.0, 0.0)) == 0

    # Point after end point
    assert tu.findNearestIndex(traj.points, tu.createPoint(100.0, -3.0, 0.0)) == 9

    # Random cases
    assert tu.findNearestIndex(traj.points, tu.createPoint(2.4, 1.3, 0.0)) == 2
    assert tu.findNearestIndex(traj.points, tu.createPoint(4.0, 0.0, 0.0)) == 4


# Passed
def test_findNearestIndex2():
    traj = gt.generateTestTrajectory(10, 1.0, 0.0, 0.0, 0.1)
    # Random cases
    assert tu.findNearestIndex(traj.points, tu.createPoint(5.1, 3.4, 0.0)) == 6


# Passed
def test_findNearestSegmentIndex():
    empty_traj = Trajectory()
    with pytest.raises(ValueError, match="Points is empty"):
        tu.findNearestSegmentIndex(empty_traj.points, gmsgs.Point())

    traj = gt.generateTestTrajectory(10, 1.0)

    # Start point
    assert tu.findNearestSegmentIndex(traj.points, tu.createPoint(0.0, 0.0, 0.0)) == 0

    # End point
    assert tu.findNearestSegmentIndex(traj.points, tu.createPoint(9.0, 0.0, 0.0)) == 8

    # Boundary conditions
    assert tu.findNearestSegmentIndex(traj.points, tu.createPoint(1.0, 0.0, 0.0)) == 0
    assert tu.findNearestSegmentIndex(traj.points, tu.createPoint(1.1, 0.0, 0.0)) == 1

    # Point before start point
    assert tu.findNearestSegmentIndex(traj.points, tu.createPoint(-4.0, 5.0, 0.0)) == 0

    # Point after end point
    assert tu.findNearestSegmentIndex(traj.points, tu.createPoint(100.0, -3.0, 0.0)) == 8

    # Random cases
    assert tu.findNearestSegmentIndex(traj.points, tu.createPoint(2.4, 1.0, 0.0)) == 2
    assert tu.findNearestSegmentIndex(traj.points, tu.createPoint(4.0, 0.0, 0.0)) == 3

    # Two nearest trajectory points are not the edges of the nearest segment.
    sparse_points = [
        tu.createPoint(0.0, 0.0, 0.0),
        tu.createPoint(10.0, 0.0, 0.0),
        tu.createPoint(11.0, 0.0, 0.0),
    ]
    assert tu.findNearestSegmentIndex(sparse_points, tu.createPoint(9.0, 1.0, 0.0)) == 0


# Passed
def test_calcLateralOffset():
    empty_traj = Trajectory()
    with pytest.raises(ValueError, match="Points is empty"):
        tu.calcLateralOffset(empty_traj.points, gmsgs.Point(), True)

    traj = gt.generateTestTrajectory(10, 1.0)

    # Trajectory size is 1
    one_point_traj = gt.generateTestTrajectory(1, 1.0)
    with pytest.raises(RuntimeError, match="Same points are given."):
        tu.calcLateralOffset(one_point_traj.points, gmsgs.Point(), True)

    # Same close points in trajectory
    invalid_traj = gt.generateTestTrajectory(10, 0.0)
    p = tu.createPoint(3.0, 0.0, 0.0)
    with pytest.raises(RuntimeError, match="Same points are given."):
        tu.calcLateralOffset(invalid_traj.points, p, True)

    # Point on trajectory
    tu.calcLateralOffset(traj.points, tu.createPoint(3.1, 0.0, 0.0), True) == pytest.approx(
        0.0, abs=epsilon
    )

    # Point before start point
    tu.calcLateralOffset(traj.points, tu.createPoint(-3.9, 3.0, 0.0), True) == pytest.approx(
        3.0, abs=epsilon
    )

    # Point after end point
    tu.calcLateralOffset(traj.points, tu.createPoint(13.3, -10.0, 0.0), True) == pytest.approx(
        -10.0, abs=epsilon
    )

    # Random cases
    tu.calcLateralOffset(traj.points, tu.createPoint(4.3, 7.0, 0.0), True) == pytest.approx(
        7.0, abs=epsilon
    )
    tu.calcLateralOffset(traj.points, tu.createPoint(1.0, -3.0, 0.0), True) == pytest.approx(
        -3.0, abs=epsilon
    )


# Passed
def test_calcSignedArcLength():
    traj = gt.generateTestTrajectory(10, 1.0)

    # Empty
    empty_traj = Trajectory()
    assert tu.calcSignedArcLength(empty_traj.points, 0, 0) == 0.0

    # Out of range
    with pytest.raises(IndexError):
        tu.calcSignedArcLength(traj.points, 0, len(traj.points) + 1)

    # Same point
    tu.calcSignedArcLength(traj.points, 3, 3) == pytest.approx(0.0, abs=epsilon)

    # Forward
    tu.calcSignedArcLength(traj.points, 0, 3) == pytest.approx(3.0, abs=epsilon)

    # Backward
    tu.calcSignedArcLength(traj.points, 9, 5) == pytest.approx(-4.0, abs=epsilon)


# Passed
def test_calcArcLength():
    traj = gt.generateTestTrajectory(10, 1.0)

    # Empty
    empty_traj = Trajectory()
    assert tu.calcArcLength(empty_traj.points) == 0.0

    # Whole Length
    tu.calcArcLength(traj.points) == pytest.approx(9.0, abs=epsilon)


# Passed
def test_normalizeRadian():
    import math

    # -math.pi <= deg < math.pi
    eps = 0.1
    v_min = -math.pi
    v_mid = 0
    v_max = math.pi

    assert math.isclose(tu.normalizeRadian(v_min - eps), v_max - eps)
    assert math.isclose(tu.normalizeRadian(v_min), v_min)
    assert math.isclose(tu.normalizeRadian(v_mid), v_mid)
    assert math.isclose(tu.normalizeRadian(v_max - eps), v_max - eps)
    assert math.isclose(tu.normalizeRadian(v_max), v_min)

    # 0 <= deg < 2 * math.pi
    v_min = 0
    v_mid = math.pi
    v_max = 2 * math.pi

    assert math.isclose(tu.normalizeRadian(v_min - eps, 0), v_max - eps)
    assert math.isclose(tu.normalizeRadian(v_min, 0), v_min)
    assert math.isclose(tu.normalizeRadian(v_mid, 0), v_mid)
    assert math.isclose(tu.normalizeRadian(v_max - eps, 0), v_max - eps)
    assert math.isclose(tu.normalizeRadian(v_max, 0), v_min)


# Passed
def test_toGeomMsgPt():
    p1 = Point3d(getId(), 1.0, 2.0, 3.0)
    p2 = Point2d(getId(), 1.0, 2.0)
    p3 = gmsgs.Point32()
    p3.x = 1.0
    p3.y = 2.0
    p3.z = 3.0
    p4 = gmsgs.Point()
    p4.x = 1.0
    p4.y = 2.0
    p4.z = 3.0
    p5 = gmsgs.Point()
    p5.x = 1.0
    p5.y = 2.0
    p5.z = 0.0

    assert tu.toGeomMsgPt(p1) == p4
    assert tu.toGeomMsgPt(p2) == p5
    assert tu.toGeomMsgPt(p3) == p4
    assert tu.toGeomMsgPt(p4) == p4


# Passed, tested by output at the same situation.
def test_getLaneletLength3d():
    pass


# Passed
def test_resamplePoseVector():
    path = []
    for i in range(10):
        path.append(createPose(i * 1.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    # same interval
    resampled_path = tu.resamplePoseVector(path, 1.0)
    assert len(resampled_path) == len(path)
    for i in range(len(path)):
        p = resampled_path[i]
        ans_p = path[i]
        p.position.x == pytest.approx(ans_p.position.x, abs=epsilon)
        p.position.y == pytest.approx(ans_p.position.y, abs=epsilon)
        p.position.z == pytest.approx(ans_p.position.z, abs=epsilon)
        p.orientation.x == pytest.approx(ans_p.orientation.x, abs=epsilon)
        p.orientation.y == pytest.approx(ans_p.orientation.y, abs=epsilon)
        p.orientation.z == pytest.approx(ans_p.orientation.z, abs=epsilon)
        p.orientation.w == pytest.approx(ans_p.orientation.w, abs=epsilon)

    # random
    resampled_path = tu.resamplePoseVector(path, 0.5)
    for i in range(len(path)):
        p = resampled_path[i]
        p.position.x == pytest.approx(0.5 * i, abs=epsilon)
        p.position.y == pytest.approx(0.0, abs=epsilon)
        p.position.z == pytest.approx(0.0, abs=epsilon)
        p.orientation.x == pytest.approx(0.0, abs=epsilon)
        p.orientation.y == pytest.approx(0.0, abs=epsilon)
        p.orientation.z == pytest.approx(0.0, abs=epsilon)
        p.orientation.w == pytest.approx(0.0, abs=epsilon)
