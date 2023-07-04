import time
import os
import sys

import numpy as np
import pybullet
from pybullet_utils import bullet_client
# import open3d as o3d
import trimesh
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from transformer import Rotation, Transform


class BtWorld(object):
    """Interface to a PyBullet physics server.

    Attributes:
        dt: Time step of the physics simulation.
        rtf: Real time factor. If negative, the simulation is run as fast as possible.
        sim_time: Virtual time elpased since the last simulation reset.
    """

    def __init__(self, gui=True):
        connection_mode = pybullet.GUI if gui else pybullet.DIRECT
        self.p = bullet_client.BulletClient(connection_mode, options="--background_color_red=255 "
                                                           "--background_color_blue=255 --background_color_green=255"
                                                                     "--width=2080, --height=1080")

        self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_SHADOWS, 0)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_WIREFRAME, 0)

        self.gui = gui
        self.dt = 1.0 / 240.0
        self.solver_iterations = 150

        self.reset()

    def set_gravity(self, gravity):
        self.p.setGravity(*gravity)

    def load_obj(self, obj_path, pose, scale, obj_path_vis):
        body = Body.from_obj(self.p, obj_path, pose, scale, obj_path_vis)
        self.bodies[body.uid] = body
        return body

    def load_urdf(self, urdf_path, pose, scale=1.0):
        body = Body.from_urdf(self.p, urdf_path, pose, scale)
        self.bodies[body.uid] = body
        return body

    def remove_body(self, body):
        self.p.removeBody(body.uid)
        del self.bodies[body.uid]

    def add_constraint(self, *argv, **kwargs):
        """See `Constraint` below."""
        constraint = Constraint(self.p, *argv, **kwargs)
        return constraint

    def add_camera(self, intrinsic, near, far):
        camera = Camera(self.p, intrinsic, near, far)
        return camera

    def get_contacts(self, bodyA):
        points = self.p.getContactPoints(bodyA.uid)
        contacts = []
        for point in points:
            contact = Contact(
                bodyA=self.bodies[point[1]],
                bodyB=self.bodies[point[2]],
                point=point[5],
                normal=point[7],
                depth=point[8],
                force=point[9],
            )
            contacts.append(contact)
        return contacts

    def reset(self):
        self.p.resetSimulation()
        self.p.setPhysicsEngineParameter(
            fixedTimeStep=self.dt, numSolverIterations=self.solver_iterations
        )
        self.bodies = {}
        self.sim_time = 0.0

    def step(self):
        self.p.stepSimulation()
        self.sim_time += self.dt
        if self.gui:
            time.sleep(self.dt)

    def save_state(self):
        return self.p.saveState()

    def restore_state(self, state_uid):
        self.p.restoreState(stateId=state_uid)

    def close(self):
        self.p.disconnect()


class Body(object):
    """Interface to a multibody simulated in PyBullet.

    Attributes:
        uid: The unique id of the body within the physics server.
        name: The name of the body.
        joints: A dict mapping joint names to Joint objects.
        links: A dict mapping link names to Link objects.
    """

    def __init__(self, physics_client, body_uid, obj_path, points_mean=None, obj_name=None):
        if points_mean is None:
            points_mean = [0, 0, 0]
        self.p = physics_client
        self.uid = body_uid
        self.name = self.p.getBodyInfo(self.uid)[1].decode("utf-8")
        self.joints, self.links = {}, {}
        self.points_mean = points_mean
        for i in range(self.p.getNumJoints(self.uid)):
            joint_info = self.p.getJointInfo(self.uid, i)
            joint_name = joint_info[1].decode("utf8")
            self.joints[joint_name] = Joint(self.p, self.uid, i)
            link_name = joint_info[12].decode("utf8")
            self.links[link_name] = Link(self.p, self.uid, i)

        self.obj_name = obj_name
        self.obj_path = obj_path

        if obj_name == 'hand':
            self.p.changeDynamics(self.uid, 2, lateralFriction=100, spinningFriction=100)
            self.p.changeDynamics(self.uid, 3, lateralFriction=100, spinningFriction=100)
            self.p.changeDynamics(self.uid, 4, lateralFriction=100, spinningFriction=100)
            self.p.changeDynamics(self.uid, 5, lateralFriction=100, spinningFriction=100)
            self.p.changeDynamics(self.uid, 6, lateralFriction=100, spinningFriction=100)
            self.p.changeDynamics(self.uid, 7, lateralFriction=100, spinningFriction=100)
            self.p.changeDynamics(self.uid, 8, lateralFriction=100, spinningFriction=100)
            self.p.changeDynamics(self.uid, 9, lateralFriction=100, spinningFriction=100)
            self.p.changeDynamics(self.uid, -1, lateralFriction=100, spinningFriction=100)

        elif obj_name is not None:
            self.p.changeDynamics(self.uid, -1, mass=0.001, lateralFriction=100, spinningFriction=100, rollingFriction=100)
        else:
            self.p.changeDynamics(self.uid, -1, lateralFriction=1)
        self.T_init_inertial = Transform(Rotation.identity(), np.asarray(points_mean))
        self.T_inertial_init = self.T_init_inertial.inverse()

    @classmethod
    def from_urdf(cls, physics_client, urdf_path, pose, scale):
        body_uid = physics_client.loadURDF(
            str(urdf_path),
            pose.translation,
            pose.rotation.as_quat(),
            globalScaling=scale,
        )
        if str(urdf_path).split('/')[-1] == 'bh.urdf':
            object_name = 'hand'
        else:
            object_name = None
        return cls(physics_client, body_uid, str(urdf_path), obj_name=object_name)

    @classmethod
    def from_obj(cls, physics_client, obj_path, pose, scale, obj_path_vis):
        points_mean = list(cls.calculate_mass_center(obj_path))
        if obj_path_vis is not None:
            obj_path = obj_path_vis
        # load obj pose is related to points_mean
        visualShapeId = physics_client.createVisualShape(shapeType=physics_client.GEOM_MESH,
                                                     fileName=str(obj_path),
                                                     # rgbaColor=color,
                                                     specularColor=[0.4, .4, 0],
                                                     meshScale=scale)
        collisionShapeId = physics_client.createCollisionShape(shapeType=physics_client.GEOM_MESH,
                                                  fileName=str(obj_path),
                                                  meshScale=scale)
        objectId = physics_client.createMultiBody(baseMass=0.5,
                                     baseInertialFramePosition=points_mean,
                                     baseCollisionShapeIndex=collisionShapeId,
                                     baseVisualShapeIndex=visualShapeId,
                                     basePosition=pose.translation,
                                     baseOrientation=pose.rotation.as_quat(),
                                     useMaximalCoordinates=True)
        physics_client.resetBasePositionAndOrientation(objectId, pose.translation, pose.rotation.as_quat())

        # obj_name = str.split(str(obj_path), '/')[-1]
        obj_name_all = obj_path.parts[-3]
        obj_name = str.split(obj_name_all, '_')[0]

        return cls(physics_client, objectId, str(obj_path), points_mean, obj_name)

    @classmethod
    def calculate_mass_center(cls, obj_path):
        obj_mesh = trimesh.load_mesh(str(obj_path))
        points_mean = np.mean(obj_mesh.vertices, 0)
        return points_mean

    def get_pose(self):
        pos, ori = self.p.getBasePositionAndOrientation(self.uid)
        return Transform(Rotation.from_quat(ori), np.asarray(pos))

    def set_pose(self, pose):
        self.p.resetBasePositionAndOrientation(
            self.uid, pose.translation, pose.rotation.as_quat()
        )

    def get_velocity(self):
        linear, angular = self.p.getBaseVelocity(self.uid)
        return linear, angular

    def reset_joints(self, joints, force, velocity, kinematics=False):
        target_joints_num = self.p.getNumJoints(self.uid)
        assert len(joints) == target_joints_num

        if not kinematics:
            for num in range(len(joints)):
                self.p.resetJointState(bodyUniqueId=self.uid, jointIndex=num, targetValue=joints[num])
        else:
            for num in range(0, target_joints_num):
                self.p.setJointMotorControl2(
                    self.uid,
                    num,
                    self.p.POSITION_CONTROL,
                    targetPosition=joints[num],
                    targetVelocity=velocity,
                    force=force[num]
                )

    def reset_joints_stable(self, joints):
        target_joints_num = self.p.getNumJoints(self.uid)
        assert len(joints) == target_joints_num

        for num in range(0, target_joints_num):
            self.p.setJointMotorControl2(
                self.uid,
                num,
                self.p.POSITION_CONTROL,
                targetPosition=joints[num],
                force=0
            )

    def reset_joints_force(self, vel_joints, forces):
        for i in range(self.p.getNumJoints(self.uid)):
            self.p.setJointMotorControl2(
                bodyUniqueId=self.uid,
                jointIndex=i,
                controlMode=pybullet.VELOCITY_CONTROL,
                targetVelocity=vel_joints[i],
                force=forces[i]
            )


    def get_joints(self):
        num = self.p.getNumJoints(self.uid)
        for i in range(num):
            print(self.p.getJointInfo(self.uid, i))

class Link(object):
    """Interface to a link simulated in Pybullet.

    Attributes:
        link_index: The index of the joint.
    """

    def __init__(self, physics_client, body_uid, link_index):
        self.p = physics_client
        self.body_uid = body_uid
        self.link_index = link_index

    def get_pose(self):
        link_state = self.p.getLinkState(self.body_uid, self.link_index)
        pos, ori = link_state[0], link_state[1]
        return Transform(Rotation.from_quat(ori), pos)


class Joint(object):
    """Interface to a joint simulated in PyBullet.

    Attributes:
        joint_index: The index of the joint.
        lower_limit: Lower position limit of the joint.
        upper_limit: Upper position limit of the joint.
        effort: The maximum joint effort.
    """

    def __init__(self, physics_client, body_uid, joint_index):
        self.p = physics_client
        self.body_uid = body_uid
        self.joint_index = joint_index

        joint_info = self.p.getJointInfo(body_uid, joint_index)
        self.lower_limit = joint_info[8]
        self.upper_limit = joint_info[9]
        self.effort = joint_info[10]

        self.t_vel = 1  # target velocity for joints when grasping
        self.maxForce = 100  # max force allowed

    def get_position(self):
        joint_state = self.p.getJointState(self.body_uid, self.joint_index)
        return joint_state[0]

    def set_position(self, position, kinematics=False):
        if kinematics:
            self.p.resetJointState(self.body_uid, self.joint_index, position)
        self.p.setJointMotorControl2(
            self.body_uid,
            self.joint_index,
            pybullet.POSITION_CONTROL,
            targetPosition=position,
            force=300
        )

    def set_torque(self, torque):
        self.p.setJointMotorControl2(
            self.body_uid,
            self.joint_index,
            pybullet.TORQUE_CONTROL,
            force=torque
        )

class Constraint(object):
    """Interface to a constraint in PyBullet.

    Attributes:
        uid: The unique id of the constraint within the physics server.
    """

    def __init__(
        self,
        physics_client,
        parent,
        parent_link,
        child,
        child_link,
        joint_type,
        joint_axis,
        parent_frame,
        child_frame,
    ):
        """
        Create a new constraint between links of bodies.

        Args:
            parent:
            parent_link: None for the base.
            child: None for a fixed frame in world coordinates.

        """
        self.p = physics_client
        parent_body_uid = parent.uid
        parent_link_index = parent_link.link_index if parent_link else -1
        child_body_uid = child.uid if child else -1
        child_link_index = child_link.link_index if child_link else -1

        self.uid = self.p.createConstraint(
            parentBodyUniqueId=parent_body_uid,
            parentLinkIndex=parent_link_index,
            childBodyUniqueId=child_body_uid,
            childLinkIndex=child_link_index,
            jointType=joint_type,
            jointAxis=joint_axis,
            parentFramePosition=parent_frame.translation,
            parentFrameOrientation=parent_frame.rotation.as_quat(),
            childFramePosition=child_frame.translation,
            childFrameOrientation=child_frame.rotation.as_quat(),
        )

    def change(self, **kwargs):
        self.p.changeConstraint(self.uid, **kwargs)


class Contact(object):
    """Contact point between two multibodies.

    Attributes:
        point: Contact point.
        normal: Normal vector from ... to ...
        depth: Penetration depth
        force: Contact force acting on body ...
    """

    def __init__(self, bodyA, bodyB, point, normal, depth, force):
        self.bodyA = bodyA
        self.bodyB = bodyB
        self.point = point
        self.normal = normal
        self.depth = depth
        self.force = force


class Camera(object):
    """Virtual RGB-D camera based on the PyBullet camera interface.

    Attributes:
        intrinsic: The camera intrinsic parameters.
    """

    def __init__(self, physics_client, intrinsic, near, far):
        self.intrinsic = intrinsic
        self.near = near
        self.far = far
        self.proj_matrix = _build_projection_matrix(intrinsic, near, far)
        self.p = physics_client

    def render(self, extrinsic):
        """Render synthetic RGB and depth images.

        Args:
            extrinsic: Extrinsic parameters, T_cam_ref.
        """
        # Construct OpenGL compatible view and projection matrices.
        gl_view_matrix = extrinsic.as_matrix()
        gl_view_matrix[2, :] *= -1  # flip the Z axis
        gl_view_matrix = gl_view_matrix.flatten(order="F")
        gl_proj_matrix = self.proj_matrix.flatten(order="F")

        result = self.p.getCameraImage(
            width=self.intrinsic.width,
            height=self.intrinsic.height,
            viewMatrix=gl_view_matrix,
            projectionMatrix=gl_proj_matrix,
            renderer=pybullet.ER_TINY_RENDERER,
        )

        rgb, z_buffer = result[2][:, :, :3], result[3]

        depth = (
            1.0 * self.far * self.near / (self.far - (self.far - self.near) * z_buffer)
        )

        segment_mask = result[4]

        # plt_show_image(rgb, z_buffer, depth, segment_mask)
        return rgb, depth, segment_mask


def plt_show_image(rgb, z_buffer, depth, segment_mask):

    import matplotlib.pyplot as plt

    plt.subplot(2, 2, 1)  # 将窗口分为两行两列四个子图，则可显示四幅图片
    plt.title('rgb')  # 第一幅图片标题
    plt.imshow(rgb)  # 绘制第一幅图片

    plt.subplot(2, 2, 2)  # 第二个子图
    plt.title('z_buffer')  # 第二幅图片标题
    plt.imshow(z_buffer)  # 绘制第二幅图片,且为灰度图

    plt.subplot(2, 2, 3)  # 将窗口分为两行两列四个子图，则可显示四幅图片
    plt.title('depth')  # 第一幅图片标题
    plt.imshow(depth)  # 绘制第一幅图片

    plt.subplot(2, 2, 4)  # 第二个子图
    plt.title('segment_mask')  # 第二幅图片标题
    plt.imshow(segment_mask)  # 绘制第二幅图片,且为灰度图
    plt.show()

def _build_projection_matrix(intrinsic, near, far):
    perspective = np.array(
        [
            [intrinsic.fx, 0.0, -intrinsic.cx, 0.0],
            [0.0, intrinsic.fy, -intrinsic.cy, 0.0],
            [0.0, 0.0, near + far, near * far],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )
    ortho = _gl_ortho(0.0, intrinsic.width, intrinsic.height, 0.0, near, far)
    return np.matmul(ortho, perspective)


def _gl_ortho(left, right, bottom, top, near, far):
    ortho = np.diag(
        [2.0 / (right - left), 2.0 / (top - bottom), -2.0 / (far - near), 1.0]
    )
    ortho[0, 3] = -(right + left) / (right - left)
    ortho[1, 3] = -(top + bottom) / (top - bottom)
    ortho[2, 3] = -(far + near) / (far - near)
    return ortho
