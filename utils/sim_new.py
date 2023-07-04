from pathlib import Path
import enum
import numpy as np
import pybullet
import os
import sys

# from grasp import Label
PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_BASE_DIR, 'utils'))
import btsim
from transformer import Rotation, Transform

from scipy.spatial import ConvexHull, distance
from math import sqrt, pi
from numpy import cross
from pyquaternion import Quaternion


# def get_obj_info(oID, p): #TODO: what about not mesh objects?
#     obj_data = p.getCollisionShapeData(oID, -1)[0]
#     geometry_type = obj_data[2]
#     #print("geometry type: " + str(geometry_type))
#     dimensions = obj_data[3]
#     #print("dimensions: "+ str(dimensions))
#     local_frame_pos = obj_data[5]
#     #print("local frome position: " + str(local_frame_pos))
#     local_frame_orn = obj_data[6]
#     #print("local frame oren: " + str(local_frame_orn))
#     diagonal = sqrt(dimensions[0]**2+dimensions[1]**2+dimensions[2]**2)
#     #print("diagonal: ", diagonal)
#     max_radius = diagonal/2
#     return local_frame_pos, max_radius

def gws(rID, oID, p):
    print("eval gws")
    # local_frame_pos, max_radius = get_obj_info(oID, p)
    # sim uses center of mass as a reference for the Cartesian world transforms in getBasePositionAndOrientation
    obj_pos, obj_orn = p.getBasePositionAndOrientation(oID)
    force_torque = []
    contact_points = p.getContactPoints(rID, oID)
    for point in contact_points:
        contact_pos = point[6]
        normal_vector_on_obj = point[7]
        normal_force_on_obj = point[9]
        force_vector = np.array(normal_vector_on_obj) * normal_force_on_obj

        radius_to_contact = np.array(contact_pos) - np.array(obj_pos)
        torque_numerator = cross(radius_to_contact, force_vector)
        torque_vector = torque_numerator

        # force_mag = norm(normal_vector_size)
        # torque_mag = norm(torque)
        # lmda = 1
        force_torque.append(np.concatenate([force_vector, torque_vector]))

    return force_torque


def get_new_normals(force_vector, normal_force, sides, radius):
    return_vectors = []
    # get arbitrary vector to get cross product which should be orthogonal to both
    vector_to_cross = np.array((force_vector[0] + 1, force_vector[1] + 2, force_vector[2] + 3))
    orthg = np.cross(force_vector, vector_to_cross)
    orthg_vector = (orthg / np.linalg.norm(orthg)) * radius
    rot_angle = (2 * pi) / sides
    split_force = normal_force / sides

    for side_num in range(0, sides):
        rotated_orthg = Quaternion(axis=force_vector, angle=(rot_angle * side_num)).rotate(orthg_vector)
        new_vect = force_vector + np.array(rotated_orthg)
        norm_vect = (new_vect / np.linalg.norm(new_vect)) * split_force
        return_vectors.append(norm_vect)

    return return_vectors


def gws_pyramid_extension(rID, oID, p, normal_force_on_obj=5, pyramid_sides=6, pyramid_radius=.01):
    print("gws pyramid creation")
    # local_frame_pos, max_radius = get_obj_info(oID)
    # sim uses center of mass as a reference for the Cartesian world transforms in getBasePositionAndOrientation
    obj_pos, obj_orn = p.getBasePositionAndOrientation(oID)
    force_torque = []
    contact_points = p.getContactPoints(rID, oID)
    for point in contact_points:
        contact_pos = point[6]
        normal_vector_on_obj = point[7]
        normal_force_on_obj = normal_force_on_obj
        force_vector = np.array(normal_vector_on_obj) * normal_force_on_obj
        if np.linalg.norm(force_vector) > 0:
            new_vectors = get_new_normals(force_vector, normal_force_on_obj, pyramid_sides, pyramid_radius)

            radius_to_contact = np.array(contact_pos) - np.array(obj_pos)

            for pyramid_vector in new_vectors:
                torque_numerator = cross(radius_to_contact, pyramid_vector)
                torque_vector = torque_numerator
                force_torque.append(np.concatenate([pyramid_vector, torque_vector]))

    return force_torque


def eplison(force_torque):
    """
    get qhull of the 6 dim vectors [fx, fy, fz, tx, ty, tz] created by gws (from contact points)
    get the distance from centroid of the hull to the closest vertex
    """
    hull = ConvexHull(points=force_torque)
    centroid = []
    for dim in range(0, 6):
        centroid.append(np.mean(hull.points[hull.vertices, dim]))
    shortest_distance = 500000000
    closest_point = None
    for point in force_torque:
        point_dist = distance.euclidean(centroid, point)
        if point_dist < shortest_distance:
            shortest_distance = point_dist
            closest_point = point

    return shortest_distance


# from utils.draw_frames import *

class Label(enum.IntEnum):
    FAILURE = 0  # grasp execution failed due to collision or slippage
    SUCCESS = 1  # object was successfully removed


class ClutterRemovalSim(object):
    def __init__(self, object_set, gripper_path, gui=True, seed=None):

        if gripper_path is None:
            print('gripper is not exist!')
            exit()
        self.urdf_root = Path(os.path.join(PROJECT_BASE_DIR, 'dataset/urdfs'))
        self.object_set = object_set  # barret_object
        self.object_names = ['002_master_chef_can',
                             '003_cracker_box',
                             '004_sugar_box',
                             '005_tomato_soup_can',
                             '006_mustard_bottle',
                             '008_pudding_box',
                             '010_potted_meat_can',
                             '019_pitcher_base',
                             '021_bleach_cleanser',
                             '025_mug',
                             '035_power_drill',
                             '036_wood_block']
        self.discover_objects_obj()

        self.gripper_path = gripper_path
        # self.global_scaling = [0.001, 0.001, 0.001]  # mm->m
        self.global_scaling = [1, 1, 1]  # m
        self.gui = gui

        rot_z_180 = np.identity(4)
        rot_z_180[1, 1] = -1
        rot_z_180[0, 0] = -1
        self.rot_z_180 = Transform.from_matrix(rot_z_180)

        self.rng = np.random.RandomState(seed) if seed else np.random
        self.world = btsim.BtWorld(self.gui)
        self.gripper = Gripper(self.world, self.gripper_path)
        self.size = 1

    @property
    def num_objects(self):
        return max(0, self.world.p.getNumBodies() - 1)  # remove table from body count

    def discover_objects_obj(self):
        root = self.urdf_root / self.object_set / 'ycb_meshes'
        self.object_objs = [f / 'google_16k' / 'model_watertight_5000def.obj'
                            for f in root.iterdir() if f.name in self.object_names]
        for obj in self.object_objs:
            if not obj.exists():
                print('obj is not exist')
                exit()

    def save_state(self):
        self._snapshot_id = self.world.save_state()

    def restore_state(self):
        self.world.restore_state(self._snapshot_id)

    def reset(self, obj_transforms, obj_names_in_scene, vis=False):

        self.world.reset()
        self.world.set_gravity([0.0, 0.0, -9.81])
        # self.draw_workspace()

        if self.gui:
            self.world.p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=0.0,
                cameraPitch=-45,
                cameraTargetPosition=[0, 0, 0],
            )

        table_height = 0
        self.place_table(table_height)
        self.generate_pile_scene_obj(obj_transforms, obj_names_in_scene, vis)

    def place_table(self, height):
        urdf = self.urdf_root / "setup" / "table.urdf"
        pose = Transform(Rotation.identity(), [0, 0, height])
        self.world.load_urdf(urdf, pose, scale=1)

    def generate_pile_scene_obj(self, obj_transforms, obj_names_in_scene, vis):
        self.map_objname_objid = {}
        for obj_transform, obj_name_in_scene in zip(obj_transforms, obj_names_in_scene):
            obj_name = obj_name_in_scene.split('#')[0]
            obj_path = \
                self.urdf_root / self.object_set / 'ycb_meshes' / obj_name / 'google_16k' / 'model_watertight_5000def.obj'
            pose = Transform.from_matrix(obj_transform)
            obj_path_vis = None
            if vis:
                obj_path_vis = \
                    self.urdf_root / self.object_set / 'ycb_meshes' / obj_name / 'google_16k' / 'textured.obj'
            obj_body = self.world.load_obj(obj_path, pose, scale=self.global_scaling, obj_path_vis=obj_path_vis)
            self.map_objname_objid[obj_name_in_scene] = obj_body.uid

    def load_gripper(self, T_world_body, joints):
        self.gripper.reset(T_world_body, joints)

    def execute_grasp_contact(self, obj_body, scene_filtered_grasp, scene_filtered_joint,
                              remove=False, allow_contact=False, upper_dis=0.2):
        current_obj_body = obj_body
        current_obj_pose = current_obj_body.get_pose()

        T_world_grasp = Transform.from_matrix(scene_filtered_grasp) * self.rot_z_180
        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp

        T_worldBefore_worldAfter = Transform(Rotation.identity(), [0.0, 0.0, upper_dis])
        T_world_retreat = T_worldBefore_worldAfter * T_world_grasp

        origin_joints = list(scene_filtered_joint)
        grasp_joints = [0, 0] + origin_joints[6:] + origin_joints[:6]
        main_joints_index = self.gripper.main_joints
        main_joints = [0 for _ in range(len(grasp_joints))]
        for i in main_joints_index:
            main_joints[i] = grasp_joints[i]

        joints = [joint for joint in grasp_joints]
        self.gripper.reset(T_world_pregrasp, main_joints)
        # self.gripper.frame_draw_manager = self.gripper.loadframe()

        if self.gripper.detect_contact():
            result = Label.FAILURE
        else:
            self.gripper.move_body_xyz(T_world_grasp, abort_on_contact=True)
            if self.gripper.detect_contact() and not allow_contact:
                result = Label.FAILURE
            else:
                grasp_joints_add = [0, 0, 0.09, 0.03, 0, 0.09, 0.03, 0, 0.09, 0.03]
                for i in range(len(joints)):
                    joints[i] += grasp_joints_add[i]
                self.gripper.move(joints)

                self.gripper.move_body_xyz(T_world_retreat, abort_on_contact=False)
                # vel_joints = [0, 0, 1, 0.3, 0, 1, 0.3, 0, 1, 0.3]
                # self.gripper.move_stable(vel_joints)
                if self.check_success(current_obj_body, current_obj_pose, upper_dis):
                    result = Label.SUCCESS
                    if remove:
                        contacts = self.world.get_contacts(self.gripper.body)
                        self.world.remove_body(contacts[0].bodyB)
                else:
                    result = Label.FAILURE
        # time.sleep(1)
        self.world.remove_body(self.gripper.body)

        if remove:
            self.remove_and_wait()
        return result

    def execute_grasp_test(self, grasp_transform, joint, upper_dis=0.2):

        object_poses = {}
        for body in self.world.bodies:
            current_obj_pose = self.world.bodies[body].get_pose()
            object_poses[self.world.bodies[body].uid] = current_obj_pose

        joint = list(joint)
        grasp_joints = [0, 0] + joint[6:] + joint[:6]

        T_world_grasp = Transform.from_matrix(grasp_transform) * self.rot_z_180

        T_worldBefore_worldAfter = Transform(Rotation.identity(), [0.0, 0.0, upper_dis])
        T_world_retreat = T_worldBefore_worldAfter * T_world_grasp

        grasp_joints_minus = [0, 0, 0.09, 0.03, 0, 0.09, 0.03, 0, 0.09, 0.03]
        for i in range(len(grasp_joints)):
            grasp_joints[i] -= grasp_joints_minus[i]
        self.gripper.reset(T_world_grasp, grasp_joints)

        for i in range(len(grasp_joints)):
            grasp_joints[i] += grasp_joints_minus[i]
        self.gripper.move(grasp_joints)

        self.gripper.move_body_xyz(T_world_retreat, abort_on_contact=False)

        result = Label.FAILURE
        prepared_remove_body = None
        object_record_id = None
        for body_id in object_poses.keys():
            object_pose_old = object_poses[body_id]
            body = self.world.bodies[body_id]

            if self.check_success(body, object_pose_old, upper_dis):
                prepared_remove_body = body
                object_record_id = body_id
                result = Label.SUCCESS
                break
            else:
                result = Label.FAILURE
        self.world.remove_body(self.gripper.body)

        if int(result) == 1:
            self.world.remove_body(prepared_remove_body)
            self.save_state()

        return result, object_record_id

    def execute_grasp_quality(self, grasp_transform, joint, upper_dis=0.2):
        object_poses = {}
        for body in self.world.bodies:
            current_obj_pose = self.world.bodies[body].get_pose()
            object_poses[self.world.bodies[body].uid] = current_obj_pose

        joint = list(joint)
        grasp_joints = [0, 0] + joint[6:] + joint[:6]

        T_world_grasp = Transform.from_matrix(grasp_transform) * self.rot_z_180

        T_worldBefore_worldAfter = Transform(Rotation.identity(), [0.0, 0.0, upper_dis])
        T_world_retreat = T_worldBefore_worldAfter * T_world_grasp

        grasp_joints_minus = [0, 0, 0.12, 0.03, 0, 0.12, 0.03, 0, 0.12, 0.03]
        # grasp_joints_minus = [0, 0, 3, 1, 0, 3, 1, 0, 3, 1]
        for i in range(len(grasp_joints)):
            grasp_joints[i] -= grasp_joints_minus[i]
        # down = -0.01
        # T_offset = Transform(Rotation.identity(), [0.0, 0.0, down])
        # T_world_grasp = T_offset * T_world_grasp

        self.gripper.reset(T_world_grasp, grasp_joints)

        for i in range(len(grasp_joints)):
            grasp_joints[i] += grasp_joints_minus[i]
        self.gripper.move(grasp_joints)

        # self.gripper.move_stable_1(grasp_joints)

        self.gripper.move_body_xyz(T_world_retreat, abort_on_contact=False)

        short_distance = None
        result = Label.FAILURE
        prepared_remove_body = None
        object_record_id = None
        for body_id in object_poses.keys():
            object_pose_old = object_poses[body_id]
            body = self.world.bodies[body_id]

            if self.check_success(body, object_pose_old, upper_dis):
                prepared_remove_body = body
                object_record_id = body_id
                result = Label.SUCCESS
                gripper_id = self.gripper.body.uid
                object_id = body.uid
                force_torque = gws_pyramid_extension(gripper_id, object_id, self.world.p)
                try:
                    short_distance = eplison(force_torque)
                except:
                    print("can't compute short distance")
                    short_distance = 0
                break
            else:
                result = Label.FAILURE
        self.world.remove_body(self.gripper.body)

        if int(result) == 1:
            self.world.remove_body(prepared_remove_body)
            self.save_state()

        return result, object_record_id, short_distance

    def remove_and_wait(self):
        # wait for objects to rest while removing bodies that fell outside the workspace
        removed_object = True
        while removed_object:
            self.wait_for_objects_to_rest()
            removed_object = self.remove_objects_outside_workspace()

    def wait_for_objects_to_rest(self, timeout=5.0, tol=0.01):
        timeout = self.world.sim_time + timeout
        objects_resting = False
        while not objects_resting and self.world.sim_time < timeout:
            # simulate a quarter of a second
            for _ in range(60):
                self.world.step()
            # check whether all objects are resting
            objects_resting = True
            for _, body in self.world.bodies.items():
                if np.linalg.norm(body.get_velocity()) > tol:
                    objects_resting = False
                    break

    def remove_objects_outside_workspace(self):
        removed_object = False
        for body in list(self.world.bodies.values()):
            xyz = body.get_pose().translation
            if np.any(xyz < 0.0) or np.any(xyz > self.size):
                self.world.remove_body(body)
                removed_object = True
        return removed_object

    def check_success(self, current_obj_body, before_obj_pose, upper_dis):
        current_obj_pose = current_obj_body.get_pose()
        diff = current_obj_pose.translation - before_obj_pose.translation
        if upper_dis * 2 / 3 < diff[2]:
            return True
        else:
            return False

    def get_sim_bodys_pose_from_camera(self, extrinsic):
        objects_pose_in_camera = []
        for _, body in self.world.bodies.items():
            if body.obj_name == None:
                continue
            obj_pose_in_camera = extrinsic * body.get_pose()
            objects_pose_in_camera.append(obj_pose_in_camera.as_matrix())
        return np.asarray(objects_pose_in_camera)

    def get_sim_bodys_pose_from_world(self):
        objects_pose_in_world = []
        for _, body in self.world.bodies.items():
            if body.obj_name == None:
                continue
            obj_pose_in_world = body.get_pose()
            objects_pose_in_world.append(obj_pose_in_world.as_matrix())
        return np.asarray(objects_pose_in_world)

    def get_sim_bodys_uid(self):
        object_uid_dic = {}
        for _, body in self.world.bodies.items():
            if body.obj_name == None:
                continue
            object_uid_dic[body.obj_path] = body.uid
        return object_uid_dic


class Gripper(object):
    """Simulated Panda hand."""

    def __init__(self, world, gripper_path):
        self.world = world
        self.urdf_path = Path(gripper_path)

        self.t_vel = .01  # target velocity for joints when grasping
        # self.maxForce = [0, 0, 200, 10000, 5000, 200, 10000, 5000, 200, 5000]  # max force allowed
        self.maxForce = [0, 0, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]  # max force allowed
        # self.maxForce = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # max force allowed
        self.time_limit = 3.0  # how long given to find a grasp

        self.main_joints = [4, 7]  # the main joints in gripper
        self.sub_joints = [2, 3, 5, 6, 8, 9]

        # self.frame_draw_manager = None

    def reset(self, T_world_body, joints):
        # T_world_body = T_world_tcp * self.T_tcp_body
        self.body = self.world.load_urdf(self.urdf_path, T_world_body)
        self.body.set_pose(T_world_body)
        self.body.reset_joints(joints, force=self.maxForce, velocity=self.t_vel, kinematics=False)
        # self.body.get_joints()
        self.constraint = self.world.add_constraint(
            self.body,
            None,
            None,
            None,
            pybullet.JOINT_FIXED,
            [0.0, 0.0, 0.0],
            Transform.identity(),
            T_world_body,
        )
        # self.update_tcp_constraint(T_world_tcp)

    def move_1(self, joints):

        self.hand_base_joint = self.body.joints['hand_joint']
        self.base_joint = self.body.joints['bh_base_joint']
        self.joint32 = self.body.joints['bh_j32_joint']
        self.joint33 = self.body.joints['bh_j33_joint']
        self.joint11 = self.body.joints['bh_j11_joint']
        self.joint12 = self.body.joints['bh_j12_joint']
        self.joint13 = self.body.joints['bh_j13_joint']
        self.joint21 = self.body.joints['bh_j21_joint']
        self.joint22 = self.body.joints['bh_j22_joint']
        self.joint23 = self.body.joints['bh_j23_joint']

        self.hand_base_joint.set_position(joints[0])
        self.base_joint.set_position(joints[1])
        self.joint32.set_position(joints[2])
        self.joint33.set_position(joints[3])
        self.joint11.set_position(joints[4])
        self.joint12.set_position(joints[5])
        self.joint13.set_position(joints[6])
        self.joint21.set_position(joints[7])
        self.joint22.set_position(joints[8])
        self.joint23.set_position(joints[9])
        for _ in range(int(5 / self.world.dt)):
            self.world.step()

    def move(self, joints):
        self.body.reset_joints(joints, force=self.maxForce, velocity=self.t_vel, kinematics=True)
        for _ in range(int(1 / self.world.dt)):
            self.world.step()

    def move_stable_1(self, joints):
        self.body.reset_joints_stable(joints)
        for _ in range(int(1 / self.world.dt)):
            self.world.step()

    def move_stable(self, vel_joints):
        self.body.reset_joints_force(vel_joints)
        for _ in range(int(2 / self.world.dt)):
            self.world.step()

    def detect_contact(self):
        if self.world.get_contacts(self.body):
            return True
        else:
            return False

    def move_body_xyz(self, target, eef_step=0.002, vel=0.10, abort_on_contact=True):

        # frame_draw_manager = self.frame_draw_manager

        T_world_body = self.body.get_pose()

        diff = target.translation - T_world_body.translation
        n_steps = int(np.linalg.norm(diff) / eef_step)
        dist_step = diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel

        for n_step in range(n_steps):
            T_world_body.translation += dist_step
            self.update_tcp_constraint(T_world_body)
            for _ in range(int(dur_step / self.world.dt)):
                self.world.step()
                # frame_draw_manager.update()
            if abort_on_contact and self.detect_contact():
                return

    def update_tcp_constraint(self, T_world_body):
        self.constraint.change(
            jointChildPivot=T_world_body.translation,
            jointChildFrameOrientation=T_world_body.rotation.as_quat(),
        )

    def getPosition(self):
        return self.body.get_pose()
