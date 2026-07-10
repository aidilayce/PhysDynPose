__all__ = ['set_pose', 'smpl_to_rbdl', 'rbdl_to_smpl', 'normalize_and_concat', 
'identify_heel_strikes', 'normalize_gait_cycle', 'plot_joint_data', 'print_title', 'Body', 'smpl_to_rbdl_data',
'RefCorrect', 'Core_utils', 'KinematicUtil', 'angle_util' ]


import enum
import torch
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from articulate.math import rotation_matrix_to_euler_angle_np, euler_angle_to_rotation_matrix_np, euler_convert_np, \
    normalize_angle
import rbdl
import copy
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import math

_smpl_to_rbdl = [0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 28, 29, 3, 4, 5, 12, 13, 14, 21, 22, 23, 30, 31, 32, 6, 7, 8,
                 15, 16, 17, 24, 25, 26, 36, 37, 38, 45, 46, 47, 51, 52, 53, 57, 58, 59, 63, 64, 65, 39, 40, 41,
                 48, 49, 50, 54, 55, 56, 60, 61, 62, 66, 67, 68, 33, 34, 35, 42, 43, 44]
_rbdl_to_smpl = [0, 1, 2, 12, 13, 14, 24, 25, 26, 3, 4, 5, 15, 16, 17, 27, 28, 29, 6, 7, 8, 18, 19, 20, 30, 31, 32,
                 9, 10, 11, 21, 22, 23, 63, 64, 65, 33, 34, 35, 48, 49, 50, 66, 67, 68, 36, 37, 38, 51, 52, 53, 39,
                 40, 41, 54, 55, 56, 42, 43, 44, 57, 58, 59, 45, 46, 47, 60, 61, 62]
_rbdl_to_bullet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                   27, 28, 29, 30, 31, 32, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 33, 34, 35,
                   36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 63, 64, 65, 66, 67, 68]
smpl_to_rbdl_data = _smpl_to_rbdl

############ physcap functions ###############

class KinematicUtil():
    def motion_update_specification(self, id_robot, jointIds, qs):
        [p.resetJointState(id_robot, jid, q) for jid, q in zip(jointIds, qs)]
        return 0
    def get_jointIds_Names(self, id_robot):
        jointNamesAll = []
        jointIdsAll = []
        jointNames = []
        jointIds = []
        for j in range(p.getNumJoints(id_robot)):
            info = p.getJointInfo(id_robot, j)
            p.changeDynamics(id_robot, j, linearDamping=0, angularDamping=0)
            jointName = info[1]
            jointType = info[2]
            jointIdsAll.append(j)
            jointNamesAll.append(jointName)
            if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
                jointIds.append(j)
                jointNames.append(jointName)
        return jointIdsAll, jointNamesAll, jointIds, jointNames

class Core_utils():
    def fcn_RotationFromTwoVectors(self,A, B):
        v = np.cross(A, B)
        v = v / np.linalg.norm(v)
        cos = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
        theta = np.arccos(cos)
        wow = Rot.from_rotvec(1.0 * theta * v)
        R = wow.as_matrix()
        return R

    def rotationMatrixToEulerAngles(self,R):
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        return np.array([x, y, z]) # return np.array([z, y, x])

    def get_projected_CoM(self,model, q, qdot, qddot):
        CoM = np.zeros(3)
        rbdl.CalcCenterOfMass(model.model, q, qdot, qddot, CoM)
        CoM_projected = copy.copy(CoM)
        CoM_projected[1] = 0
        return CoM_projected

    def get_J_lth_rth(self,model, q, rbdl_ids):
        l_toe_J6D = np.zeros([6, model.qdot_size])
        l_heel_J6D = np.zeros([6, model.qdot_size])
        r_toe_J6D = np.zeros([6, model.qdot_size])
        r_heel_J6D = np.zeros([6, model.qdot_size])
        rbdl.CalcPointJacobian6D(model.model, q, rbdl_ids["l_toe"], np.array([0., 0., 0.]), l_toe_J6D)
        rbdl.CalcPointJacobian6D(model.model, q, rbdl_ids["l_heel"], np.array([0., 0., 0.]), l_heel_J6D)
        rbdl.CalcPointJacobian6D(model.model, q, rbdl_ids["r_toe"], np.array([0., 0., 0.]), r_toe_J6D)
        rbdl.CalcPointJacobian6D(model.model, q, rbdl_ids["r_heel"], np.array([0., 0., 0.]), r_heel_J6D)

        lth_J6D = np.concatenate((l_toe_J6D, l_heel_J6D), 0)
        rth_J6D = np.concatenate((r_toe_J6D, r_heel_J6D), 0)
        lth_rth_J6D = np.concatenate((lth_J6D, rth_J6D), 0)

        return lth_rth_J6D

    def get_supp_polygon_corners(self,model, q, rbdl_ids):
        l_toe_coord = rbdl.CalcBodyToBaseCoordinates(model.model, q, rbdl_ids["l_toe"], np.zeros(3))
        l_heel_coord = rbdl.CalcBodyToBaseCoordinates(model.model, q, rbdl_ids["l_heel"], np.zeros(3))
        r_toe_coord = rbdl.CalcBodyToBaseCoordinates(model.model, q, rbdl_ids["r_toe"], np.zeros(3))
        r_heel_coord = rbdl.CalcBodyToBaseCoordinates(model.model, q, rbdl_ids["r_heel"], np.zeros(3))

        corners = np.array([r_toe_coord, l_toe_coord, l_heel_coord, r_heel_coord])
        center_of_supprt = np.average(corners, axis=0)
        return (corners - center_of_supprt) + center_of_supprt
        
    def support_polygon_checker(self,target, corners):
        xz = [0, 2]
        target = target[xz]
        corners = corners[:, xz]
        out = np.array([np.cross(corners[i] - target, corners[0] - corners[i]) if i + 1 == len(corners) else np.cross(
            corners[i] - target, corners[i + 1] - corners[i]) for i in range(len(corners))])
        judgement = np.all(out > 0) if out[0] > 0 else np.all(out < 0)
        return judgement

    def isin_flag(self,target_id, contact_id):
        if target_id in contact_id:
            return 1
        else:
            return 0

class angle_util():
    def angle_clean(self,q):
        mod = q % (2 * math.pi)
        if mod >= math.pi:
            return mod - 2 * math.pi
        else:
            return mod
    def modder(self,radian):
        if radian >= 0:
            return radian%(math.pi*2)
        else:
            return radian%(-math.pi*2)

    def positive_rad(self,radian):
        if radian < 0:
            return radian + math.pi*2
        else:
            return radian

    def get_clean_angle(self,radian):
        #returns angle between 0 and 2pi
        return np.array(list(map(lambda x: self.positive_rad(self.modder(x)), radian)))

    def torque_getter(self,target_rad, current_rad):

        if target_rad >= current_rad:
            A = copy.copy(current_rad)
            B = copy.copy(target_rad)
        else:
            A = copy.copy(target_rad)
            B = copy.copy(current_rad)

        d = min(abs(B - A), abs(math.pi * 2 - B + A))
 
        if B - A >= math.pi:
            if B == target_rad:
                return -d
            elif B == current_rad:
                return d
            else:
                print("does not match any patterns")
                return 0

        elif B - A < math.pi:
            if B == target_rad:
                return d
            elif B == current_rad:
                return -d
            else:
                print("does not match any patterns")
                return 0
        else:
            print("does not match any patterns")
            return 0

    def compute_difference(self,target_rads, current_rads): 
        target_rads = self.get_clean_angle(target_rads)
        current_rads = self.get_clean_angle(current_rads) 
        torques = [self.torque_getter(target_rad, current_rad) for target_rad,current_rad in zip(target_rads, current_rads)]
        if math.pi/2<=current_rads[2] and current_rads[2]<= 3*math.pi/2:
            torques[0] = -torques[0]
        if math.pi / 2 <= current_rads[2] and current_rads[2] <= 3 * math.pi / 2:
            torques[1] = -torques[1] 
        return np.array(torques)

CU = Core_utils()
class RefCorrect():
    def __init__(self,stationary_flags):
        self.knee_correct_count=0
        self.inter_flag=0
        self.inter_count=0
        self.pre_correct_flag=0
        self.stationary_flags=stationary_flags

    def ref_motion_correction(self,id_robot_vnect,count,target_base_ori,target_base_ori_original,judgement,q,q_ref):

        end = p.getLinkState(id_robot_vnect, 47)[0]# 47 top torso
        basePos, _ = p.getBasePositionAndOrientation(id_robot_vnect)
        vec_from = np.array(end) - np.array(basePos)
        vec_to = np.array([0, 1, 0])
        R_correct = CU.fcn_RotationFromTwoVectors(vec_from, vec_to)

        """  this R conversion is necessary due to the difference of euler convention """
        eulerR = CU.rotationMatrixToEulerAngles(R_correct)
        r_calib = Rot.from_euler('xyz', eulerR)
        R_calib = r_calib.as_matrix()

        r3 = Rot.from_euler('xyz', target_base_ori) #return np.array([z, y, x])
        mat = r3.as_matrix()
        mat = np.dot(mat, R_calib)
        r4 = Rot.from_matrix(mat)
        target_vec = r4.as_euler('xyz') #target_vec = r4.as_euler('zyx')

        key_times = [0, 1]
        current_r = Rot.from_euler('xyz', target_base_ori_original) #Rot.from_euler('zyx', np.array(list(map(AU.angle_clean, target_base_ori_original))))
        xyz_base = current_r.as_euler('xyz')
        r1 = Rot.from_euler('xyz', [xyz_base, target_vec]) # -target_base_ori[1]+
        # current_r = Rot.from_euler('zyx', np.array(list(map(AU.angle_clean, target_base_ori_original))))
        # xyz_base = current_r.as_euler('zyx')
        # r1 = Rot.from_euler('zyx', [xyz_base, target_vec])

        slerp = Slerp(key_times, r1)
        times = np.arange(0, 1.0, 0.05)
        interp_rots = slerp(times)
        eulers = interp_rots.as_euler('xyz') # eulers = interp_rots.as_euler('zyx')

        if self.stationary_flags[count] and not judgement:
            self.inter_flag = 1

        if not self.stationary_flags[count]:
            self.inter_flag = 0
            self.inter_count = 0
            self.pre_correct_flag = 0

        if self.inter_flag and abs(q[3]) < 0.2:
            # p.addUserDebugText(" im in", [0, 1, 0], [0, 0, 1], textSize=3, replaceItemUniqueId=30)
            target_base_ori = eulers[self.inter_count]
            if abs(target_base_ori[0] - q[0]) > 2.7 or abs(target_base_ori[2] - q[2]) > 2.7:
                target_base_ori[0] += math.pi
                target_base_ori[1] = math.pi - target_base_ori[1]
                target_base_ori[2] += math.pi
            if self.inter_count != len(eulers) - 1 and not judgement:
                self.inter_count += 1
            if abs(q[3]) < 0.1 and not judgement:
               # p.addUserDebugText(str(self.knee_correct_count), [0, 1.1, 0], [1, 0, 0], textSize=3, replaceItemUniqueId=31)
                w = 0.01
                q_ref[3] /= w * self.knee_correct_count + 1
                q_ref[4] /= w * self.knee_correct_count + 1
                q_ref[2] /= w * self.knee_correct_count + 1
                q_ref[1] /= w * self.knee_correct_count + 1
                q_ref[14] /= w * self.knee_correct_count + 1
                q_ref[13] /= w * self.knee_correct_count + 1
                q_ref[12] /= w * self.knee_correct_count + 1
                q_ref[11] /= w * self.knee_correct_count + 1
                self.knee_correct_count += 1
            else:
                self.knee_correct_count = 0
                #p.addUserDebugText(" ", [0, 1, 0], [0, 0, 1], textSize=3, replaceItemUniqueId=30)
        else:
            self.knee_correct_count = 0 
    
        return target_base_ori,q_ref



##############################################

def set_pose(id_robot, q):
    r"""
    Set the robot configuration.
    """
    p.resetJointStatesMultiDof(id_robot, list(range(1, p.getNumJoints(id_robot))), q[6:][_rbdl_to_bullet].reshape(-1, 1))
    glb_rot = p.getQuaternionFromEuler(euler_convert_np(q[3:6], 'zyx', 'xyz')[[2, 1, 0]])
    p.resetBasePositionAndOrientation(id_robot, q[:3], glb_rot)


def smpl_to_rbdl(poses, trans):
    r"""
    Convert smpl poses and translations to robot configuration q. (numpy, batch)

    :param poses: Array that can reshape to [n, 24, 3, 3].
    :param trans: Array that can reshape to [n, 3].
    :return: Ndarray in shape [n, 75] (3 root position + 72 joint rotation).
    """
    poses = np.array(poses).reshape(-1, 24, 3, 3)
    trans = np.array(trans).reshape(-1, 3)
    euler_poses = rotation_matrix_to_euler_angle_np(poses[:, 1:], 'XYZ').reshape(-1, 69)
    euler_glbrots = rotation_matrix_to_euler_angle_np(poses[:, :1], 'xyz').reshape(-1, 3)
    euler_glbrots = euler_convert_np(euler_glbrots[:, [2, 1, 0]], 'xyz', 'zyx')
    qs = np.concatenate((trans, euler_glbrots, euler_poses[:, _smpl_to_rbdl]), axis=1)
    qs[:, 3:] = normalize_angle(qs[:, 3:])
    return qs


def rbdl_to_smpl(qs):
    r"""
    Convert robot configuration q to smpl poses and translations. (numpy, batch)

    :param qs: Ndarray that can reshape to [n, 75] (3 root position + 72 joint rotation).
    :return: Poses ndarray in shape [n, 24, 3, 3] and translation ndarray in shape [n, 3].
    """
    qs = qs.reshape(-1, 75)
    trans, euler_glbrots, euler_poses = qs[:, :3], qs[:, 3:6], qs[:, 6:][:, _rbdl_to_smpl]
    euler_glbrots = euler_convert_np(euler_glbrots, 'zyx', 'xyz')[:, [2, 1, 0]]
    glbrots = euler_angle_to_rotation_matrix_np(euler_glbrots, 'xyz').reshape(-1, 1, 3, 3)
    poses = euler_angle_to_rotation_matrix_np(euler_poses, 'XYZ').reshape(-1, 23, 3, 3)
    poses = np.concatenate((glbrots, poses), axis=1)
    return poses, trans


def normalize_and_concat(glb_acc, glb_rot):
    glb_acc = glb_acc.view(-1, 6, 3)
    glb_rot = glb_rot.view(-1, 6, 3, 3)
    acc = torch.cat((glb_acc[:, :5] - glb_acc[:, 5:], glb_acc[:, 5:]), dim=1).bmm(glb_rot[:, -1])
    ori = torch.cat((glb_rot[:, 5:].transpose(2, 3).matmul(glb_rot[:, :5]), glb_rot[:, 5:]), dim=1)
    data = torch.cat((acc.flatten(1), ori.flatten(1)), dim=1)
    return data

# Identify heel strikes using contact flags for determining gait cycles
def identify_heel_strikes(contact_flags):
    start_left_foot_strikes = []
    end_left_foot_strikes = []
    start_right_foot_strikes = []
    end_right_foot_strikes = []

    # if contacts start from 1, add the first frame (wih index 0) as a heel strike
    if contact_flags[0, 0] == 1:  # Left foot on
        start_left_foot_strikes.append(0)
    elif contact_flags[0, 1] == 1:  # Right foot on
        start_right_foot_strikes.append(0)

    for i in range(1, len(contact_flags)):
        if contact_flags[i-1, 0] == 0 and contact_flags[i, 0] == 1:  # Left foot on
            start_left_foot_strikes.append(i)
        elif contact_flags[i-1, 0] == 1 and contact_flags[i, 0] == 0:  # Left foot off
            end_left_foot_strikes.append(i-1)
        elif contact_flags[i-1, 1] == 0 and contact_flags[i, 1] == 1:  # Right foot on
            start_right_foot_strikes.append(i)
        elif contact_flags[i-1, 1] == 1 and contact_flags[i, 1] == 0:  # Right foot off
            end_right_foot_strikes.append(i-1)

    # if contacts end with 1, add the last frame as a heel strike
    if contact_flags[-1, 0] == 1:
        # end_left_foot_strikes.append(len(contact_flags) - 1)
        start_left_foot_strikes.pop() # remove the last stride as it will not be completed
    
    if contact_flags[-1, 1] == 1:
        # end_right_foot_strikes.append(len(contact_flags) - 1)
        start_right_foot_strikes.pop()

    # remove 1 frame length strikes
    remove_lfoot = []
    remove_rfoot = []
    for j in range(len(start_left_foot_strikes)):
        if start_left_foot_strikes[j] == end_left_foot_strikes[j]:
            remove_lfoot.append(j)
    start_left_foot_strikes = np.delete(start_left_foot_strikes, remove_lfoot)
    end_left_foot_strikes = np.delete(end_left_foot_strikes, remove_lfoot)

    for j in range(len(start_right_foot_strikes)):
        if start_right_foot_strikes[j] == end_right_foot_strikes[j]:
            remove_rfoot.append(j)
    start_right_foot_strikes = np.delete(start_right_foot_strikes, remove_rfoot)
    end_right_foot_strikes = np.delete(end_right_foot_strikes, remove_rfoot)

    return start_left_foot_strikes, end_left_foot_strikes, start_right_foot_strikes, end_right_foot_strikes

# Normalize gait cycles
def normalize_gait_cycle(data, start_heel_strikes, end_heel_strikes, only_x_axis=False):
    gait_cycles = []
    for i in range(len(start_heel_strikes) - 1):
        start, end = start_heel_strikes[i], end_heel_strikes[i]
        cycle = data[start:end+1]
        num_points = 500
        if only_x_axis == True:
            cycle = cycle[:, 0] # torque around x-axis TODO: check this
            normalized_cycle = np.interp(np.linspace(0, 100, num_points), np.linspace(0, 100, len(cycle)), cycle)
        else:
            normalized_cycle = np.zeros((num_points, cycle.shape[1]))
            for dim in range(cycle.shape[1]):
                normalized_cycle[:, dim] = np.interp(np.linspace(0, 100, num_points), np.linspace(0, 100, len(cycle)), cycle[:, dim])
        gait_cycles.append(normalized_cycle)
    return np.array(gait_cycles)


# Plot the graphs
def plot_joint_data(mean_data, std_data, joint_list):
    fig, axes = plt.subplots(len(joint_list), 1, figsize=(12, len(joint_list) * 4), sharex=True)
    fig.suptitle('Joint Torques')
    for i, joint in enumerate(joint_list):
        ax = axes[i]
        ax.plot(mean_data[i], label=f'{joint} Mean')
        ax.fill_between(range(len(mean_data[i])), mean_data[i] - std_data[i], mean_data[i] + std_data[i], alpha=0.2)
        ax.set_ylabel('Torque (Nm/kg)')
        ax.legend()
        ax.grid(True)
    plt.xlabel('Normalized Gait Cycle (%)')
    plt.show()


def print_title(s):
    print('============ %s ============' % s)


class Body(enum.Enum):
    r"""
    Prefix L = left; Prefix R = right.
    """
    ROOT = 2
    PELVIS = 2
    SPINE = 2
    LHIP = 5
    RHIP = 17
    SPINE1 = 29
    LKNEE = 8
    RKNEE = 20
    SPINE2 = 32
    LANKLE = 11
    RANKLE = 23
    SPINE3 = 35
    LFOOT = 14
    RFOOT = 26
    NECK = 68
    LCLAVICLE = 38
    RCLAVICLE = 53
    HEAD = 71
    LSHOULDER = 41
    RSHOULDER = 56
    LELBOW = 44
    RELBOW = 59
    LWRIST = 47
    RWRIST = 62
    LHAND = 50
    RHAND = 65
