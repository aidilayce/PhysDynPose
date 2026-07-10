import articulate as art
import numpy as np
import pybullet as p
import torch
from articulate.utils.bullet import *
from articulate.utils.rbdl import *
from qpsolvers import solve_qp

from config import paths
from utils import *


class PhysicsOptimizer:
    test_contact_joints = [
        "LHIP",
        "RHIP",
        "SPINE1",
        "LKNEE",
        "RKNEE",
        "SPINE2",
        "SPINE3",
        "LSHOULDER",
        "RSHOULDER",
        "HEAD",
        "LELBOW",
        "RELBOW",
        "LHAND",
        "RHAND",
        "LFOOT",
        "RFOOT",
    ]

    def __init__(self, debug=False):
        mu = 0.6
        supp_poly_size = 0.2
        self.debug = debug
        self.model = RBDLModel(paths.physics_model_file, update_kinematics_by_hand=True)
        self.params = read_debug_param_values_from_json(paths.physics_parameter_file)
        self.friction_constraint_matrix = np.array(
            [
                [np.sqrt(2), -mu, 0],
                [-np.sqrt(2), -mu, 0],
                [0, -mu, np.sqrt(2)],
                [0, -mu, -np.sqrt(2)],
            ]
        )
        self.support_polygon = np.array(
            [
                [-supp_poly_size / 2, 0, -supp_poly_size / 2],
                [supp_poly_size / 2, 0, -supp_poly_size / 2],
                [-supp_poly_size / 2, 0, supp_poly_size / 2],
                [supp_poly_size / 2, 0, supp_poly_size / 2],
            ]
        )

        connection_mode = p.GUI if debug else p.DIRECT
        if not p.isConnected():
            p.connect(connection_mode)
        if debug:
            p.configureDebugVisualizer(flag=p.COV_ENABLE_Y_AXIS_UP, enable=1)
            load_debug_params_into_bullet_from_json(paths.physics_parameter_file)

        self.seq_id = None
        self.last_x = []
        self.q = None
        self.qdot = np.zeros(self.model.qdot_size)
        self.reset_states()
        self.frame_no = 0
        self.height_map = np.load(paths.height_map_file).astype(np.float32)
        self.height_map = np.maximum(self.height_map, 0.0)

        self.mass = 0.0
        for index in range(len(self.model.model.mBodies)):
            self.mass += self.model.model.mBodies[index].mMass

    def reset_states(self):
        self.last_x = []
        self.q = None
        self.qdot = np.zeros(self.model.qdot_size)
        self.qddot = np.zeros(self.model.qdot_size)
        self.prev_qddot = np.zeros(self.model.qdot_size)
        self.translation = torch.zeros(3)

        if self.seq_id is not None:
            self.id_robot = p.loadURDF(
                paths.physics_model_file,
                [0.0, 0, 0.0],
                useFixedBase=False,
                flags=p.URDF_MERGE_FIXED_LINKS,
            )
            change_color(self.id_robot, [198 / 255, 238 / 255, 0, 1.0])
            self.scene = p.loadURDF(paths.plane_file, [0, 0, 0.0])

            aabb = p.getAABB(self.scene)
            self.scene_xmin, _, self.scene_zmin = aabb[0]
            self.scene_xmax, _, self.scene_zmax = aabb[1]
            self.map_width = abs(self.scene_xmax - self.scene_xmin)
            self.map_height = abs(self.scene_zmax - self.scene_zmin)
            self.map_res = self.height_map.shape[0]

    def get_heightmap_index(self, pos):
        norm_x = (pos[0] - self.scene_xmin) / (self.scene_xmax - self.scene_xmin)
        norm_z = (pos[2] - self.scene_zmin) / (self.scene_zmax - self.scene_zmin)
        i = int(norm_x * (self.map_res - 1))
        j = int(norm_z * (self.map_res - 1))
        i = max(0, min(i, self.map_res - 1))
        j = max(0, min(j, self.map_res - 1))
        return i, j

    def change_base(self, q_ref):
        lheel_pos = p.getLinkState(self.id_robot, 10)[4][1]
        rheel_pos = p.getLinkState(self.id_robot, 22)[4][1]

        if lheel_pos < 0:
            q_ref[1] = q_ref[1] + (-lheel_pos)
        elif rheel_pos < 0:
            q_ref[1] = q_ref[1] + (-rheel_pos)
        return q_ref

    def optimize_frame(self, pose, jvel, contact, joint3D, joint3D_p1, joint3D_p2):
        q_ref = smpl_to_rbdl(pose, self.translation)[0]
        v_ref = jvel.numpy()
        c_ref = contact.numpy()
        q = self.q
        qdot = self.qdot

        if q is None:
            self.translation = joint3D.reshape((24, 3))[0]
            q_ref[:3] = self.translation
            self.q = q_ref
            return pose, self.translation, torch.zeros((75,)), torch.zeros((6,)), self.q, self.qdot, torch.zeros((75,))

        self.model.update_kinematics(q, qdot, np.zeros(self.model.qdot_size))
        set_pose(self.id_robot, q)
        q = self.change_base(q)
        Js = [np.empty((0, self.model.qdot_size))]
        collision_points, collision_joints = [], []
        floor_height_dict = {joint_name: 0.0 for joint_name in self.test_contact_joints}
        col_point_dict = {joint_name: [] for joint_name in self.test_contact_joints}

        for joint_name in self.test_contact_joints:
            joint_id = vars(Body)[joint_name]
            pos = self.model.calc_body_position(q, joint_id)
            x_joint, z_joint = self.get_heightmap_index(pos)
            floor_height_dict[joint_name] = self.height_map[x_joint, z_joint]

            if (joint_id == Body.LFOOT and c_ref[0] > 0.5) or (joint_id == Body.RFOOT and c_ref[1] > 0.5):
                collision_joints.append(joint_name)
                for ps in [pos]:
                    collision_points.append(ps)
                    col_point_dict[joint_name].append(ps)
                    pb = self.model.calc_base_to_body_coordinates(q, joint_id, ps)
                    Js.append(self.model.calc_point_Jacobian(q, joint_id, pb))

        Js = np.vstack(Js)
        nc = len(collision_points)

        As1, bs1, As2, bs2, As3, bs3, As4, bs4 = [np.zeros((0, self.model.qdot_size))], [np.empty(0)], [np.empty((0, nc * 3))], [np.empty(0)], [np.zeros((0, self.model.qdot_size))], [np.empty(0)], [np.zeros((0, self.model.qdot_size))], [np.empty(0)]
        Gs1, hs1, Gs2, hs2, Gs3, hs3, Gs4, hs4 = [np.zeros((0, self.model.qdot_size))], [np.empty(0)], [np.empty((0, nc * 3))], [np.empty(0)], [np.zeros((0, self.model.qdot_size))], [np.empty(0)], [np.zeros((0, self.model.qdot_size))], [np.empty(0)]
        A_, b_ = None, None

        A = np.hstack((np.zeros((self.model.qdot_size - 3, 3)), np.eye((self.model.qdot_size - 3))))
        b = self.params["kp_angular"] * art.math.angle_difference(q_ref[3:], q[3:]) - self.params["kd_angular"] * qdot[3:]
        As1.append(A)
        bs1.append(b)

        for joint_name, velocity in zip(
            [
                "ROOT",
                "LHIP",
                "RHIP",
                "SPINE1",
                "LKNEE",
                "RKNEE",
                "SPINE2",
                "LANKLE",
                "RANKLE",
                "SPINE3",
                "LFOOT",
                "RFOOT",
                "NECK",
                "LCLAVICLE",
                "RCLAVICLE",
                "HEAD",
                "LSHOULDER",
                "RSHOULDER",
                "LELBOW",
                "RELBOW",
                "LWRIST",
                "RWRIST",
            ],
            v_ref[:22],
        ):
            joint_id = vars(Body)[joint_name]
            if joint_id == Body.LFOOT or joint_id == Body.RFOOT:
                continue
            cur_vel = self.model.calc_point_velocity(q, qdot, joint_id)
            a_des = self.params["kp_linear"] * velocity * self.params["delta_t"] - self.params["kd_linear"] * cur_vel
            A = self.model.calc_point_Jacobian(q, joint_id)
            b = -self.model.calc_point_acceleration(q, qdot, np.zeros(75), joint_id) + a_des
            As1.append(A * self.params["coeff_jvel"])
            bs1.append(b * self.params["coeff_jvel"])

        if nc != 0:
            A = [np.eye(3) * max(points[1] - floor_height_dict[cp], 0.005) for cp in collision_joints for points in col_point_dict[cp]]
            A = art.math.block_diagonal_matrix_np(A)
            As2.append(A * self.params["coeff_lambda"])
            bs2.append(np.zeros(nc * 3))

        As3.append(
            art.math.block_diagonal_matrix_np(
                [
                    np.eye(6) * self.params["coeff_virtual"],
                    np.eye(self.model.qdot_size - 6) * self.params["coeff_tau"],
                ]
            )
        )
        bs3.append(np.zeros(self.model.qdot_size))

        for joint_name in self.test_contact_joints[:-2]:
            joint_id = vars(Body)[joint_name]
            pos = self.model.calc_body_position(q, joint_id)
            x_joint, z_joint = self.get_heightmap_index(pos)
            floor_height = self.height_map[x_joint, z_joint]

            if pos[1] <= floor_height:
                J = self.model.calc_point_Jacobian(q, joint_id)
                v = self.model.calc_point_velocity(q, qdot, joint_id)
                Gs1.append(-self.params["delta_t"] * J)
                hs1.append(v - [-1e-1, 0, -1e-1])
                Gs1.append(self.params["delta_t"] * J)
                hs1.append(-v + [1e-1, 1e2, 1e-1])

        for joint_name, stable in zip(["LFOOT", "RFOOT"], c_ref):
            joint_id = vars(Body)[joint_name]
            pos = self.model.calc_body_position(q, joint_id)
            J = self.model.calc_point_Jacobian(q, joint_id)
            v = self.model.calc_point_velocity(q, qdot, joint_id)

            x_joint, z_joint = self.get_heightmap_index(pos)
            floor_height = self.height_map[x_joint, z_joint]

            if stable == 0:
                stable = 0.0001
            th = -np.log(min(stable, 0.84999) / 0.85)
            th_y = (floor_height - pos[1]) / self.params["delta_t"]
            Gs1.append(-self.params["delta_t"] * J)
            hs1.append(v - [-th, th_y, -th])
            Gs1.append(self.params["delta_t"] * J)
            hs1.append(-v + [th, max(th, th_y) + 1e-6, th])

        if nc > 0:
            Gs2.append(art.math.block_diagonal_matrix_np([self.friction_constraint_matrix] * nc))
            hs2.append(np.zeros(nc * 4))

        M = self.model.calc_M(q)
        h = self.model.calc_h(q, qdot)
        A_ = np.hstack((-M, Js.T, np.eye(self.model.qdot_size)))
        b_ = h

        if joint3D_p1 is not None and joint3D_p2 is not None:
            delta_t_square = np.hstack((np.eye(3), np.zeros((3, self.model.qdot_size * 2 - 3 + nc * 3))))
            rhs = (
                joint3D_p2[0].numpy() / (self.params["delta_t"] ** 2)
                - joint3D_p1[0].numpy() / (self.params["delta_t"] ** 2)
                - qdot[:3] / self.params["delta_t"]
            )
            A_ = np.vstack((A_, delta_t_square))
            b_ = np.concatenate((b_, rhs))

        As1 = np.vstack(As1)
        bs1 = np.concatenate(bs1)
        As2 = np.vstack(As2)
        bs2 = np.concatenate(bs2)
        As3 = np.vstack(As3)
        bs3 = np.concatenate(bs3)
        Gs1 = np.vstack(Gs1)
        hs1 = np.concatenate(hs1)
        Gs2 = np.vstack(Gs2)
        hs2 = np.concatenate(hs2)
        Gs3 = np.vstack(Gs3)
        hs3 = np.concatenate(hs3)

        G_ = art.math.block_diagonal_matrix_np([Gs1, Gs2, Gs3])
        h_ = np.concatenate((hs1, hs2, hs3))
        P_ = art.math.block_diagonal_matrix_np([As1.T.dot(As1), As2.T.dot(As2), As3.T.dot(As3)])
        q_ = np.concatenate((-As1.T.dot(bs1), -As2.T.dot(bs2), -As3.T.dot(bs3)))

        init = self.last_x if len(self.last_x) == len(q_) else None
        x = solve_qp(P_, q_, G_, h_, A_, b_, solver="clarabel", initvals=init)
        if x is None or np.linalg.norm(x) > 10000:
            x = solve_qp(P_, q_, G_, h_, A_, b_, solver="cvxopt", initvals=init)
        if x is None:
            raise RuntimeError("Quadratic program failed during optimization.")

        qddot = x[: self.model.qdot_size]
        grf = x[self.model.qdot_size : self.model.qdot_size + nc * 3]
        tau = x[self.model.qdot_size + nc * 3 : self.model.qdot_size + nc * 3 + self.model.qdot_size]

        qdot = qdot + qddot * self.params["delta_t"]
        q = q + qdot * self.params["delta_t"]

        self.q = q
        self.qdot = qdot
        self.last_x = x
        self.qddot = qddot
        self.frame_no += 1

        if self.debug:
            set_pose(self.id_robot, q)
            self.params = read_debug_param_values_from_bullet()

        pose_opt, tran_opt = rbdl_to_smpl(q)
        pose_opt = torch.from_numpy(pose_opt).float()[0]
        tran_opt = torch.from_numpy(tran_opt).float()[0]
        self.translation = tran_opt
        return pose_opt, tran_opt, torch.from_numpy(tau).float(), torch.from_numpy(grf).float().view(-1, 3), self.q, self.qdot, qddot
