import torch

from dynamics_scene import PhysicsOptimizer


class PIP:
    name = "PIP"

    def __init__(self, debug=False):
        self.dynamics_optimizer = PhysicsOptimizer(debug=debug)

    @torch.no_grad()
    def predict(self, glb_rot, pose, joint_velocity, contacts, sequence_id, joint_positions):
        self.dynamics_optimizer.seq_id = sequence_id
        self.dynamics_optimizer.reset_states()

        joint_velocity = torch.cat((joint_velocity[0].unsqueeze(0), joint_velocity), dim=0)
        joint_velocity = joint_velocity.view(-1, 24, 3).bmm(glb_rot[:, -1].transpose(1, 2))
        contacts = contacts[: len(joint_positions)]

        pose_opt = []
        translation_opt = []

        for frame_index, (pose_frame, vel_frame, contact_frame, joints_frame) in enumerate(
            zip(pose, joint_velocity, contacts, joint_positions)
        ):
            if frame_index < len(contacts) - 2:
                joints_next = joint_positions[frame_index + 1]
                joints_next_next = joint_positions[frame_index + 2]
            else:
                joints_next = None
                joints_next_next = None

            optimized_pose, optimized_translation, _, _, _, _, _ = self.dynamics_optimizer.optimize_frame(
                pose_frame,
                vel_frame,
                contact_frame,
                joints_frame,
                joints_next,
                joints_next_next,
            )
            pose_opt.append(optimized_pose)
            translation_opt.append(optimized_translation)

        translation_opt[0] = torch.as_tensor(translation_opt[0]).clone().detach()
        return torch.stack(pose_opt), torch.stack(translation_opt)
