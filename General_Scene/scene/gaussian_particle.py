import torch
import math
class GaussianParticle:
    def __init__(self, config, scene_extent, sh_degree=2):
        num_drops = config['nums']
        self.num_drops = config['nums']
        self.positions = (torch.rand(num_drops, 3).cuda() - 0.5) * scene_extent

        # Scaling
        mean = torch.tensor(config["scaling_mean"]).cuda()
        std_dev = torch.tensor(config["scaling_std"]).cuda()
        self.scaling = torch.normal(mean=mean.repeat(num_drops, 1),
                                    std=std_dev.repeat(num_drops, 1)).cuda()

        layer1, layer2, layer3 = int(num_drops), int(2 * num_drops / 3), int(num_drops / 3)
        scaling_factors = [1, 2, 3]
        self.scaling[:layer1] *= scaling_factors[0]
        self.scaling[layer1:layer2] *= scaling_factors[1]
        self.scaling[layer2:layer3] *= scaling_factors[2]

        # Rotation
        base_angle = -math.pi / 32
        std_angle = math.pi / 64
        random_angles = torch.normal(mean=base_angle, std=std_angle, size=(num_drops,), device="cuda")
        axis = torch.tensor(config["rotation_axis"], device="cuda", dtype=torch.float32)

        angle_tensors = random_angles / 2
        sin_half_angles = torch.sin(angle_tensors)
        cos_half_angles = torch.cos(angle_tensors)
        self.rotation = torch.cat([
            cos_half_angles.unsqueeze(1),
            sin_half_angles.unsqueeze(1) * axis
        ], dim=1)

        # Opacity
        self.opacity = torch.normal(
            mean=config["opacity_mean"],
            std=config["opacity_std"],
            size=(num_drops, 1)
        ).cuda()

        # Features
        self.features_dc = torch.ones((num_drops, 3, 1), device="cuda") * config["features_dc"]
        self.features_rest = torch.zeros((num_drops, 3, (sh_degree + 1) ** 2 - 1), device="cuda")

        # Velocity
        self.velocity = torch.tensor(config["velocity"], device="cuda").repeat(num_drops, 1)

        self.scene_extent = scene_extent


    def update_positions(self, delta_time):
        """Update the positions of raindrops based on velocity."""
        # pass
        self.positions[:, 1] += self.velocity[:, 1] * delta_time  # ✅ y 正方向表示往下落

        # 使用 scene_extent 作为边界条件（底部界限）
        reset_condition = self.positions[:, 1] > self.scene_extent / 2  # ✅ 超出下边界就重置

        # 将落到底部的粒子重置到上方，并重新随机 XZ 位置
        self.positions[reset_condition, 1] = -self.scene_extent / 2  # 重置到顶部

    def get_static_gaussians(self):
        return {
            'positions': self.positions,
            'scaling': self.scaling,
            'rotation': self.rotation,
            'opacity': self.opacity,
            'features_dc': self.features_dc,
            'features_rest': self.features_rest
        }