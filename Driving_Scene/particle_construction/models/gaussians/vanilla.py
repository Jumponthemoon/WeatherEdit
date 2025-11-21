"""
Filename: 3dgs.py

Author: Ziyu Chen (ziyu.sjtu@gmail.com)

Description:
Unofficial implementation of 3DGS based on the work by Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 
This implementation is modified from the nerfstudio GaussianSplattingModel.

- Original work by Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis.
- Codebase reference: nerfstudio GaussianSplattingModel (https://github.com/nerfstudio-project/nerfstudio/blob/gaussian-splatting/nerfstudio/models/gaussian_splatting.py)

Original paper: https://arxiv.org/abs/2308.04079
"""
import torch.nn.functional as F 
from typing import Dict, List, Tuple
from omegaconf import OmegaConf
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn as nn
from torch.nn import Parameter
import open3d as o3d
from models.gaussians.basics import *





class VanillaGaussians(nn.Module):

    def __init__(
        self,
        class_name: str,
        ctrl: OmegaConf,
        reg: OmegaConf = None,
        networks: OmegaConf = None,
        scene_scale: float = 30.,
        scene_origin: torch.Tensor = torch.zeros(3),
        num_train_images: int = 300,
        device: torch.device = torch.device("cuda"),
        particle_cfg: OmegaConf = None,
        weather_type: str = None,
        **kwargs
    ):
        super().__init__()
        self.class_prefix = class_name + "#"
        self.ctrl_cfg = ctrl
        self.reg_cfg = reg
        self.networks_cfg = networks
        self.scene_scale = scene_scale
        self.scene_origin = scene_origin
        self.num_train_images = num_train_images
        self.step = 0
        self.particle_cfg = particle_cfg
        self.device = device
        self.ball_gaussians=self.ctrl_cfg.get("ball_gaussians", False)
        self.gaussian_2d = self.ctrl_cfg.get("gaussian_2d", False)
        # for evaluation
        self.in_test_set = False
        # init models
        self.xys_grad_norm = None
        self.max_2Dsize = None
        self._means = torch.zeros(1, 3, device=self.device)
        if self.ball_gaussians:
            self._scales = torch.zeros(1, 1, device=self.device)
        else:
            if self.gaussian_2d:
                self._scales = torch.zeros(1, 2, device=self.device)
            else:
                self._scales = torch.zeros(1, 3, device=self.device)
        self._quats = torch.zeros(1, 4, device=self.device)
        self._opacities = torch.zeros(1, 1, device=self.device)
        self._features_dc = torch.zeros(1, 3, device=self.device)
        self._features_rest = torch.zeros(1, num_sh_bases(self.sh_degree) - 1, 3, device=self.device)

        if self.particle_cfg:
            self.initialize_weather(weather_type)

    @property
    def sh_degree(self):
        return self.ctrl_cfg.sh_degree

    def create_from_pcd(self, init_means: torch.Tensor, init_colors: torch.Tensor) -> None:
        self._means = Parameter(init_means)
        
        distances, _ = k_nearest_sklearn(self._means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True).to(self.device)
        if self.ball_gaussians:
            self._scales = Parameter(torch.log(avg_dist.repeat(1, 1)))
        else:
            if self.gaussian_2d:
                self._scales = Parameter(torch.log(avg_dist.repeat(1, 2)))
            else:
                self._scales = Parameter(torch.log(avg_dist.repeat(1, 3)))
        self._quats = Parameter(random_quat_tensor(self.num_points).to(self.device))
        dim_sh = num_sh_bases(self.sh_degree)

        fused_color = RGB2SH(init_colors) # float range [0, 1] 
        shs = torch.zeros((fused_color.shape[0], dim_sh, 3)).float().to(self.device)
        if self.sh_degree > 0:
            shs[:, 0, :3] = fused_color
            shs[:, 1:, 3:] = 0.0
        else:
            shs[:, 0, :3] = torch.logit(init_colors, eps=1e-10)
        self._features_dc = Parameter(shs[:, 0, :])
        self._features_rest = Parameter(shs[:, 1:, :])
        self._opacities = Parameter(torch.logit(0.1 * torch.ones(self.num_points, 1, device=self.device)))
        
    @property
    def colors(self):
        if self.sh_degree > 0:
            return SH2RGB(self._features_dc)
        else:
            return torch.sigmoid(self._features_dc)
    @property
    def shs_0(self):
        return self._features_dc
    @property
    def shs_rest(self):
        return self._features_rest
    @property
    def num_points(self):
        return self._means.shape[0]
    @property
    def get_scaling(self):
        if self.ball_gaussians:
            if self.gaussian_2d:
                scaling = torch.exp(self._scales).repeat(1, 2)
                scaling = torch.cat([scaling, torch.zeros_like(scaling[..., :1])], dim=-1)
                return scaling
            else:
                return torch.exp(self._scales).repeat(1, 3)
        else:
            if self.gaussian_2d:
                scaling = torch.exp(self._scales)
                scaling = torch.cat([scaling[..., :2], torch.zeros_like(scaling[..., :1])], dim=-1)
                return scaling
            else:
                return torch.exp(self._scales)
    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacities)
    @property
    def get_quats(self):
        return self.quat_act(self._quats)
    
    def quat_act(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True)
    
    def preprocess_per_train_step(self, step: int):
        self.step = step
        
    def postprocess_per_train_step(
        self,
        step: int,
        optimizer: torch.optim.Optimizer,
        radii: torch.Tensor,
        xys_grad: torch.Tensor,
        last_size: int,
    ) -> None:
        self.after_train(radii, xys_grad, last_size)
        if step % self.ctrl_cfg.refine_interval == 0:
            self.refinement_after(step, optimizer)

    def after_train(
        self,
        radii: torch.Tensor,
        xys_grad: torch.Tensor,
        last_size: int,
    ) -> None:
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (radii > 0).flatten()
            #print(radii.size())
            full_mask = torch.zeros(self.num_points, device=radii.device, dtype=torch.bool)
           # print(full_mask.size(),self.filter_mask.size())
            full_mask[self.filter_mask] = visible_mask
            
            grads = xys_grad.norm(dim=-1)
            if self.xys_grad_norm is None:
                self.xys_grad_norm = torch.zeros(self.num_points, device=grads.device, dtype=grads.dtype)
                self.xys_grad_norm[self.filter_mask] = grads
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                assert self.vis_counts is not None
                self.vis_counts[full_mask] = self.vis_counts[full_mask] + 1
                self.xys_grad_norm[full_mask] = grads[visible_mask] + self.xys_grad_norm[full_mask]

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros(self.num_points, device=radii.device, dtype=torch.float32)
            newradii = radii[visible_mask]
            self.max_2Dsize[full_mask] = torch.maximum(
                self.max_2Dsize[full_mask], newradii / float(last_size)
            )
        
    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            self.class_prefix+"xyz": [self._means],
            self.class_prefix+"sh_dc": [self._features_dc],
            self.class_prefix+"sh_rest": [self._features_rest],
            self.class_prefix+"opacity": [self._opacities],
            self.class_prefix+"scaling": [self._scales],
            self.class_prefix+"rotation": [self._quats],
        }
    
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return self.get_gaussian_param_groups()

    def refinement_after(self, step, optimizer: torch.optim.Optimizer) -> None:
        assert step == self.step
        if self.step <= self.ctrl_cfg.warmup_steps:
            return
        with torch.no_grad():
            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.ctrl_cfg.reset_alpha_interval
            do_densification = (
                self.step < self.ctrl_cfg.stop_split_at
                and self.step % reset_interval > max(self.num_train_images, self.ctrl_cfg.refine_interval)
            )
            # split & duplicate
            print(f"Class {self.class_prefix} current points: {self.num_points} @ step {self.step}")
            if do_densification:
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                
                avg_grad_norm = self.xys_grad_norm / self.vis_counts
                high_grads = (avg_grad_norm > self.ctrl_cfg.densify_grad_thresh).squeeze()
                
                splits = (
                    self.get_scaling.max(dim=-1).values > \
                        self.ctrl_cfg.densify_size_thresh * self.scene_scale
                ).squeeze()
                if self.step < self.ctrl_cfg.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.ctrl_cfg.split_screen_size).squeeze()
                splits &= high_grads
                nsamps = self.ctrl_cfg.n_split_samples
                (
                    split_means,
                    split_feature_dc,
                    split_feature_rest,
                    split_opacities,
                    split_scales,
                    split_quats,
                ) = self.split_gaussians(splits, nsamps)

                dups = (
                    self.get_scaling.max(dim=-1).values <= \
                        self.ctrl_cfg.densify_size_thresh * self.scene_scale
                ).squeeze()
                dups &= high_grads
                (
                    dup_means,
                    dup_feature_dc,
                    dup_feature_rest,
                    dup_opacities,
                    dup_scales,
                    dup_quats,
                ) = self.dup_gaussians(dups)
                
                self._means = Parameter(torch.cat([self._means.detach(), split_means, dup_means], dim=0))
                self._features_dc = Parameter(torch.cat([self._features_dc.detach(), split_feature_dc, dup_feature_dc], dim=0))
                self._features_rest = Parameter(torch.cat([self._features_rest.detach(), split_feature_rest, dup_feature_rest], dim=0))
                self._opacities = Parameter(torch.cat([self._opacities.detach(), split_opacities, dup_opacities], dim=0))
                self._scales = Parameter(torch.cat([self._scales.detach(), split_scales, dup_scales], dim=0))
                self._quats = Parameter(torch.cat([self._quats.detach(), split_quats, dup_quats], dim=0))
                
                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [self.max_2Dsize, torch.zeros_like(split_scales[:, 0]), torch.zeros_like(dup_scales[:, 0])],
                    dim=0,
                )
                
                split_idcs = torch.where(splits)[0]
                param_groups = self.get_gaussian_param_groups()
                dup_in_optim(optimizer, split_idcs, param_groups, n=nsamps)

                dup_idcs = torch.where(dups)[0]
                param_groups = self.get_gaussian_param_groups()
                dup_in_optim(optimizer, dup_idcs, param_groups, 1)

            # cull NOTE: Offset all the opacity reset logic by refine_every so that we don't
                # save checkpoints right when the opacity is reset (saves every 2k)
            if self.step % reset_interval > max(self.num_train_images, self.ctrl_cfg.refine_interval):
                deleted_mask = self.cull_gaussians()
                param_groups = self.get_gaussian_param_groups()
                remove_from_optim(optimizer, deleted_mask, param_groups)
            print(f"Class {self.class_prefix} left points: {self.num_points}")
                    
            # reset opacity
            if self.step % reset_interval == self.ctrl_cfg.refine_interval:
                # NOTE: in nerfstudio, reset_value = cull_alpha_thresh * 0.8
                    # we align to original repo of gaussians spalting
                reset_value = torch.min(self.get_opacity.data,
                                        torch.ones_like(self._opacities.data) * self.ctrl_cfg.reset_alpha_value)
                self._opacities.data = torch.logit(reset_value)
                # reset the exp of optimizer
                for group in optimizer.param_groups:
                    if group["name"] == self.class_prefix+"opacity":
                        old_params = group["params"][0]
                        param_state = optimizer.state[old_params]
                        param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                        param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])
            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

    def cull_gaussians(self):
        """
        This function deletes gaussians with under a certain opacity threshold
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (self.get_opacity.data < self.ctrl_cfg.cull_alpha_thresh).squeeze()
        if self.step > self.ctrl_cfg.reset_alpha_interval:
            # cull huge ones
            toobigs = (
                torch.exp(self._scales).max(dim=-1).values > 
                self.ctrl_cfg.cull_scale_thresh * self.scene_scale
            ).squeeze()
            culls = culls | toobigs
            if self.step < self.ctrl_cfg.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                culls = culls | (self.max_2Dsize > self.ctrl_cfg.cull_screen_size).squeeze()
        self._means = Parameter(self._means[~culls].detach())
        self._scales = Parameter(self._scales[~culls].detach())
        self._quats = Parameter(self._quats[~culls].detach())
        # self.colors_all = Parameter(self.colors_all[~culls].detach())
        self._features_dc = Parameter(self._features_dc[~culls].detach())
        self._features_rest = Parameter(self._features_rest[~culls].detach())
        self._opacities = Parameter(self._opacities[~culls].detach())

        print(f"     Cull: {n_bef - self.num_points}")
        return culls
    

    def initialize_weather(self, weather_type="foggy",dataset_name="pandaset"):
        """
        Initialize particles for different weather conditions.

        Parameters:
        - weather_type: The type of weather (e.g., "rainy", "snowy", "foggy").
        """
        # 从 particle_cfg 中加载对应天气的参数
        weather_cfg = self.particle_cfg.dataset.get(dataset_name).get(weather_type, {})
        # weather_cfg = self.particle_cfg.dataset_name.get(weather_type, {})
        num_drops = weather_cfg.num_drops
        field_param = weather_cfg.field_param
        sh_degree = weather_cfg.sh_degree
        opacity_mean = weather_cfg.opacity_mean
        opacity_std = weather_cfg.opacity_std
        scaling_mean = weather_cfg.scaling_mean
        scaling_std = weather_cfg.scaling_std
        rotation_axes = weather_cfg.rotation_axes
        rotation_base_angle = weather_cfg.rotation_base_angle
        rotation_std_angle = weather_cfg.rotation_std_angle
        scaling_factors = weather_cfg.scaling_factors
        color = weather_cfg.color
        self.fall_speed = weather_cfg.fall_speed

        # Construct local coordinate
        self._initialize_local_coordinates()
        # initialize particle positions
        self.particle_positions = torch.rand(num_drops, 3).cuda()
        field_scale, shift_scale, depth_offset, height_offset = field_param
        if dataset_name == "pandaset" or dataset_name == "waymo":
            self.particle_positions[:, 0] = (self.particle_positions[:, 0] - shift_scale) * field_scale
            self.particle_positions[:, 1] = self.particle_positions[:, 1] * field_scale * 2 - depth_offset
            self.particle_positions[:, 2] = self.particle_positions[:, 2] * field_scale - height_offset
        else:
            self.particle_positions[:,0] = (self.particle_positions[:,0]-shift_scale) * field_scale + 7 
            self.particle_positions[:,1] = self.particle_positions[:,1] * field_scale - height_offset - 10
            self.particle_positions[:,2] = self.particle_positions[:,2] * field_scale - 30

        # initialize particle scaling
        mean = torch.tensor(scaling_mean).cuda()
        std_dev = torch.tensor(scaling_std).cuda()
        self.particle_scaling = torch.normal(mean=mean.repeat(num_drops, 1), std=std_dev.repeat(num_drops, 1)).cuda()
        layer1, layer2, layer3 = int(num_drops / 3), int(2 * num_drops / 3), int(num_drops)
        self.particle_scaling[:layer1, 1] *= scaling_factors[0]  
        self.particle_scaling[layer1:layer2, 1] *= scaling_factors[1]  
        self.particle_scaling[layer2:layer3, 1] *= scaling_factors[2]  
        # initialize particle rotations
        random_angles = torch.normal(mean=rotation_base_angle, std=rotation_std_angle, size=(num_drops,), device="cuda")
        rotation_axes = torch.tensor(rotation_axes, device="cuda", dtype=torch.float32)
        angle_tensors = random_angles / 2
        sin_half_angles = torch.sin(angle_tensors)
        cos_half_angles = torch.cos(angle_tensors)
        self.particle_rotations = torch.cat([
            cos_half_angles.unsqueeze(1),
            sin_half_angles.unsqueeze(1) * rotation_axes
        ], dim=1)


        # 初始化粒子颜色
        self.particle_features_dc = torch.ones((num_drops, 3, 1), device="cuda") * color
        self.particle_features_rest = torch.zeros((num_drops, 3, (sh_degree + 1) ** 2 - 1), device="cuda")


        if weather_type == "foggy":
        
            ## Calculate transmission ###
            lateral_dir = 0
            if dataset_name == "pandaset" or dataset_name == "waymo":
                forward_dir = 1
            else:
                forward_dir = 2
            #### x #####
            beta = weather_cfg.beta
            depth_y = self.particle_positions[:, lateral_dir]
            depth_y = torch.abs(depth_y) / field_param[0] * 2   # 归一化深度
            sorted_indices_y = torch.argsort(depth_y)
            depths_sorted_y = depth_y[sorted_indices_y]
            transmissions_y = torch.exp(-beta * depths_sorted_y)  # 透射率
            opacity_y = 1 - transmissions_y  # 垂直透明度
            opacity_y = opacity_y[torch.argsort(sorted_indices_y)].unsqueeze(1)  # 恢复顺序

            depth = torch.clamp(self.particle_positions[:,forward_dir] ,min=0)
            depth = depth/field_param[0]
            sorted_indices = torch.argsort(depth)
            depths_sorted = depth[sorted_indices]
            transmissions = torch.exp(-beta * depths_sorted)     
            opacity_accumulated = 1 - transmissions 

            self.particle_opacity = opacity_accumulated[torch.argsort(sorted_indices)].unsqueeze(1)
            self.particle_opacity = torch.max(self.particle_opacity, opacity_y)
        else:
            self.particle_opacity = torch.normal(mean=opacity_mean, std=opacity_std, size=(num_drops, 1), device="cuda")


    def split_gaussians(self, split_mask: torch.Tensor, samps: int) -> Tuple:
        """
        This function splits gaussians that are too large
        """

        n_splits = split_mask.sum().item()
        print(f"    Split: {n_splits}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            self.get_scaling[split_mask].repeat(samps, 1) * centered_samples
            # torch.exp(self._scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quat_act(self._quats[split_mask])  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self._means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        # new_colors_all = self.colors_all[split_mask].repeat(samps, 1, 1)
        new_feature_dc = self._features_dc[split_mask].repeat(samps, 1)
        new_feature_rest = self._features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self._opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self._scales[split_mask]) / size_fac).repeat(samps, 1)
        self._scales[split_mask] = torch.log(torch.exp(self._scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self._quats[split_mask].repeat(samps, 1)
        return new_means, new_feature_dc, new_feature_rest, new_opacities, new_scales, new_quats

    def dup_gaussians(self, dup_mask: torch.Tensor) -> Tuple:
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        print(f"      Dup: {n_dups}")
        dup_means = self._means[dup_mask]
        # dup_colors = self.colors_all[dup_mask]
        dup_feature_dc = self._features_dc[dup_mask]
        dup_feature_rest = self._features_rest[dup_mask]
        dup_opacities = self._opacities[dup_mask]
        dup_scales = self._scales[dup_mask]
        dup_quats = self._quats[dup_mask]
        return dup_means, dup_feature_dc, dup_feature_rest, dup_opacities, dup_scales, dup_quats

    def get_gaussians(self, cam: dataclass_camera) -> Dict:
        filter_mask = torch.ones_like(self._means[:, 0], dtype=torch.bool)
        self.filter_mask = filter_mask
        
        # get colors of gaussians
        colors = torch.cat((self._features_dc[:, None, :], self._features_rest), dim=1)
        if self.sh_degree > 0:
            viewdirs = self._means.detach() - cam.camtoworlds.data[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.ctrl_cfg.sh_degree_interval, self.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])
            
        activated_opacities = self.get_opacity
        activated_scales = self.get_scaling
        activated_rotations = self.get_quats
        actovated_colors = rgbs
        
        # collect gaussians information
        gs_dict = dict(
            _means=self._means[filter_mask],
            _opacities=activated_opacities[filter_mask],
            _rgbs=actovated_colors[filter_mask],
            _scales=activated_scales[filter_mask],
            _quats=activated_rotations[filter_mask],
        )
        
        # check nan and inf in gs_dict
        for k, v in gs_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in gaussian {k} at step {self.step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in gaussian {k} at step {self.step}")
                
        return gs_dict
        
    def get_gaussians_with_particles(self, cam: dataclass_camera, movement:None,front_cam=False,first_frame=False, dataset=None) -> Dict:
        filter_mask = torch.ones_like(self._means[:, 0], dtype=torch.bool)
        self.filter_mask = filter_mask
        
        # get colors of gaussians
        colors = torch.cat((self._features_dc[:, None, :], self._features_rest), dim=1)
        if self.sh_degree > 0:
            viewdirs = self._means.detach() - cam.camtoworlds.data[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.ctrl_cfg.sh_degree_interval, self.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])
            
        activated_opacities = self.get_opacity
        activated_scales = self.get_scaling
        activated_rotations = self.get_quats
        actovated_colors = rgbs

        if front_cam:
            if dataset == "pandaset" or dataset == "waymo":
                self.particle_positions[:,2] -= self.fall_speed
            else:

                self.particle_positions[:,1] += self.fall_speed


        movement = movement.to(device=self.particle_positions.device) 

        rotation_matrix = movement[:3, :3]

        translation_vector = movement[:3, 3]

        rotated_positions = self.particle_positions @ rotation_matrix.T

        self.particle_positions = rotated_positions + translation_vector


        if front_cam:
            self.local_coord = self.local_coord @ rotation_matrix.T + translation_vector
            translated_points = torch.matmul(self.particle_positions - self.local_coord,self.local_coord_basis.T)  
            # Pandaset
            if dataset == "pandaset" or dataset == "waymo":

                local_max = torch.max(translated_points[:,2])  
                local_min = torch.min(translated_points[:,2])  
            else:
                local_max = torch.max(translated_points[:,1])  
                local_min = torch.min(translated_points[:,1])  

            if first_frame:
                self.local_max=local_max
                self.local_min=local_min
             
            # pandaset
            if dataset == "pandaset" or dataset == "waymo":
                out_of_view = translated_points[:, 2] < self.local_min 
                offset = translated_points[out_of_view,2]
                translated_points[out_of_view,2] = offset+ self.local_max-self.local_min
            else:
                out_of_view = translated_points[:, 1] > self.local_max 
                offset = translated_points[out_of_view,1]
                translated_points[out_of_view,1] = offset - (self.local_max - self.local_min)
            self.particle_positions= torch.matmul(translated_points,self.local_coord_basis) + self.local_coord
     
        particle_rgbs = self.particle_features_dc.squeeze(dim=2) 

        combined_means = torch.cat([self._means, self.particle_positions], dim=0)
        combined_opacities = torch.cat([activated_opacities, self.particle_opacity], dim=0)
        combined_rgbs = torch.cat([actovated_colors, particle_rgbs], dim=0)
        combined_scales = torch.cat([activated_scales, self.particle_scaling], dim=0)
        combined_rotations = torch.cat([activated_rotations, self.particle_rotations], dim=0)

        # Return combined Gaussians, including raindrops
        gs_dict = dict(
            _means=combined_means,
            _opacities=combined_opacities,
            _rgbs=combined_rgbs,
            _scales=combined_scales,
            _quats=combined_rotations,
        )
        # Check for NaNs and infinite values
        for k, v in gs_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in gaussian {k} at step {self.step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in gaussian {k} at step {self.step}")
                
        return gs_dict


    
    def compute_reg_loss(self):
        loss_dict = {}
        sharp_shape_reg_cfg = self.reg_cfg.get("sharp_shape_reg", None)
        if sharp_shape_reg_cfg is not None:
            w = sharp_shape_reg_cfg.w
            max_gauss_ratio = sharp_shape_reg_cfg.max_gauss_ratio
            step_interval = sharp_shape_reg_cfg.step_interval
            if self.step % step_interval == 0:
                # scale regularization
                scale_exp = self.get_scaling
                scale_reg = torch.maximum(scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1), torch.tensor(max_gauss_ratio)) - max_gauss_ratio
                scale_reg = scale_reg.mean() * w
                loss_dict["sharp_shape_reg"] = scale_reg

        flatten_reg = self.reg_cfg.get("flatten", None)
        if flatten_reg is not None:
            sclaings = self.get_scaling
            min_scale, _ = torch.min(sclaings, dim=1)
            min_scale = torch.clamp(min_scale, 0, 30)
            flatten_loss = torch.abs(min_scale).mean()
            loss_dict["flatten"] = flatten_loss * flatten_reg.w
        
        sparse_reg = self.reg_cfg.get("sparse_reg", None)
        if sparse_reg:
            if (self.cur_radii > 0).sum():
                opacity = torch.sigmoid(self._opacities)
                opacity = opacity.clamp(1e-6, 1-1e-6)
                log_opacity = opacity * torch.log(opacity)
                log_one_minus_opacity = (1-opacity) * torch.log(1 - opacity)
                sparse_loss = -1 * (log_opacity + log_one_minus_opacity)[self.cur_radii > 0].mean()
                loss_dict["sparse_reg"] = sparse_loss * sparse_reg.w

        # compute the max of scaling
        max_s_square_reg = self.reg_cfg.get("max_s_square_reg", None)
        if max_s_square_reg is not None and not self.ball_gaussians:
            loss_dict["max_s_square"] = torch.mean((self.get_scaling.max(dim=1).values) ** 2) * max_s_square_reg.w
        return loss_dict
    
    def load_state_dict(self, state_dict: Dict, **kwargs) -> str:
        N = state_dict["_means"].shape[0]
        self._means = Parameter(torch.zeros((N,) + self._means.shape[1:], device=self.device))
        self._scales = Parameter(torch.zeros((N,) + self._scales.shape[1:], device=self.device))
        self._quats = Parameter(torch.zeros((N,) + self._quats.shape[1:], device=self.device))
        self._features_dc = Parameter(torch.zeros((N,) + self._features_dc.shape[1:], device=self.device))
        self._features_rest = Parameter(torch.zeros((N,) + self._features_rest.shape[1:], device=self.device))
        self._opacities = Parameter(torch.zeros((N,) + self._opacities.shape[1:], device=self.device))
        msg = super().load_state_dict(state_dict, **kwargs)
        return msg
    


    def export_gaussians_to_ply(self, alpha_thresh: float) -> Dict:
        means = self._means
        direct_color = self.colors
        
        activated_opacities = self.get_opacity
        mask = activated_opacities.squeeze() > alpha_thresh
        return {
            "positions": means[mask],
            "colors": direct_color[mask],
        }


    def _initialize_local_coordinates(self):
        self.local_coord = torch.tensor([[1.0, 1.0, 1.0]], device=self.device)
        self.local_coord_basis = torch.eye(3, device=self.device)
        self.local_max = None
        self.local_min = None