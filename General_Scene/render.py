import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.gaussian_particle import GaussianParticle
from scene.cameras import Camera
from PIL import Image
import numpy as np
import json

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def interpolate_camera(cam1, cam2, alpha):
    interpolated_center = (1 - alpha) * cam1.camera_center + alpha * cam2.camera_center
    interpolated_wvt = (1 - alpha) * cam1.world_view_transform + alpha * cam2.world_view_transform
    interpolated_proj = (1 - alpha) * cam1.full_proj_transform + alpha * cam2.full_proj_transform

    dummy_R = np.eye(3)
    dummy_T = np.zeros(3)
    width, height = cam1.image_width, cam1.image_height
    blank_image_pil = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))

    new_cam = Camera(
        resolution=(width, height),
        colmap_id=-1,
        R=dummy_R,
        T=dummy_T,
        FoVx=cam1.FoVx,
        FoVy=cam1.FoVy,
        image=blank_image_pil,
        invdepthmap=None,
        depth_params=None,
        image_name=f"interpolated_{alpha:.2f}",
        uid=9999,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device=cam1.data_device
    )

    new_cam.camera_center = interpolated_center
    new_cam.world_view_transform = interpolated_wvt
    new_cam.full_proj_transform = interpolated_proj

    return new_cam


def load_particle_config(gaussians, weather_type="snow", config_path=None):
    if not os.path.exists(config_path):
        print("No weather config found.")
        return None
    with open(config_path, "r") as f:
        weather_config = json.load(f)
    if weather_type not in weather_config:
        print(f"Warning: weather type '{weather_type}' not found in config.")
        return None
    return GaussianParticle(
        config=weather_config[weather_type],
        scene_extent=7,
        sh_degree=gaussians.max_sh_degree
    )


def render_interpolated_segments(model_path, iteration, cam_list, gaussians, pipeline, background,
                                 train_test_exp, weather_type,separate_sh, particle=None, delta_time=1, frames_per_segment=20):
    render_path = os.path.join(model_path, "interpolated", f"ours_{iteration}", "renders_"+weather_type)
    makedirs(render_path, exist_ok=True)

    idx = 0
    for i in tqdm(range(len(cam_list) - 1), desc="Rendering interpolated"):
        cam1, cam2 = cam_list[i], cam_list[i + 1]
        for j in range(frames_per_segment):
            alpha = j / (frames_per_segment - 1)
            interp_cam = interpolate_camera(cam1, cam2, alpha)

            particle_gaussians = particle.get_static_gaussians() if particle else None
            rendering = render(interp_cam, gaussians, pipeline, background,
                               use_trained_exp=train_test_exp,
                               separate_sh=separate_sh,
                               pg=particle_gaussians)["render"]

            if particle:
                particle.update_positions(delta_time)

            if train_test_exp:
                rendering = rendering[..., rendering.shape[-1] // 2:]

            torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:05d}.png"))
            del rendering
            torch.cuda.empty_cache()
            idx += 1



def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams,
                skip_train: bool, skip_test: bool, separate_sh: bool, fps: int, weather_type=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Load optional weather system
        particle = load_particle_config(gaussians, weather_type=weather_type,config_path="weather_config.json")
        delta_time = 1
        interp_fps =  fps
        train_cameras = scene.getTrainCameras()
        render_interpolated_segments(
            model_path=dataset.model_path,
            iteration=scene.loaded_iter,
            cam_list=train_cameras,
            gaussians=gaussians,
            pipeline=pipeline,
            background=background,
            train_test_exp=dataset.train_test_exp,
            weather_type=weather_type,
            separate_sh=separate_sh,
            particle=particle,
            delta_time=delta_time,
            frames_per_segment=interp_fps
        )




if __name__ == "__main__":
    parser = ArgumentParser(description="Rendering script with optional weather support")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fps", default=20,type=int, help="Frames per second for interpolation segments")

    parser.add_argument("--weather", type=str, default=None, help="Weather type to render (e.g., snow, rain, fog)")
   
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args),
                args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE, args.fps, weather_type=args.weather)
