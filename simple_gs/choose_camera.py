import os
import torch
from tqdm import tqdm
import torchvision
from os import makedirs
from setproctitle import setproctitle
from renderer import render
from utils.arguments import PipelineParams
from scene import Scene, GaussianModel
from argparse import ArgumentParser

class DummyPipeArgs:
    def __init__(self):
        self.debug = False
        self.antialiasing = False

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def ratio_calc(img1, img2):
    diff = torch.abs(img1 - img2)
    total_pixels = diff.numel() // 3
    significant_diff_pixels = torch.sum((diff > 1e-3).any(dim=0)).item()
    ratio = significant_diff_pixels / total_pixels
    return ratio

def filter_cameras_by_visibility_with_img_save(full_gaussians, aabb_gaussians, viewpoint_stack_full, threshold=0.1, start=1, end=28):
    dummy_pipe_args = DummyPipeArgs()
    pipeline_params = PipelineParams(ArgumentParser())
    pipe = pipeline_params.extract(Namespace())
    pipe.debug = dummy_pipe_args.debug
    pipe.antialiasing = dummy_pipe_args.antialiasing
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    viewpoint_stack_block = []
    with torch.no_grad():
        full_save_path = "./test_render/full"
        aabb_save_path = "./test_render/aabb"
        makedirs(full_save_path, exist_ok=True)
        makedirs(aabb_save_path, exist_ok=True)

        # for idx, cam in enumerate(tqdm(viewpoint_stack_full, desc="Rendering progress")):
        for idx in tqdm(range(start, end)):
            full_render = render(viewpoint_stack_full[idx], full_gaussians, pipe, background)["render"]
            aabb_render = render(viewpoint_stack_full[idx], aabb_gaussians, pipe, background)["render"]
            torchvision.utils.save_image(full_render, os.path.join(full_save_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(aabb_render, os.path.join(aabb_save_path, '{0:05d}'.format(idx) + ".png"))
            ratio = ratio_calc(full_render, aabb_render)
            if ratio > threshold:
                viewpoint_stack_block.append(viewpoint_stack_full[idx])
            del full_render, aabb_render
            torch.cuda.empty_cache()

    return viewpoint_stack_block

if __name__ == "__main__":
    class DummyModelParams:
        def __init__(self, source_path):
            self.source_path = source_path
            self.model_path = os.path.join(self.source_path, "output")
            self.resolution = -1
            self.white_background = False
            self.sh_degree = 3
            self.train_test_exp = False
            self.eval = False

    setproctitle("Ruixiang's Work 😆")
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    full_mp = DummyModelParams("./data/full")
    full_gaussians = GaussianModel(full_mp.sh_degree, optimizer_type="sparse_adam")
    full_scene = Scene(full_mp, full_gaussians, load_iteration=-1, shuffle=False)
    
    aabb_mp = DummyModelParams("./data/aabb")
    aabb_gaussians = GaussianModel(aabb_mp.sh_degree, optimizer_type="sparse_adam")
    aabb_scene = Scene(aabb_mp, aabb_gaussians, load_iteration=-1, shuffle=False)
    
    # filtered_cameras = filter_cameras_by_visibility(full_gaussians, aabb_gaussians, full_scene.getTrainCameras(), threshold=0.1)
    filtered_cameras = filter_cameras_by_visibility_with_img_save(full_gaussians, aabb_gaussians, 
                                                                  full_scene.getTrainCameras(), threshold=0.1, start=1, end=14)
    filtered_cameras = filter_cameras_by_visibility_with_img_save(full_gaussians, aabb_gaussians, 
                                                                  full_scene.getTrainCameras(), threshold=0.1, start=16, end=28)
    # print(f"Filtered cameras: {len(filtered_cameras)}")