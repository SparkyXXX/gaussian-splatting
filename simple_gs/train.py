import os
import sys
import uuid
import torch
from tqdm import tqdm
from random import randint
from setproctitle import setproctitle
from argparse import ArgumentParser, Namespace
from torch.utils.tensorboard import SummaryWriter
from renderer import render
from fused_ssim import fused_ssim
from scene import Scene, GaussianModel
from utils.graphics_utils import psnr
from utils.general_utils import l1_loss, safe_state
from utils.arguments import ModelParams, PipelineParams, OptimizationParams

def training(model_params, opt_params, pipe_params, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(model_params)
    gaussians = GaussianModel(model_params.sh_degree, opt_params.optimizer_type)
    scene = Scene(model_params, gaussians)
    gaussians.training_setup(opt_params)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt_params)

    use_sparse_adam = opt_params.optimizer_type == "sparse_adam"
    bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt_params.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt_params.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe_params.debug = True
        bg = torch.rand((3), device="cuda") if opt_params.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe_params, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - opt_params.lambda_dssim) * Ll1 + opt_params.lambda_dssim * (1.0 - ssim_value)
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                current_mem = torch.cuda.memory_allocated() / (1024 **3)
                peak_mem = torch.cuda.max_memory_allocated() / (1024** 3)
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}", 
                                          "current_gpu_mem": f"{current_mem:.2f}GB", "peak_gpu_mem": f"{peak_mem:.2f}GB"})
                progress_bar.update(10)
            if iteration == opt_params.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), 
                            testing_iterations, scene, render, (pipe_params, background, 1.))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt_params.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration > opt_params.densify_from_iter and iteration % opt_params.densification_interval == 0:
                    size_threshold = 20 if iteration > opt_params.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt_params.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                if iteration % opt_params.opacity_reset_interval == 0 or (model_params.white_background and iteration == opt_params.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt_params.iterations:
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(model_params):    
    if not model_params.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        model_params.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = SummaryWriter(os.path.join(model_params.model_path, "logs"))
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, 
                    testing_iterations, scene : Scene, renderFunc, otherRenderArgs):
    current_mem = torch.cuda.memory_allocated() / (1024 **3)
    peak_mem = torch.cuda.max_memory_allocated() / (1024** 3)
    tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
    tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
    tb_writer.add_scalar('iter_time', elapsed, iteration)
    tb_writer.add_scalar('gpu_memory/current_GB', current_mem, iteration)
    tb_writer.add_scalar('gpu_memory/peak_GB', peak_mem, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *otherRenderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.cuda(), 0.0, 1.0)
                    if idx < 5:
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
        tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def get_gpu_memory_usage():
    current = torch.cuda.memory_allocated() / (1024 **3)
    peak = torch.cuda.max_memory_allocated() / (1024** 3)
    torch.cuda.reset_peak_memory_stats()
    return current, peak

if __name__ == "__main__":
    setproctitle("Ruixiang's Work 😆")
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    parser = ArgumentParser(description="Training script parameters")
    mp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--debug_from', type=int, default=-1)
    args = parser.parse_args()
    args.save_iterations.append(args.iterations)
    
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(mp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, 
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    print("\nTraining complete.")