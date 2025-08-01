import math
import torch
from scene.gaussian_model import GaussianModel
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def render(viewpoint_camera, gaussian : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0):
    # Render the scene. Background tensor (bg_color) must be on GPU!
    screenspace_points = torch.zeros_like(gaussian.get_xyz, dtype=gaussian.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.projection_matrix,
        sh_degree=gaussian.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth_image = rasterizer(
        means3D = gaussian.get_xyz,
        means2D = screenspace_points,
        dc = gaussian.get_features_dc,
        shs = gaussian.get_features_rest,
        colors_precomp = None,
        opacities = gaussian.get_opacity,
        scales = gaussian.get_scaling,
        rotations = gaussian.get_rotation,
        cov3D_precomp = None)
        
    rendered_image = rendered_image.clamp(0, 1).cuda()
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }
    return out