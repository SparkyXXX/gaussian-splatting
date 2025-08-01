import torch
import numpy as np
from PIL import Image
from torch import nn
from utils.general_utils import PILtoTorch
from utils.graphics_utils import getWorld2View, getProjectionMatrix, fov2focal

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, image, image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0,
                 train_test_exp = False, is_test_dataset = False, is_test_view = False):
        super(Camera, self).__init__()
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].cuda()
        else: 
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].cuda())

        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        self.original_image = gt_image.clamp(0.0, 1.0).cuda()
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale
        self.world_view_transform = torch.tensor(getWorld2View(R, T, trans, scale)).transpose(0, 1).cuda()
        temp_projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.projection_matrix = (self.world_view_transform.unsqueeze(0).bmm(temp_projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

WARNED = False
def cameraList_from_camInfos(cam_infos, resolution_scale, model_params, is_test_dataset):
    camera_list = []
    for id, cam_info in enumerate(cam_infos):
        image = Image.open(cam_info.image_path)
        orig_w, orig_h = image.size
        if model_params.resolution in [1, 2, 4, 8]:
            resolution = round(orig_w/(resolution_scale * model_params.resolution)), round(orig_h/(resolution_scale * model_params.resolution))
        else:
            if model_params.resolution == -1:
                if orig_w > 1600:
                    global WARNED
                    if not WARNED:
                        print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                            "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                        WARNED = True
                    global_down = orig_w / 1600
                else:
                    global_down = 1
            else:
                global_down = orig_w / model_params.resolution

            scale = float(global_down) * float(resolution_scale)
            resolution = (int(orig_w / scale), int(orig_h / scale))
        camera = Camera(resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=image, image_name=cam_info.image_name, uid=id,
                  train_test_exp=model_params.train_test_exp, is_test_dataset=is_test_dataset, is_test_view=cam_info.is_test)
        camera_list.append(camera)
    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry