import os
import random
from scene.colmap_loader import sceneLoadTypeCallbacks
from scene.camera_utils import cameraList_from_camInfos
from scene.gaussian_model import GaussianModel
from utils.arguments import ModelParams
from utils.general_utils import searchForMaxIteration

class Scene:
    gaussians : GaussianModel
    def __init__(self, model_params : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        self.model_path = model_params.model_path
        self.loaded_iter = None

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        if os.path.exists(os.path.join(model_params.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](model_params.source_path, model_params.eval, model_params.train_test_exp)
        elif os.path.exists(os.path.join(model_params.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](model_params.source_path, model_params.white_background, model_params.eval)
        else:
            assert False, "Wrong source path or Could not recognize scene type!"
        
        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)
        
        self.train_cameras = {}
        self.test_cameras = {}
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        for resolution_scale in resolution_scales:
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, model_params, False)
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, model_params, True)
        
        self.gaussians = gaussians
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
