import os
import shutil
import logging
from argparse import ArgumentParser

use_gpu = 1
colmap_command = "colmap"

def feat_extracton(args):
    feat_extracton_cmd = (
        f"{colmap_command} feature_extractor "
        f"--database_path {args.source_path}/distorted/database.db "
        f"--image_path {args.source_path}/input "
        f"--ImageReader.single_camera 1 "
        f"--ImageReader.camera_model {args.camera} "
        f"--SiftExtraction.use_gpu {use_gpu}"
    )
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error("feat extracton failed!")
        exit(1)

def feat_matching(args):
    feat_matching_cmd = (
        f"{colmap_command} exhaustive_matcher "
        f"--database_path {args.source_path}/distorted/database.db "
        f"--SiftMatching.use_gpu {use_gpu}"
    )
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error("feature matching failed!")
        exit(1)

def mapper(args):
    mapper_cmd = (
        f"{colmap_command} mapper "
        f"--database_path {args.source_path}/distorted/database.db "
        f"--image_path {args.source_path}/input "
        f"--output_path {args.source_path}/distorted/sparse "
        f"--Mapper.ba_global_function_tolerance=0.000001"
    )
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error("mapper failed!")
        exit(1)

def img_undistort(args):
    img_undistort_cmd = (
        f"{colmap_command} image_undistorter "
        f"--image_path {args.source_path}/input "
        f"--input_path {args.source_path}/distorted/sparse/0 "
        f"--output_path {args.source_path} "
        f"--output_type COLMAP"
    )
    exit_code = os.system(img_undistort_cmd)
    if exit_code != 0:
        logging.error("image undistort failed!")
        exit(1)

def handle_COLMAP_tempfile():
    files = os.listdir(args.source_path + "/sparse")
    os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(args.source_path, "sparse", file)
        destination_file = os.path.join(args.source_path, "sparse", "0", file)
        shutil.move(source_file, destination_file)

    for name in ["stereo"]:
        path = os.path.join(args.source_path, name)
        if os.path.exists(path):
            shutil.rmtree(path)

    for name in os.listdir(args.source_path):
        if name.startswith("run-colmap-") and name.endswith(".sh"):
            os.remove(os.path.join(args.source_path, name))

if __name__ == "__main__":
    parser = ArgumentParser("COLMAP Converter")
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--skip_matching", action='store_true')
    parser.add_argument("--camera", default="OPENCV", type=str)
    args = parser.parse_args()

    if args.skip_matching:
        img_undistort(args)
        handle_COLMAP_tempfile()
    else:
        os.makedirs(os.path.join(args.source_path, "distorted/sparse"), exist_ok=True)
        feat_extracton(args)
        feat_matching(args)
        mapper(args)
        img_undistort(args)
        handle_COLMAP_tempfile()

    print("Done!")