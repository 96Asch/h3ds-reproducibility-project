import subprocess
from operator import indexOf
import os
import shutil
from h3ds.dataset import H3DS
import numpy as np
from preprocess_cameras import get_normalization
import utils.general as utils
import json

H3DS_PATH = '/home/h3ds/h3ds_v0.2'
NAME = 'h3ds_3'
TARGET_DIR = os.path.join('/home/idr/data/', NAME)

h3ds = H3DS(path=H3DS_PATH)

scenes = h3ds.scenes(tags={'h3d-net'}) # returns the scenes used in H3D-Net paper

CONFIG = {}
CONFIG_PATH = f'/home/idr/{NAME}_scene_config.json'

scan_count = 0

def create_idr_scene(scene, config):
    global scan_count
    scan_id = scan_count
    scan_count += 1

    print(f"Creating data for {scene} ({config} images) in dir scan{scan_id}")

    src_dir = os.path.join(H3DS_PATH, scene)
    src_img_dir = os.path.join(src_dir, "image")
    src_mask_dir = os.path.join(src_dir, "mask")

    out_dir = os.path.join(TARGET_DIR, f"scan{scan_id}")
    utils.mkdir_ifnotexists(out_dir)

    img_dir = os.path.join(out_dir, "image")
    mask_dir = os.path.join(out_dir, "mask")
    utils.mkdir_ifnotexists(img_dir)
    utils.mkdir_ifnotexists(mask_dir)

    CONFIG[scan_id] = {
        "scene": scene,
        "config": config,
        "dir": out_dir
    }

    # 1. move cameras.npz file over
    shutil.copy(os.path.join(src_dir, "cameras.npz"), os.path.join(out_dir, "cameras.npz"))

    # subprocess.call(["python", "preprocess_cameras.py", "--source_dir", src_dir])
    # shutil.move(os.path.join(out_dir, "cameras.npz"), os.path.join(out_dir, "cameras.npz.bak"))
    # shutil.copy(os.path.join(src_dir, "cameras_new.npz"), os.path.join(out_dir, "cameras.npz"))

    # 2. save mesh, images & masks
    img_indices = h3ds.helper._config['scenes'][scene]['default_views_configs'][config]
    for i in img_indices:
        fn = f"img_{i:04d}.jpg"
        shutil.copy(os.path.join(src_img_dir, fn), os.path.join(img_dir, fn))
        fn = f"mask_{i:04d}.png"
        shutil.copy(os.path.join(src_mask_dir, fn), os.path.join(mask_dir, fn))

    # _, images, masks, cameras = h3ds.load_scene(scene_id=scene, views_config_id=config)


    
    # utils.mkdir_ifnotexists(img_dir)
    # for (i, img) in enumerate(images):
    #     img.save(os.path.join(img_dir, f"img_{i:04d}.jpg"), 'jpeg')

    
    # utils.mkdir_ifnotexists(mask_dir)
    # for (i, mask) in enumerate(masks):
    #     mask.save(os.path.join(mask_dir, f"mask_{i:04d}.jpg"), 'jpeg')

    # np.savez(os.path.join(out_dir, 'cameras'), cameras)

if __name__ == '__main__':
    utils.mkdir_ifnotexists(TARGET_DIR)
    for scene in scenes:
        for config in h3ds.default_views_configs(scene_id=scene): # '3', '4', '8', '16' and '32'
            create_idr_scene(scene, config)
    with open(CONFIG_PATH, 'w') as f:
        json.dump(CONFIG, f, indent=2)
