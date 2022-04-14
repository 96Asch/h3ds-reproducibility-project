from operator import indexOf
import os
import shutil
from h3ds.dataset import H3DS
from preprocess_cameras import get_normalization
import utils.general as utils
import json

H3DS_PATH = '/home/h3ds/h3ds_v0.2'
TARGET_DIR = '/home/idr/data'

h3ds = H3DS(path=H3DS_PATH)

scenes = h3ds.scenes(tags={'h3d-net'}) # returns the scenes used in H3D-Net paper

CONFIG = {}
CONFIG_PATH = '/home/idr/h3ds_scene_config.json'

function generateDataset():
    pass

if __name__ == '__main__':
