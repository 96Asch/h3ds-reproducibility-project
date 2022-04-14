import trimesh
import sklearn.metrics as metrics
import numpy as np
from pyhocon import ConfigFactory
import os

def find_closest_vertex_index(mesh, pos):
    distances = metrics.pairwise_distances(mesh.vertices, pos.reshape(1, -1), metric='euclidean')
    return np.argmin(distances)

    # idx, dist = None, None
    # for i, v in enumerate(mesh.vertices):
    #     d = sklearn.metrics.pairwise_distances(v, v)
    #     if dist is None or d < dist:
    #         idx, dist = i, v
    return idx

def parse_landmarks(filename):
    return ConfigFactory.parse_file(filename)

def generate_landmarks(mesh_path, landmark_positions, filename='landmarks.txt'):
    mesh = trimesh.load(mesh_path)

    with open(filename, 'w+') as f:
        for lm, pos in landmark_positions.items():
            idx = find_closest_vertex_index(mesh, np.array(pos))
            f.write(f"{lm} {idx}" + "\n")

if __name__ == '__main__':

    mesh_path = '/home/idr/exps/iccv_h3dnet_3_views_0/2022_04_11_16_31_49/plots/surface_2000.ply'
    landmark_positions = '/home/idr/exps/iccv_h3dnet_3_views_0/2022_04_11_16_31_49/plots/landmarks.conf'

    conf = parse_landmarks(landmark_positions)

    generate_landmarks(mesh_path, conf['landmarks'], os.path.join(os.path.dirname(mesh_path), 'landmarks.txt'))

    landmark_positions = {
        'right_eye': (0, 0, 0),
        'left_eye': (0, 0, 0),
        'nose_base': (0, 0, 0),
        'right_lips': (0, 0, 0),
        'left_lips': (0, 0, 0),
        'nose_tip': (0, 0, 0),
    }


    
