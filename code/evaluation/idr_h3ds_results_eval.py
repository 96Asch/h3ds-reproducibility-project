import os
import json
from h3ds.dataset import H3DS
from h3ds.mesh import Mesh
from h3ds.log import logger
from h3ds.utils import error_to_color

# import utils.general as utils

import numpy as np

H3DS_PATH = '/home/h3ds/h3ds_v0.2'
h3ds = H3DS(H3DS_PATH)
scenes = h3ds.scenes(tags={'h3d-net'})

IDR_PATH = '/home/idr'

EXPS_PATH = os.path.join(IDR_PATH, 'exps')

SCENE_CONF_PATH = os.path.join(IDR_PATH, 'h3ds_scene_config.json')

with open(SCENE_CONF_PATH) as f:
    CONF = json.load(f)

# all result directories, e.g. (7, 'dtu_fixed_cameras_7')
RESULTS = [(int(d.split('_')[3]), d) for d in os.listdir(EXPS_PATH)]

def find_mesh_pred(parent):
    """Returns the path of the last predicted mesh"""
    p_dir = os.path.join(EXPS_PATH, parent)
    run_dirs = os.listdir(p_dir)
    for d in run_dirs:
        # get all the .ply files in the plot subdir
        plot_dir = os.path.join(os.path.join(p_dir, d), 'plots')
        candidates = [f for f in os.listdir(plot_dir) if f.endswith('2000.ply')]
        if len(candidates) > 0: # found final .ply
            return os.path.join(plot_dir, candidates[0])
    return None

def get_results():
    """needed data for all finished models"""
    results = []
    for (i, path) in RESULTS:
        mesh_path = find_mesh_pred(path)
        conf = CONF[str(i)]
        if mesh_path is not None:
            results.append((
                conf['scene'],
                conf['config'],
                mesh_path
            ))
    return results

def eval(scene_id, views_config_id, mesh_path, method, output_dir):
    """Evaluates a single scene and returns face/head errors"""
    # utils.mkdir_ifnotexists(output_dir)
    
    # Evaluate `method` on all the scenes used in the h3d-net paper and store the metric
    metrics_head = {}
    metrics_face = {}
    eval_dir = os.path.join(output_dir, 'evaluation', method)

    metrics_head[scene_id] = {}
    metrics_face[scene_id] = {}

    logger.info(
        f'Evaluating {method} reconstruction with {views_config_id} views from scene {scene_id}.'
    )

    # Get scene in millimiters
    mesh_gt, _, _, _ = h3ds.load_scene(scene_id, views_config_id)

    # Load predicted 3D reconstruction.
    mesh_pred = Mesh().load(mesh_path)
    landmarks_pred = None

    # Evaluate scene. The `landmarks_pred` are optional and, if provided, they will be used
    # for an initial alignment in the evaluation process. If not provided, it will be assumed
    # that the predicted mesh is already coarsely aligned with the ground truth mesh.
    chamfer_gt_pred, chamfer_pred_gt, mesh_gt, mesh_pred_aligned = \
        h3ds.evaluate_scene(scene_id, mesh_pred, landmarks_pred)

    metrics_head[scene_id][views_config_id] = np.mean(chamfer_gt_pred)
    logger.info(
        f' > Chamfer distance full head (mm): {metrics_head[scene_id][views_config_id]}'
    )
    mesh_gt.save(
        os.path.join(eval_dir, 'full_head',
                        f'{scene_id}_{views_config_id}_gt.obj'))

    # The chamfer computed from prediction to ground truth is only provided for
    # visualization purporses (i.e. heatmaps).
    mesh_pred_aligned.vertices_color = error_to_color(chamfer_pred_gt,
                                                        clipping_error=5)
    mesh_pred_aligned.save(
        os.path.join(eval_dir, 'full_head',
                        f'{scene_id}_{views_config_id}_pred.obj'))

    # Evaluate reconstruction in the facial region, defined by a sphere of radius 95mm centered
    # in the tip of the nose. In this case, a more fine alignment is performed, taking into account
    # only the vertices from this region. This evaluation should be used when assessing methods
    # that only reconstruct the frontal face area (i.e. Basel Face Bodel)
    chamfer_gt_pred, chamfer_pred_gt, mesh_gt_region, mesh_pred_aligned = \
        h3ds.evaluate_scene(scene_id, mesh_pred, landmarks_pred, region_id='face_sphere')

    # Note that in both cases we only report the chamfer distane computed from the ground truth
    # to the prediction, since here we have control over the region where the metric is computed.
    metrics_face[scene_id][views_config_id] = np.mean(chamfer_gt_pred)
    logger.info(
        f' > Chamfer distance face (mm): {metrics_face[scene_id][views_config_id]}'
    )
    mesh_gt_region.save(
        os.path.join(eval_dir, 'face_sphere',
                        f'{scene_id}_{views_config_id}_gt.obj'))

    # Again, the chamfer computed from prediction to ground truth is only provided for
    # visualization purporses (i.e. heatmaps).
    mesh_pred_aligned.vertices_color = error_to_color(chamfer_pred_gt,
                                                        clipping_error=5)

    # For improved visualization the predicted mesh is cut to be inside the unit sphere of 95mm.
    # Ideally one should use landmarks_pred but here we are using landmarks_true because the
    # landmarks_pred are not available.
    landmarks_true = h3ds.load_landmarks(scene_id)
    mask_sphere = np.where(
        np.linalg.norm(mesh_pred_aligned.vertices -
                        mesh_gt.vertices[landmarks_true['nose_tip']],
                        axis=-1) < 95)
    mesh_pred_aligned = mesh_pred_aligned.cut(mask_sphere)

    mesh_pred_aligned.save(
        os.path.join(eval_dir, 'face_sphere',
                        f'{scene_id}_{views_config_id}_pred.obj'))

    # Show results per view
    logger.info(f'Average Chamfer Distances for {method} as face / head in mm:')
    # for v in h3ds_views_configs:
    metric_head = np.mean(metrics_head[scene_id][views_config_id])
    metric_face = np.mean(metrics_face[scene_id][views_config_id])
    logger.info(f'  > scene: {scene_id} views: {views_config_id} - error: {metric_face} / {metric_head}')
    return metric_face, metric_head

def main(method):
    """runs the evaluation for all trained models and returns the average error for each config"""
    metrics_head = {}
    metrics_face = {}

    for (scene, config, mesh) in get_results():
        if scene not in metrics_head:
            metrics_head[scene] = {}
            metrics_face[scene] = {}
        face, head = eval(scene, config, mesh, method, os.path.join(os.path.join(IDR_PATH, 'evals'), f"{method}_{scene}_{config}"))
        metrics_face[scene][config] = face
        metrics_head[scene][config] = head
    with open(os.path.join(IDR_PATH, f"results_h3ds_{method}.json"), "w+") as f:
        json.dump({
            "face": metrics_face,
            "head": metrics_head
        }, f)

if __name__ == '__main__':
    main('idr')


