import os
import argparse

import numpy as np

from h3ds.dataset import H3DS
from h3ds.mesh import Mesh
from h3ds.log import logger
from h3ds.utils import error_to_color

H3DS_PATH = '/home/h3ds/h3ds_v0.2'

# def main(input_dir, method, output_dir):

#     # Create instance of h3ds and download it if not available
#     h3ds = H3DS(H3DS_PATH)

#     # Evaluate `method` on all the scenes used in the h3d-net paper and store the metric
#     metrics_head = {}
#     metrics_face = {}
#     h3ds_scenes = h3ds.scenes(tags={'h3d-net'})
#     eval_dir = os.path.join(output_dir, 'evaluation', method)

#     for scene_id in h3ds_scenes:

#         metrics_head[scene_id] = {}
#         metrics_face[scene_id] = {}

#         h3ds_views_configs = h3ds.default_views_configs(scene_id)
#         for views_config_id in h3ds_views_configs:

#             logger.info(
#                 f'Evaluating {method} reconstruction with {views_config_id} views from scene {scene_id}.'
#             )

#             # Get scene in millimiters
#             mesh_gt, _, _, _ = h3ds.load_scene(scene_id, views_config_id)

#             # Load predicted 3D reconstruction.
#             mesh_pred = Mesh().load(
#                 os.path.join(input_dir, f'{scene_id}_2000_{views_config_id}.ply'))
#             landmarks_pred = None

#             # Evaluate scene. The `landmarks_pred` are optional and, if provided, they will be used
#             # for an initial alignment in the evaluation process. If not provided, it will be assumed
#             # that the predicted mesh is already coarsely aligned with the ground truth mesh.
#             chamfer_gt_pred, chamfer_pred_gt, mesh_gt, mesh_pred_aligned = \
#                 h3ds.evaluate_scene(scene_id, mesh_pred, landmarks_pred)

#             metrics_head[scene_id][views_config_id] = np.mean(chamfer_gt_pred)
#             logger.info(
#                 f' > Chamfer distance full head (mm): {metrics_head[scene_id][views_config_id]}'
#             )
#             mesh_gt.save(
#                 os.path.join(eval_dir, 'full_head',
#                              f'{scene_id}_{views_config_id}_gt.obj'))

#             # The chamfer computed from prediction to ground truth is only provided for
#             # visualization purporses (i.e. heatmaps).
#             mesh_pred_aligned.vertices_color = error_to_color(chamfer_pred_gt,
#                                                               clipping_error=5)
#             mesh_pred_aligned.save(
#                 os.path.join(eval_dir, 'full_head',
#                              f'{scene_id}_{views_config_id}_pred.obj'))

#             # Evaluate reconstruction in the facial region, defined by a sphere of radius 95mm centered
#             # in the tip of the nose. In this case, a more fine alignment is performed, taking into account
#             # only the vertices from this region. This evaluation should be used when assessing methods
#             # that only reconstruct the frontal face area (i.e. Basel Face Bodel)
#             chamfer_gt_pred, chamfer_pred_gt, mesh_gt_region, mesh_pred_aligned = \
#                 h3ds.evaluate_scene(scene_id, mesh_pred, landmarks_pred, region_id='face_sphere')

#             # Note that in both cases we only report the chamfer distane computed from the ground truth
#             # to the prediction, since here we have control over the region where the metric is computed.
#             metrics_face[scene_id][views_config_id] = np.mean(chamfer_gt_pred)
#             logger.info(
#                 f' > Chamfer distance face (mm): {metrics_face[scene_id][views_config_id]}'
#             )
#             mesh_gt_region.save(
#                 os.path.join(eval_dir, 'face_sphere',
#                              f'{scene_id}_{views_config_id}_gt.obj'))

#             # Again, the chamfer computed from prediction to ground truth is only provided for
#             # visualization purporses (i.e. heatmaps).
#             mesh_pred_aligned.vertices_color = error_to_color(chamfer_pred_gt,
#                                                               clipping_error=5)

#             # For improved visualization the predicted mesh is cut to be inside the unit sphere of 95mm.
#             # Ideally one should use landmarks_pred but here we are using landmarks_true because the
#             # landmarks_pred are not available.
#             landmarks_true = h3ds.load_landmarks(scene_id)
#             mask_sphere = np.where(
#                 np.linalg.norm(mesh_pred_aligned.vertices -
#                                mesh_gt.vertices[landmarks_true['nose_tip']],
#                                axis=-1) < 95)
#             mesh_pred_aligned = mesh_pred_aligned.cut(mask_sphere)

#             mesh_pred_aligned.save(
#                 os.path.join(eval_dir, 'face_sphere',
#                              f'{scene_id}_{views_config_id}_pred.obj'))

#     # Show results per view
#     logger.info(f'Average Chamfer Distances for {method} as face / head in mm:')
#     for v in h3ds_views_configs:
#         metric_head = np.mean([metrics_head[s][v] for s in h3ds_scenes])
#         metric_face = np.mean([metrics_face[s][v] for s in h3ds_scenes])
#         logger.info(f'  > views: {v} - error: {metric_face} / {metric_head}')


def main(input_dir, scene_id, views_config_id, method, output_dir):

    # Create instance of h3ds and download it if not available
    h3ds = H3DS(H3DS_PATH)

    # Evaluate `method` on all the scenes used in the h3d-net paper and store the metric
    metrics_head = {}
    metrics_face = {}
    h3ds_scenes = h3ds.scenes(tags={'h3d-net'})
    eval_dir = os.path.join(output_dir, 'evaluation', method)

    # for scene_id in h3ds_scenes:

    metrics_head[scene_id] = {}
    metrics_face[scene_id] = {}

        # h3ds_views_configs = h3ds.default_views_configs(scene_id)
        # for views_config_id in h3ds_views_configs:

    logger.info(
        f'Evaluating {method} reconstruction with {views_config_id} views from scene {scene_id}.'
    )

    # Get scene in millimiters
    mesh_gt, _, _, _ = h3ds.load_scene(scene_id, views_config_id)

    # Load predicted 3D reconstruction.
    mesh_pred = Mesh().load(
        os.path.join(input_dir, f'surface_2000.ply'))
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
    logger.info(f'  > views: {views_config_id} - error: {metric_face} / {metric_head}')



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Exemplifies how to evaluate a method')
    parser.add_argument('--input-dir', help='Input directory containing *.ply files', required=True)
    parser.add_argument('--scene_id', help='The id of the scene to compare to', required=True)
    parser.add_argument('--views_config_id', help='The number of views', required=True)
    parser.add_argument('--method', help='[idr, h3d-net]', default='h3d-net')
    parser.add_argument('--output-dir',
                        help='Output directory to store the results',
                        required=True)

    args = parser.parse_args()
    # main(input_dir=args.input_dir,
    #      method=args.method,
    #      output_dir=args.output_dir)

    main(input_dir=args.input_dir,
         scene_id='1b2a8613401e42a8',
         views_config_id=args.views_config_id,
         method=args.method,
         output_dir=args.output_dir)