import json
import subprocess
import multiprocessing as mp
import os

IDR_PATH = '/home/idr'
EXPS_PATH = os.path.join(IDR_PATH, 'exps')

def find_latest_checkpoint(scene_id):
    scene_paths = [os.path.join(EXPS_PATH, d) for d in os.listdir(EXPS_PATH) if d == f"h3ds_idr_{scene_id}"]#  d.split('_')[3] == scene_id]

    latest_checkpoint = (None, None) # timestamp, checkpoint epoch

    if len(scene_paths) > 0:
        for path in scene_paths:
            runs = os.listdir(path)
            for run in runs:
                # run is timestamp
                run_path = os.path.join(path, run)
                cp_path = os.path.join(run_path, 'checkpoints')
                mp_path = os.path.join(cp_path, 'ModelParameters')
                op_path = os.path.join(cp_path, 'OptimizerParameters')
                sp_path = os.path.join(cp_path, 'SchedulerParameters')
                if os.path.exists(mp_path) and os.path.exists(op_path) and os.path.exists(sp_path):
                    # get last checkpoint 
                    key_fn = lambda fn: int(os.path.splitext(fn)[0])
                    mps = [key_fn(c) for c in os.listdir(mp_path) if c != 'latest.pth']
                    ops = [key_fn(c) for c in os.listdir(op_path) if c != 'latest.pth']
                    sps = [key_fn(c) for c in os.listdir(sp_path) if c != 'latest.pth']

                    mps.sort(reverse=True)
                    ops.sort(reverse=True)
                    sps.sort(reverse=True)
                    if len(mps) == 0 or len(ops) == 0 or len(sps) == 0:
                        continue
                    last_complete = min(max(mps), max(ops), max(sps))
                    if latest_checkpoint[0] is None or last_complete > latest_checkpoint[1]:
                        latest_checkpoint = (run, last_complete)
    return latest_checkpoint

def train(scene_id):
    # check if there are checkpoints
    timestamp, checkpoint = find_latest_checkpoint(scene_id)
    # if done do nothing
    if checkpoint is None:
        # start from scratch
        print(f"training scene {scene_id} from scratch ...")
        with open(f"/home/idr/logs/scan{scene_id}.log", "w+") as log_file:
            print('starting training ...')
            subprocess.call(["python", "training/exp_runner.py", "--conf", "./confs/idr_h3ds.conf", "--scan_id", scene_id], stdout=log_file, stderr=log_file)
    elif checkpoint < 2000:
        print(f"training scene {scene_id} from checkpoint {checkpoint} ...")
        with open(f"/home/idr/logs/scan{scene_id}_cp{checkpoint}.log", "w+") as log_file:
            print('resuming training ...')
            subprocess.call(["python", "training/exp_runner.py", "--conf", "./confs/idr_h3ds.conf", "--scan_id", scene_id, "--is_continue", "--timestamp", timestamp, "--checkpoint", str(checkpoint)], stdout=log_file, stderr=log_file)
    else:
        print(f"scene {scene_id} already trained")
    
# run as 
# nohup python idr_h3ds_train_runner.py &> idr_h3ds_train.log &
if __name__ == '__main__':
    with open('/home/idr/h3ds_scene_config.json') as f:
        conf = json.load(f)
    
    pool = mp.Pool(3)

    print("starting training ...")
    pool.map(train, conf.keys())
    print("done")