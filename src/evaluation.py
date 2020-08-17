import mesh_to_sdf
import matplotlib.pyplot as plt
from drawnow import drawnow
import numpy as np
import argparse
import pickle
import trimesh
from klampt.io import resource
from scipy.spatial.transform.rotation import Rotation
import os
import sys
sys.path.append("./3rdparty/bop_toolkit")
sys.path.append("./3rdparty/bop_toolkit/bop_toolkit_lib")
import bop_toolkit_lib.inout as inout
import bop_toolkit_lib.misc as misc
from bop_toolkit_lib.renderer_py import RendererPython
import bop_toolkit_lib.pose_error as error
sys.path.append(".")
sys.path.append("./src")
import src.sampling as sampling
import src.plausibility as plausibility


YCBV_PATH = "/PATH_TO/BOP19/ycbv"
assert YCBV_PATH != "/PATH_TO/BOP19/ycbv"  # set to your model directory

# load model meta data
DATA_PATH = os.path.join(os.path.abspath(os.getcwd()), "data")
models_info = inout.load_json(os.path.join(YCBV_PATH, f"models_eval/models_info.json"), keys_to_int=True)
models_meta = inout.load_json(os.path.join(DATA_PATH, f"models_meta.json"), keys_to_int=True)
obj_roff = np.array([models_meta[obj_id]['R_to_canonical'] for obj_id in range(1, 22)]).reshape(21, 3, 3)
obj_toff = np.array([models_meta[obj_id]['t_to_canonical'] for obj_id in range(1, 22)]).reshape(21, 3)


def init(target):
    """
    Initialize scene for given [target] object to evaluate pose-error functions. Prepare rendering and evaluation.
    """

    renderer = RendererPython(640, 480, bg_color=(1.0, 1.0, 1.0, 1.0), shading='flat')

    # base scene (static support)
    ground_volume = resource.get(f"{DATA_PATH}/cube.ply").convert('VolumeGrid')
    renderer.add_object(0, f"{DATA_PATH}/cube.ply")

    env = [ground_volume]
    env_meshes = [trimesh.load(f"{DATA_PATH}/cube.ply")]
    env_ids = [0]
    env_Ts = [np.eye(4)]

    # initialize remaining scene objects based on target object
    Tgt = np.eye(4)
    if target == 'bowl':
        # bowl
        obj_id = 13
        cloud, ply, mesh = get_tgt_model(obj_id, renderer)
        Tgt[:3, 3] = np.array([0, 0, -models_info[obj_id]['min_z'] - obj_toff[obj_id - 1][2]])
        t_obj_offset = [-125, 0, 0]
    elif target == 'marker':
        # marker
        obj_id = 18
        cloud, ply, mesh = get_tgt_model(obj_id, renderer)
        Tgt[:3, :3] = Rotation.from_euler('xz', [21.5, 90.0], degrees=True).as_dcm()
        Tgt[:3, 3] = [33.42, 0, 30.14]
        t_obj_offset = [-300, 25, 50]

        # banana
        env_model, env_mesh = get_env_model(10, renderer)
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler('xyz', [0, 11.8, 0], degrees=True).as_dcm()
        T[:3, 3] = [0, 0, 16]
        env_mesh.apply_transform(T)

        env.append(env_model)
        env_meshes.append(env_mesh)
        env_ids.append(-10)
        env_Ts.append(T.copy())
    elif target == 'clamp':
        # clamp
        obj_id = 19
        cloud, ply, mesh = get_tgt_model(obj_id, renderer)
        Tgt[:3, :3] = Rotation.from_euler('xyz', [-6, -1.4, -90], degrees=True).as_dcm()
        Tgt[:3, 3] = [9.59, 0, 91.74]
        t_obj_offset = [-150, 0, 65]

        # pudding
        env_model, env_mesh = get_env_model(7, renderer)
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler('xyz', [-88.4, 0.4, 90], degrees=True).as_dcm()
        T[:3, 3] = [-61.66, 1.24, 44.27]
        env_mesh.apply_transform(T)

        env.append(env_model)
        env_meshes.append(env_mesh)
        env_ids.append(-7)
        env_Ts.append(T.copy())

        # jello
        env_model, env_mesh = get_env_model(8, renderer)
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler('xyz', [-90, 90, 180], degrees=True).as_dcm()
        T[:3, 3] = [74.90, -1.09, 36.37]
        env_mesh.apply_transform(T)

        env.append(env_model)
        env_meshes.append(env_mesh)
        env_ids.append(-8)
        env_Ts.append(T.copy())
    else:
        raise ValueError("'target' must be one of 'bowl', 'marker' or 'clamp'.")
    Tsyms = misc.get_symmetry_transformations(models_info[obj_id], 0.01)  # symmetry

    # prepare renderer: intrinsics and extrinsics
    K = np.array([1066.778, 0.0, 312.9869, 0.0, 1067.487, 241.3109, 0.0, 0.0, 1.0]).reshape(3, 3)
    Rview = Rotation.from_euler('zx', [-90 if target == 'bowl' else 0, 90 if target == 'bowl' else 110],
                                degrees=True).as_dcm()
    tview = np.array([-t_obj_offset[1], t_obj_offset[2], 850+t_obj_offset[0]])

    return env_meshes, env, env_ids, env_Ts, mesh, cloud, ply, Tgt, Tsyms, renderer, K, Rview, tview


def get_tgt_model(obj_id, renderer):
    """
    Load the target object with given [obj_id] as Klampt PointCloud, numpy array, mesh and add it to renderer.
    """

    path_ply = os.path.join(YCBV_PATH, f"models_eval/obj_{obj_id:06d}.ply")
    cloud = resource.get(path_ply).convert('PointCloud')
    cloud.transform(obj_roff[obj_id - 1].T.reshape(9).tolist(), obj_toff[obj_id - 1].reshape(3).tolist())

    ply = inout.load_ply(path_ply)
    ply['pts'] = (obj_roff[obj_id - 1] @ ply['pts'].T + obj_toff[obj_id - 1].reshape(3, 1)).T

    mesh = trimesh.load(path_ply)
    T = np.eye(4)
    T[:3, :3] = obj_roff[obj_id - 1]
    T[:3, 3] = obj_toff[obj_id - 1]
    mesh = mesh.apply_transform(T)

    renderer.add_object(obj_id, os.path.join(YCBV_PATH, f"models/obj_{obj_id:06d}.ply"),
                        offset=[obj_roff[obj_id - 1], obj_toff[obj_id - 1]])
    return cloud, ply, mesh


def get_env_model(env_id, renderer, volume=True, resolution=1):
    """
    Load the scene object with given [env_id] as Klampt VolumeGrid, mesh and add it to renderer.
    """

    path = os.path.join(YCBV_PATH, f"models_eval/obj_{env_id:06d}.ply")
    env_model = resource.get(path)
    env_model.transform(obj_roff[env_id - 1].T.reshape(9).tolist(), obj_toff[env_id - 1].reshape(3).tolist())
    if volume:
        env_model = env_model.convert('VolumeGrid', resolution)

    mesh = trimesh.load(path)
    T = np.eye(4)
    T[:3, :3] = obj_roff[env_id - 1]
    T[:3, 3] = obj_toff[env_id - 1]
    mesh = mesh.apply_transform(T)

    path = os.path.join(YCBV_PATH, f"models/obj_{env_id:06d}.ply")
    renderer.add_object(-env_id, path, offset=[obj_roff[env_id - 1], obj_toff[env_id - 1]])
    return env_model, mesh


def render_sequence(renderer, env_ids, env_Ts, obj_id, Re, te, Rview, tview, K):
    """
    Render given scene objects (env) and target object under all given pose estimates [Re, te].
    """

    # render environment models
    env_rgbd = {'rgb': np.ones((480, 640, 3), dtype=np.uint8)*255, 'depth': np.ones((480, 640), dtype=np.float32)*1e6}
    for env_id, env_T in zip(env_ids, env_Ts):
        R = Rview @ env_T[:3, :3]
        t = Rview @ env_T[:3, 3] + tview

        frame = renderer.render_object(env_id, R, t, K[0, 0], K[1, 1], K[0, 2], K[1, 2])

        z_test = np.logical_or(np.logical_and(frame['depth'] < env_rgbd['depth'], frame['depth'] > 0),
                               env_rgbd['depth'] == 0)
        env_rgbd['rgb'][z_test > 0] = frame['rgb'][z_test > 0]
        env_rgbd['depth'][z_test > 0] = frame['depth'][z_test > 0]

    # render estimated poses from sequence; compose with environment rendering
    frames = []
    for R, t in zip(Re, te):
        R = Rview @ R
        t = Rview @ t + tview
        frame = renderer.render_object(obj_id, R, t, K[0, 0], K[1, 1], K[0, 2], K[1, 2])

        z_test = np.logical_or(np.logical_and(frame['depth'] < env_rgbd['depth'], frame['depth'] > 0),
                               env_rgbd['depth'] == 0)

        rgb = env_rgbd['rgb'].copy()
        rgb[z_test > 0] = frame['rgb'][z_test > 0]
        rgb[z_test == 0] = np.uint8(frame['rgb'][z_test == 0] * 0.3 + rgb[z_test == 0] * 0.7)
        depth = env_rgbd['depth'].copy()
        depth[z_test > 0] = frame['depth'][z_test > 0]
        frame = {'rgb': rgb, 'depth': depth}

        frames.append(frame)
    return frames


def evaluate_sequence(renderer, obj_id, Re, te, Rview, tview, K, pts, Rgt, tgt, syms, depth_test, Rs, ts):
    """
    Evaluate target object under all given pose estimates [Re, te]. Values for MSSD, MSPD, VSD and ours are computed.
    """

    # BOP toolkit settings and parameters
    delta, taus, normalized_by_diameter, diameter, r2 = 15, [0.2], True, models_info[obj_id]['diameter'],\
                                                        models_meta[obj_id]['frobenius_scale']
    im_width = 640
    theta_adi, theta_mssd, theta_mspd, theta_vsd = 0.1, 0.2, 20, 0.3
    metric, phi = 'frobenius', 20

    # ground-truth pose in camera space
    Rgt_cam = Rview @ Rgt
    tgt_cam = Rview @ tgt + tview

    Rps = []
    tps = []
    errors = []
    for R, t in zip(Re, te):
        # pose in camera space
        Rcam = Rview @ R
        tcam = Rview @ t + tview

        # reshape estimate for BOP toolkit
        tgt = tgt.reshape(3, 1)
        tgt_cam = tgt_cam.reshape(3, 1)
        t = t.reshape(3, 1)
        tcam = tcam.reshape(3, 1)

        # compute baseline pose-error functions
        mssd = error.mssd(R, t, Rgt, tgt, pts, syms) / diameter
        mspd = error.mspd(Rcam, tcam, Rgt_cam, tgt_cam, K, pts, syms) * (640/im_width)
        vsd = np.mean(error.vsd(Rcam, tcam, Rgt_cam, tgt_cam, depth_test, K, delta, taus,
                                normalized_by_diameter, diameter, renderer, obj_id))

        # ours: match estimate to closest plausible pose
        Rp, tp, dp = plausibility.find_closest(R, t, Rs, ts, metric, pts, syms, r2)

        # matched plausible pose in camera space; reshape for BOP toolkit
        Rp_cam = Rview @ Rp
        tp_cam = (Rview @ tp + tview).reshape(3, 1)
        tp = tp.reshape(3, 1)

        # ours: compute implausibility and physical plausibility error terms
        implausibility = np.clip(dp / phi, 0, 1)

        mssd_pp = mssd + theta_mssd * implausibility
        mspd_pp = mspd + theta_mspd * implausibility
        vsd_pp = vsd + theta_vsd * implausibility

        mssd_op = mssd + error.mssd(R, t, Rp, tp, pts, syms) / diameter
        mspd_op = mspd + error.mspd(Rcam, tcam, Rp_cam, tp_cam, K, pts, syms) * (640/im_width)
        vsd_op = vsd + np.mean(error.vsd(Rcam, tcam, Rp_cam, tp_cam, depth_test, K, delta, taus,
                                         normalized_by_diameter, diameter, renderer, obj_id))

        errors.append([mssd, mspd, vsd, implausibility,
                       mssd_pp, mspd_pp, vsd_pp,
                       mssd_op, mspd_op, vsd_op])
        Rps.append(Rp)
        tps.append(tp)
    return np.array(errors), np.array(Rps).reshape(-1, 3, 3), np.array(tps).reshape(-1, 3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Physically Plausible Poses')
    parser.add_argument('--target', type=str, default='bowl', choices=['bowl', 'marker', 'clamp'],
                        help='Target object for evaluation. Selects the scene accordingly.')
    parser.add_argument('--mode', type=str, default='rotation', choices=['rotation', 'translation'],
                        help='Applies either rotation or translation error to ground-truth pose.')
    parser.add_argument('--t_per_frame', type=float, default=0.5,
                        help='Duration for which the visualization of each step is displayed.')
    args = parser.parse_args()

    # === INIT SCENE
    print("Initializing scene and estimates...")
    if args.target == 'bowl':
        obj_id = 13
        PHI_t, PHI_r, DELTA_t, DELTA_r = 50, 180, 5, 15
    else:
        if args.target == 'marker':
            obj_id = 18
        elif args.target == 'clamp':
            obj_id = 19
        else:
            raise ValueError("'target' must be one of 'bowl', 'marker' or 'clamp'.")
        PHI_t, PHI_r, DELTA_t, DELTA_r = 20, 20, 4, 4
    EPS, FRICTION = 2, 0.1

    env_meshes, env, env_ids, env_Ts, mesh, cloud, ply, Tgt, Tsyms, renderer, K, Rview, tview = init(args.target)
    ground_volume = env[0]
    Rgt, tgt = Tgt[:3, :3], Tgt[:3, 3]

    # === LOAD PLAUSIBLE POSES
    with open(f"{DATA_PATH}/samples_{args.target}.pkl", 'rb') as file:
        samples = pickle.load(file)
    Rs, ts = samples['Rs'], samples['ts']

    # === RENDER GT SCENE
    frame_gt = render_sequence(renderer, env_ids, env_Ts, obj_id, [Rgt], [tgt], Rview, tview, K)[0]

    # === GENERATE POSE ESTIMATES
    if args.mode == 'rotation':
        if args.target == 'bowl':  # around x-axis
            Re, te = sampling.samples_axis_rotation(ply['pts'].T, PHI_r, [DELTA_r, 0, 0], snap=True)
        elif args.target == 'marker':  # around y-axis
            Re, te = sampling.samples_axis_rotation(ply['pts'].T, PHI_r, [0, DELTA_r, 0], snap=True)

            # apply GT rotation and adapt z-translation accordingly
            Re = np.array([re @ Rgt for re in Re]).reshape(-1, 3, 3)
            te = np.array([[0, 0, -(R @ ply['pts'].T).min(axis=1)[2]] for R in Re])
            te += np.array([Tgt[0, 3], Tgt[1, 3], Tgt[2, 3] + (Tgt[:3, :3] @ ply['pts'].T).min(axis=1)[2]]).reshape(
                1, 3)
        elif args.target == 'clamp':  # around z-axis
            Re, te = sampling.samples_axis_rotation(ply['pts'].T, PHI_r, [0, 0, DELTA_r], snap=True)

            # apply GT rotation and adapt z-translation accordingly
            Re = np.array([re @ Rgt for re in Re]).reshape(-1, 3, 3)
            te = np.array([[0, 0, -(R @ ply['pts'].T).min(axis=1)[2]] for R in Re])
            te += np.array([Tgt[0, 3], Tgt[1, 3], Tgt[2, 3] + (Tgt[:3, :3] @ ply['pts'].T).min(axis=1)[2]]).reshape(
                1, 3)
        else:
            raise ValueError("'target' must be one of 'bowl', 'marker' or 'clamp'.")
        offsets = list(range(-PHI_r, PHI_r+1, DELTA_r))
    elif args.mode == 'translation':
        if args.target in ['bowl', 'clamp']:  # along z-axis
            Re, te = sampling.samples_axis_translation([[tgt[0], tgt[0]],
                                                        [tgt[1], tgt[1]],
                                                        [tgt[2] - PHI_t, tgt[2] + PHI_t]],
                                                       [1, 1, DELTA_t], R=Rgt)
        elif args.target == 'marker':  # along y-axis (for target 'marker')
            Re, te = sampling.samples_axis_translation([[tgt[0], tgt[0]],
                                                        [tgt[1] - PHI_t, tgt[1] + PHI_t],
                                                        [tgt[2], tgt[2]]],
                                                       [1, DELTA_t, 1], R=Rgt)
        else:
            raise ValueError("'target' must be one of 'bowl', 'marker' or 'clamp'.")
        offsets = list(range(-PHI_t, PHI_t+1, DELTA_t))
    else:
        raise ValueError(f"'mode' must be one of 'rotation' or 'translation'.")

    # === EVALUATE POSE ESTIMATES
    print("Evaluating estimates...")
    errors, Rps, tps = evaluate_sequence(renderer, obj_id, Re, te, Rview, tview, K, ply['pts'],
                                         Rgt, tgt, Tsyms, frame_gt['depth'], Rs, ts)

    # === PLOT ERRORS TOGETHER WITH RENDERING
    print("Rendering estimates and matched plausible pose...")
    frames_Te = render_sequence(renderer, env_ids, env_Ts, obj_id, Re, te, Rview, tview, K)
    frames_Tp = render_sequence(renderer, env_ids, env_Ts, obj_id, Rps, tps, Rview, tview, K)
    frame, frame_p, offset = frames_Te[0], frames_Tp[0], offsets[0]

    def vis():
        # Te
        ax = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=2)
        plt.axis('off')
        plt.imshow(frame['rgb'])
        ax.set_title("Te")

        # GT
        ax = plt.subplot2grid((3, 3), (0, 2))
        plt.axis('off')
        plt.imshow(frame_gt['rgb'])
        ax.set_title("Tgt")

        # Tp
        ax = plt.subplot2grid((3, 3), (1, 2))
        plt.axis('off')
        plt.imshow(frame_p['rgb'])
        ax.set_title("Tp")

        def subplot(grid, pos, x, y, label, th, cur_x, gt=0):
            plt.subplot2grid(grid, pos)
            plt.plot(x, y[0], 'k-')
            plt.plot(x, y[1], 'g-')
            plt.plot(x, y[2], 'b-')
            plt.plot(x, y[0], 'k.')
            plt.plot(x, y[1], 'g.')
            plt.plot(x, y[2], 'b.')
            plt.ylabel(label)
            plt.hlines(th, np.min(x), np.max(x), 'r', linestyles='-')
            plt.vlines([gt, offset], 0, th, ['gray', 'r'], linestyles=['-', '--'])

        # pose-error functions
        subplot((3, 3), (2, 0), offsets, [errors[:, 0], errors[:, 4], errors[:, 7]], "mssd", 0.2, offset)
        subplot((3, 3), (2, 1), offsets, [errors[:, 1], errors[:, 5], errors[:, 8]], "mspd", 20, offset)
        subplot((3, 3), (2, 2), offsets, [errors[:, 2], errors[:, 6], errors[:, 9]], "vsd (tau=0.2)", 0.3, offset)

    print("Visualizing results...")
    for i, (frame, frame_p, offset) in enumerate(zip(frames_Te, frames_Tp, offsets)):
        drawnow(vis)
        plt.pause(args.t_per_frame)
