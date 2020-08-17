import mesh_to_sdf
import numpy as np
from scipy.spatial.transform.rotation import Rotation
import trimesh


def min_interaction_distance(mesh):
    """
    Computes the minimal interaction distance of a given [mesh].
    If the origin of the object lies within the mesh, the minimal interaction distance is the distance from origin to
    the closest point in the mesh. Otherwise, the minimal interaction distance is negative and determined as the
    distance from origin to the farthest front-facing point in the mesh. The origin is assumed to be the zero vector.
    :param mesh: trimesh object.
    :return: Minimal interaction distance of the given mesh (same scale and units as mesh).
    """
    origin = np.zeros((1, 3), dtype=np.float32)
    min_distance = float(trimesh.proximity.signed_distance(mesh, origin))  # equivalent to min distance if inside

    if min_distance < 0:  # origin outside mesh
        to_origin = origin - mesh.vertices
        origin_dist = np.linalg.norm(origin - mesh.vertices, axis=1)
        to_origin /= np.repeat(origin_dist.reshape(-1, 1), 3, axis=1)  # normalized vectors
        cos = np.einsum('ij,ij->i', mesh.vertex_normals.reshape(-1, 3), to_origin)  # cos > 0 --> front facing
        min_distance = -origin_dist[cos > 0].max()  # max distance among front facing
    return min_distance


def max_interaction_distance(mesh):
    """
    Computes the maximal interaction distance of a given [mesh], i.e., the distance from origin to the farthest point
    in the mesh. The origin is assumed to be the zero vector.
    :param mesh: trimesh object.
    :return: Maximal interaction distance of the given mesh (same scale and units as mesh).
    """
    return np.linalg.norm(mesh.vertices, axis=1).max()


def stable_poses(mesh, use_quasi_stable=False):
    """
    Computes the rest poses of a given [mesh] using 1) quasi-stable estimation or 2) our approach (default) based on
    classifying faces of the convex hull and the support polygon principle.
    :param mesh: trimesh object.
    :param use_quasi_stable: If true, use QSE. Otherwise, use our approach.
    :return: Set of rest poses [Rs, ts] for the given mesh.
    """
    if use_quasi_stable:  # Goldberg et al., "Part Pose Statistics: Estimators and Experiments"
        Ts, probs = mesh.compute_stable_poses(center_mass=mesh.center_mass, sigma=0.0, n_samples=1, threshold=0.0)
        Rs = Ts[:, :3, :3]
        ts = Ts[:, :3, 3]
    else:  # ours
        hull = mesh.convex_hull
        center_mass = mesh.center_mass.reshape(1, 3)
        normals = hull.face_normals

        # faces of convex hull that support the object
        hit_triangle, hit_ray = hull.ray.intersects_id(np.repeat(center_mass, normals.shape[0], axis=0), normals)
        hit_normal = normals[hit_triangle]
        cast_normal = normals[hit_ray]
        critical_ids = [j for j, (hit, cast) in enumerate(zip(hit_normal, cast_normal))
                        if np.dot(hit, cast) > np.cos(np.deg2rad(0.01))]
        print(f"{len(critical_ids)} stable rotations ({normals.shape[0]} faces in total)")

        # supporting faces to rest poses: rotate according to normal, snap (translate) to plane
        Rs, ts = [], []
        for fid in critical_ids:
            if hull.area_faces[fid] < 50:  # prune poses related to small hull faces (improbable to rest there)
                continue
            n = hull.face_normals[fid]
            R = trimesh.geometry.align_vectors(n.reshape(3), np.array([0, 0, -1]))[:3, :3]
            Rs.append(R)
            ts.append([0, 0, -(R @ mesh.vertices.T).min(axis=1)[2]])
        Rs = np.array(Rs)
        ts = np.array(ts)

    return Rs, ts


def samples_axis_rotation(pts, phi_r, delta_rs, snap=True):
    """
    Samples poses with rotational offset in [-phi_r, phi_r] at a rate of [delta_r] per given axis. Optionally, the
    translation to snap the given object (represented by [pts]) to the ground plane is computed.
    :param pts: numpy array, surface points of the object.
    :param phi_r: Rotational offset range.
    :param delta_rs: Per axis, the rotational offset sampling rate.
    :param snap: If true, determine translation such that object is snapped to ground plane.
    :return: Rotation and translation of sampled poses.
    """
    Rs = []
    if delta_rs[0] > 0:
        offset = np.arange(-phi_r, phi_r+1, delta_rs[0])
        Rs += [Rotation.from_euler('x', roff, degrees=True).as_dcm() for roff in offset]
    if delta_rs[1] > 0:
        offset = np.arange(-phi_r, phi_r + 1, delta_rs[1])
        Rs += [Rotation.from_euler('y', roff, degrees=True).as_dcm() for roff in offset]
    if delta_rs[2] > 0:
        offset = np.arange(-phi_r, phi_r + 1, delta_rs[2])
        Rs += [Rotation.from_euler('z', roff, degrees=True).as_dcm() for roff in offset]

    Rs = np.array(Rs).reshape(-1, 3, 3)
    ts = np.array([[0, 0, -(R @ pts).min(axis=1)[2]] for R in Rs]) if snap else np.zeros((Rs.shape[0], 3))

    return Rs, ts


def samples_axis_translation(bounds, delta_ts, R=np.eye(4)):
    """
    Samples poses with translation offset in [bounds] at a rate of [delta_t] per given axis. For all translation
    samples, the rotation is given by [R].
    :param bounds: Per axis, the upper and lower translation offset bounds.
    :param delta_ts: Per axis, the translation offset sampling rate.
    :param R: The rotation matrix to use for all samples.
    :return: Rotation and translation of sampled poses.
    """
    tx, ty, tz = np.meshgrid(np.arange(bounds[0][0], bounds[0][1] + delta_ts[0], delta_ts[0]),
                             np.arange(bounds[1][0], bounds[1][1] + delta_ts[1], delta_ts[1]),
                             np.arange(bounds[2][0], bounds[2][1] + delta_ts[2], delta_ts[2]), indexing='ij')

    ts = np.hstack((tx.reshape(-1, 1), ty.reshape(-1, 1), tz.reshape(-1, 1)))
    Rs = np.tile(R, (ts.shape[0], 1, 1))

    return Rs, ts


def samples_isolated(mesh, phi_t, delta_t, phi_r, delta_r):
    """
    Sample poses for an isolated object [mesh] within [phi_t, phi_r] at rate [delta_t, delta_r].
    :param mesh: Used to compute the stable rest poses of the object.
    :param phi_t: Sampling range for in-plane translations.
    :param delta_t: Sampling rate for in-plane translations.
    :param phi_r: Sampling range for in-plane rotation.
    :param delta_r: Sampling rate for in-plane rotation.
    :return: Pose samples [Rs, ts] for [mesh].
    """
    # == get stable pose candidates
    Rs_xy, ts_z = stable_poses(mesh)
    num_stable = Rs_xy.shape[0]

    # == sample in-plane rotation and in-plane translation (within phi_t from Tgt)
    Rs_z = np.array([Rotation.from_euler('z', rz, degrees=True).as_dcm()
                     for rz in np.arange(-phi_r, phi_r+delta_r, delta_r)]).reshape(-1, 3, 3)  # [-pi, pi] range
    num_inplane_r = Rs_z.shape[0]

    toff = np.arange(-phi_t, phi_t + delta_t, delta_t)  # [-phi_t, phi_t] range
    tx, ty = np.meshgrid(toff, toff, indexing='ij')  # [-phi_t, phi_t]x[-phi_t, phi_t] grid
    ts_xy = np.hstack((tx.reshape(-1, 1), ty.reshape(-1, 1), np.zeros_like(tx).reshape(-1, 1)))
    ts_xy = ts_xy[np.linalg.norm(ts_xy, axis=1) <= phi_t]  # disk with radius phi_t
    num_inplane_t = ts_xy.shape[0]

    # == combine: for every in-plane sample, try every stable candidate
    # combine in-plane: every rotation at every translation
    Rs_z = np.tile(Rs_z, (num_inplane_t, 1, 1))
    ts_xy = np.repeat(ts_xy, num_inplane_r, axis=0)
    num_inplane = num_inplane_r * num_inplane_t

    # combine in-plane rotations with stable xy rotations
    Rs_xy = np.tile(Rs_xy, (num_inplane, 1, 1))
    Rs_z = np.repeat(Rs_z, num_stable, axis=0)
    Rs_xyz = np.array([Rz @ Rxy for Rxy, Rz in zip(Rs_xy, Rs_z)]).reshape(-1, 3, 3)

    # combine in-plane translations with stable z translations
    ts_xy = np.repeat(ts_xy, num_stable, axis=0)
    ts_z = np.tile(ts_z, (num_inplane, 1))
    ts_xyz = np.hstack((ts_xy[:, :2], ts_z[:, 2].reshape(-1, 1)))

    return Rs_xyz, ts_xyz


def samples_interacting(mesh, env, Tgt, phi_t, delta_t, phi_r, delta_r, epsilon):
    """
    Sample poses for an object [mesh] interacting with scene [env] within [phi_t, phi_r] at rate [delta_t, delta_r]
    around the target's ground-truth pose [Tgt].
    :param mesh: Target object.
    :param env: List of scene objects.
    :param Tgt: Ground-truth pose of target.
    :param phi_t: Sampling range for in-plane translations.
    :param delta_t: Sampling rate for in-plane translations.
    :param phi_r: Sampling range for in-plane rotation.
    :param delta_r: Sampling rate for in-plane rotation.
    :param epsilon: Intersection threshold (used for pruning).
    :return: Pose samples [Rs, ts] for target [mesh] within scene [env].
    """
    # == translation samples
    toff = np.arange(-phi_t, phi_t + delta_t, delta_t).astype(np.float32)  # [-phi_t, phi_t] range
    tx, ty, tz = np.meshgrid(toff, toff, toff, indexing='ij')  # [-phi_t, phi_t]^3 grid
    ts = np.hstack((tx.reshape(-1, 1), ty.reshape(-1, 1), tz.reshape(-1, 1)))
    print(f"initial ts: {ts.shape[0]}")

    # prune translation samples outside sphere with radius phi_t
    ts = ts[np.linalg.norm(ts, axis=1) <= phi_t]
    print(f"ts within phi_t: {ts.shape[0]}")
    ts += Tgt[:3, 3]  # shift from origin to GT

    # prune translation samples outside interaction range [d_int^min, d_int^max]
    env_sdfs = [mesh_to_sdf.get_surface_point_cloud(other_mesh) for other_mesh in env]
    min_id = min_interaction_distance(mesh)
    max_id = max_interaction_distance(mesh)

    floating = np.ones((ts.shape[0]), dtype=np.bool)
    intersecting = np.zeros((ts.shape[0]), dtype=np.bool)
    for env_sdf in env_sdfs:
        distances = env_sdf.get_sdf_in_batches(ts)
        floating = np.logical_and(floating, max_id < distances)  # remove floating samples
        intersecting = np.logical_or(intersecting, distances < min_id)  # remove intersecting samples
    valid = np.logical_and(~floating, ~intersecting)
    print(f"ts within interaction distance: {valid.sum()}")

    ts = ts[valid]
    num_t = ts.shape[0]

    # == rotation samples
    roff = np.arange(-phi_r, phi_r + delta_r, delta_r).astype(np.float32)  # [-phi_r, phi_r] range
    rx, ry, rz = np.meshgrid(roff, roff, roff, indexing='ij')  # [-phi_r, phi_r]^3 grid
    Rs = np.hstack((rx.reshape(-1, 1), ry.reshape(-1, 1), rz.reshape(-1, 1)))  # Euler angles
    Rs = np.array([Rotation.from_euler('xyz', [Rx, Ry, Rz], degrees=True).as_dcm()
                   for Rx, Ry, Rz in Rs]).reshape(-1, 3, 3)  # to rotation matrices
    Rs = np.array([R @ Tgt[:3, :3] for R in Rs]).reshape(-1, 3, 3)  # rotate from origin to GT

    num_r = Rs.shape[0]
    print(f"initial Rs: {num_r}")

    # compute minimal point of [mesh] under each rotation sample
    pts = [(R @ mesh.vertices.T) for R in Rs]
    pts_min = np.min(pts, axis=2)
    ts_min = np.array([pt[:, np.argwhere(pt[2, :] == pt_min[2])].reshape(1, 3)
                       for pt, pt_min in zip(pts, pts_min)]).reshape(-1, 3)

    # prune if minimal point is below ground plane under sample pose (in [Rs, ts])
    ts_min_test = np.tile(ts_min, (num_t, 1)) + np.repeat(ts, ts_min.shape[0], axis=0)
    valid = ts_min_test[:, 2] >= -epsilon

    # prune samples where the minimal point of [mesh] would intersect the scene objects
    for env_sdf in env_sdfs:
        distances = env_sdf.get_sdf_in_batches(ts_min_test[valid])
        valid[valid] = np.logical_and(valid[valid], distances >= -epsilon)  # positive is outside
    valid = valid.reshape(ts_min.shape[0], num_t)
    print(f"valid samples: {valid.sum()} (of {num_r * num_t} candidates)")

    # == combine: every rotation at every translation
    Rs = np.tile(Rs, (num_t, 1, 1))[valid.T.reshape(-1)]
    ts = np.repeat(ts, num_r, axis=0)[valid.reshape(-1)]

    return Rs, ts


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Physically Plausible Poses')
    parser.add_argument('--target', type=str, default='bowl', choices=['bowl', 'marker', 'clamp'],
                        help='Target object to sample physically plausible poses for. Selects the scene accordingly.')
    args = parser.parse_args()

    # === INIT SCENE
    print("Initializing scene and estimates...")
    if args.target == 'bowl':
        obj_id = 13
        ISOLATED = True

        PHI_t, PHI_r, DELTA_t, DELTA_r = 50, 180, 5, 15
    else:
        if args.target == 'marker':
            obj_id = 18
        elif args.target == 'clamp':
            obj_id = 19
        else:
            raise ValueError("'target' must be one of 'bowl', 'marker' or 'clamp'.")

        ISOLATED = False
        PHI_t, PHI_r, DELTA_t, DELTA_r = 20, 20, 4, 4
    EPS, FRICTION = 2, 0.1

    import sys
    sys.path.append(".")
    sys.path.append("./src")
    import src.evaluation as eval
    env_meshes, env, env_ids, env_Ts, mesh, cloud, ply, Tgt, Tsyms, renderer, K, Rview, tview = eval.init(args.target)
    ground_volume = env[0]
    Rgt, tgt = Tgt[:3, :3], Tgt[:3, 3]

    # === COMPUTE PLAUSIBLE POSES
    # sample poses
    if ISOLATED:
        Rs, ts = samples_isolated(mesh, PHI_t, DELTA_t, PHI_r, DELTA_r)
    else:
        Rs, ts = samples_interacting(mesh, env_meshes, Tgt, PHI_t, DELTA_t, PHI_r, DELTA_r, EPS)

    # classify samples
    import src.plausibility as plausibility
    checks = plausibility.classify_sequence(cloud, env, env_Ts, Rs, ts, FRICTION, EPS)
    Rs, ts = Rs[checks[:, -1]], ts[checks[:, -1]]  # only plausible samples

    # store physically plausible samples
    import os
    import pickle
    samples = {'Rs': Rs, 'ts': ts}
    with open(os.path.join(eval.DATA_PATH, f"samples_{args.target}_new.pkl"), 'wb') as file:
        pickle.dump(samples, file)
