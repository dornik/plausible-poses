import numpy as np
import bop_toolkit_lib.pose_error as error
import klampt


def classify_sequence(target, envs, env_Ts, Rs, ts, obj_friction=0.1, tolerance=5.0):
    """
    For poses [Rs, ts] of [target] object, determine whether they are 1) floating, 2) intersecting, 3) feasible,
    4) stable and 5) plausible wrt the environment [envs].
    :param target: Target object in klampt.
    :param envs: Environment objects in klampt.
    :param env_Ts: Poses of the environment objects in the current scene.
    :param Rs: Set of rotations of the target object to be classified.
    :param ts: Set of translations of the target object to be classified.
    :param obj_friction: Contact friciton coefficient to be used.
    :param tolerance: Surface distance tolerance for computing the contact and intersecting points, i.e., epsilon.
    :return: 5-tuple of booleans [not floating, not intersecting, feasible, stable, plausible] per pose candidate [R, t].
    """

    EPS, K = tolerance, obj_friction
    checks = np.ones((Rs.shape[0], 5), dtype=np.bool)  # [not floating, not intersecting, feasible, stable, plausible]

    # go through all candidates [Rs, ts]...
    for i, (R, t) in enumerate(zip(Rs, ts)):
        if i % 100 == 0:
            print(f"   checking {i+1}/{len(Rs)}")

        # prepare target object
        target.setCurrentTransform(R.T.reshape(9).tolist(), t.reshape(3).tolist())

        # go through environment objects and compute contact points C, intersecting points I and surface distances d...
        depths = np.array([])  # penetration depth
        contact = np.array([]).reshape(0, 7)  # [contact point xyz, contact normal nxnynz, contact friction]
        for env, env_T in zip(envs, env_Ts):
            # prepare environment object
            env.setCurrentTransform(env_T[:3, :3].T.reshape(9).tolist(), env_T[:3, 3].reshape(3).tolist())

            # === contacts and intersecting points
            # points within EPS outside and any inside env
            contacts = env.contacts(target, padding1=EPS, padding2=0, maxContacts=0)  # covers EPS outside case
            depths = np.append(depths, contacts.depths)
            contact = np.vstack((contact,
                                np.hstack((np.array(contacts.points1).reshape(-1, 3),
                                           np.array(contacts.normals).reshape(-1, 3),
                                           np.ones((len(contacts.points1), 1)) * K))))
        # check conditions for target object wrt union of all contact points C wrt environment

        # === floating: at least one contact point
        not_floating = contact.shape[0] > 0
        if not not_floating:
            checks[i] = [False] * 5
            continue

        # === intersecting: no point further inside than eps
        depths = EPS - depths  # depths are wrt padding
        not_intersecting = depths.min() > -EPS
        if not not_intersecting:
            checks[i] = [True] + [False] * 4
            continue

        # === stable: find set of valid contact forces such that object is in equilibrium
        obj_com = t  # note: models are centered at CoM, thus CoM and origin are the same

        stable = klampt.comEquilibrium(contact, [0, 0, -9.81], obj_com) is not None
        if not stable:
            checks[i] = [True] * 3 + [False] * 2
            continue

    return checks


def find_closest(Re, te, Rs, ts, metric, pts, syms, r2):
    """
    Given a pose estimate [Re, te], finds the closest plausible pose in [Rs, ts] wrt [metric].
    :param Re: Rotation estimate.
    :param te: translation estimate.
    :param Rs: Plausible rotations.
    :param ts: Plausible translations.
    :param metric: Pose-error function to be used. Valid choices are [te, frobenius, add, adi, mssd].
    :param pts: Point cloud [xyz] of the target object.
    :param syms: Set of transformations representing the symmetric poses of the object. Used by [frobenius, adi, mssd].
    :param r2: Scaling parameter for the rotation term in the Frobenius-based metric.
    :return: Rotation, translation and distance wrt [metric] of the closest plausible pose.
    """

    if metric == 'te':
        dist = np.linalg.norm(ts.reshape(-1, 3) - te.reshape(-1, 3), axis=1, ord=2)
    elif metric == 'frobenius':
        # eval metric defined in Brégier et al., "Defining the Pose of any 3D Rigid Object and an Associated Distance"
        # dist = np.sqrt(np.linalg.norm(ts.reshape(-1, 3) - te.reshape(-1, 3), axis=1, ord=2)**2
        #                + r2*np.linalg.norm(Rs - Re, axis=(1, 2), ord='fro')**2)

        # symmetric version
        dists = []
        for sym in syms:
            # symmetric pose candidate
            Rsym = Re @ sym['R']
            tsym = (Re @ sym['t']).reshape(3, 1) + te.reshape(3, 1)

            dists.append(np.sqrt(np.linalg.norm(ts.reshape(-1, 3) - tsym.reshape(-1, 3), axis=1, ord=2) ** 2
                                 + r2 * np.linalg.norm(Rs - Rsym, axis=(1, 2), ord='fro') ** 2))
        dist = np.min(dists, axis=0)

    elif metric == 'add':
        # serial evaluation (takes twice as long)
        # dist = [error.add(R.reshape(3, 3), t.reshape(3, 1), Re, te, pts) for R, t in zip(Rs, ts.T)]

        # precompute relative rotation and relative translation of all plausible poses to speed-up computation
        R_ = (Re - Rs).reshape(-1, 3, 3)
        t_ = (te.reshape(1, 3) - ts.reshape(-1, 3))
        dist = [np.linalg.norm(R @ pts.T + t.reshape(3, 1), axis=0, ord=2).mean() for R, t in zip(R_, t_)]
    elif metric == 'adi':
        dist = [error.adi(R.reshape(3, 3), t.reshape(3, 1), Re, te, pts) for R, t in zip(Rs, ts)]
    elif metric == 'mssd':
        # serial evaluation (takes twice as long)
        # dist = [error.mssd(R.reshape(3, 3), t.reshape(3, 1), Re, te, pts, syms) for R, t in zip(Rs, ts)]

        dists = []
        for sym in syms:
            # symmetric pose candidate
            Rsym = Re @ sym['R']
            tsym = (Re @ sym['t']).reshape(3, 1) + te.reshape(3, 1)

            # precompute relative rotation and relative translation of all plausible poses to speed-up computation
            R_ = (Rsym - Rs).reshape(-1, 3, 3)
            t_ = (tsym.reshape(1, 3) - ts.reshape(-1, 3))
            dists.append([np.linalg.norm(R @ pts.T + t.reshape(3, 1), axis=0, ord=2).max() for R, t in zip(R_, t_)])
        dist = np.min(dists, axis=0)
    else:
        raise ValueError(f"Parameter 'metric' must be one of 'te', 'frobenius', 'add', 'adi' or 'mssd'.")

    closest = np.argmin(dist)
    return Rs[closest], ts[closest], np.min(dist)
