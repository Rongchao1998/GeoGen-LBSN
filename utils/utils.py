from math import sin, cos, sqrt, atan2, radians, asin
import numpy as np
import torch


def resample_trajectory(x, length=200):
    """
    Resamples a trajectory to a new length.

    Parameters:
        x (np.ndarray): original trajectory, shape (N, 2)
        length (int): length of resampled trajectory

    Returns:
        np.ndarray: resampled trajectory, shape (length, 2)
    """
    len_x = len(x)
    time_steps = np.arange(length) * (len_x - 1) / (length - 1)
    x = x.T
    resampled_trajectory = np.zeros((2, length))
    for i in range(2):
        resampled_trajectory[i] = np.interp(time_steps, np.arange(len_x), x[i])
    return resampled_trajectory.T


def time_warping(x, length=200):
    """
    Resamples a trajectory to a new length.
    """
    len_x = len(x)
    time_steps = np.arange(length) * (len_x - 1) / (length - 1)
    x = x.T
    warped_trajectory = np.zeros((2, length))
    for i in range(2):
        warped_trajectory[i] = np.interp(time_steps, np.arange(len_x), x[i])
    return warped_trajectory.T


def gather(consts: torch.Tensor, t: torch.Tensor):
    """
    Gather consts for $t$ and reshape to feature map shape
    :param consts: (N, 1, 1)
    :param t: (N, H, W)
    :return: (N, H, W)
    """
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)


def q_xt_x0(x0, t, alpha_bar):
    # get mean and var of xt given x0
    mean = gather(alpha_bar, t) ** 0.5 * x0
    var = 1 - gather(alpha_bar, t)
    # sample xt from q(xt | x0)
    eps = torch.randn_like(x0).to(x0.device)
    xt = mean + (var ** 0.5) * eps
    return xt, eps  # also returns noise


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)
    return a


def p_xt(xt, noise, t, next_t, beta, eta=0):
    at = compute_alpha(beta.cuda(), t.long())
    at_next = compute_alpha(beta, next_t.long())
    x0_t = (xt - noise * (1 - at).sqrt()) / at.sqrt()
    c1 = (eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt())
    c2 = ((1 - at_next) - c1 ** 2).sqrt()
    eps = torch.randn(xt.shape, device=xt.device)
    xt_next = at_next.sqrt() * x0_t + c1 * eps + c2 * noise
    return xt_next


def divide_grids(boundary, grids_num):
    lati_min, lati_max = boundary['lati_min'], boundary['lati_max']
    long_min, long_max = boundary['long_min'], boundary['long_max']
    # Divide the latitude and longitude into grids_num intervals.
    lati_interval = (lati_max - lati_min) / grids_num
    long_interval = (long_max - long_min) / grids_num
    # Create arrays of latitude and longitude values.
    latgrids = np.arange(lati_min, lati_max, lati_interval)
    longrids = np.arange(long_min, long_max, long_interval)
    return latgrids, longrids


# calculte the distance between two points
def distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r * 1000


def get_discrete_trajectories(poi_coords, visit_probs, visit_dist, query_trajectories, interval=0.5):
    discrete_trajs = []

    for trajectory in query_trajectories:
        discrete_traj = []
        for checkin in trajectory:
            query_coord = np.array(checkin[0])
            query_time = int(checkin[1] / interval)

            # Compute distances from the query point to each POI
            distances = np.linalg.norm(poi_coords - query_coord, axis=1)

            # Incorporate visiting distribution at the specific timestamp and overall visiting probabilities
            adjusted_distances = distances / (visit_dist[:, query_time] * visit_probs)

            # Find the index of the smallest adjusted distance from all non-inf distances
            valid_distances = {}
            for i, dist in enumerate(adjusted_distances):
                if not np.isinf(dist) and not np.isnan(dist):
                    valid_distances[i] = dist

            if len(valid_distances) == 0:
                # select by distance
                nearest_poi_id = np.argmin(distances)
            else:
                nearest_poi_id = min(valid_distances, key=valid_distances.get)

            # Append the POI ID and timestamp to the discrete trajectory
            discrete_traj.append([nearest_poi_id, checkin[1]])

        # Append the completed discrete trajectory to the list of trajectories
        discrete_trajs.append(discrete_traj)

    return discrete_trajs