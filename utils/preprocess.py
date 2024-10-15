import numpy as np


def interpolate(trajs, interval=0.5):
    inter_trajs = []
    for traj in trajs:
        inter_traj = [-1] * int(24 * 7 / interval)
        for i in range(0, len(traj)):
            # find the index of the time offset
            index = int((traj[i][1] + interval) / interval)
            if index < len(inter_traj) and inter_traj[index] == -1:
                inter_traj[index] = traj[i][0]
        for i in range(1, len(inter_traj)):
            if inter_traj[i] == -1:
                inter_traj[i] = inter_traj[i-1]

        # replace all -1 with the last poi
        for i in range(0, len(inter_traj)):
            if inter_traj[i] == -1:
                inter_traj[i] = inter_traj[len(inter_traj) - 1]
        inter_trajs.append(inter_traj)

    inter_trajs = np.array(inter_trajs)
    return inter_trajs


# transform the interpolated data into the same format as the real data,
def transform_poi_trajectories(trajectories, interval=0.5):
    transformed_trajectories = []

    for trajectory in trajectories:
        transformed_trajectory = []
        previous_poi = None

        for index, poi_id in enumerate(trajectory):
            if poi_id != previous_poi:
                # Calculate time offset in hours
                time_offset = index * interval  # since each index represents a 30-minute interval

                # Append the new check-in point
                transformed_trajectory.append([poi_id, time_offset])

                # Update the previous POI id
                previous_poi = poi_id
        # remove the first check-in point
        transformed_trajectory = transformed_trajectory[1:]

        # Store the transformed trajectory
        if len(transformed_trajectory) > 0:
            transformed_trajectories.append(transformed_trajectory)

    return transformed_trajectories


def read_real_trajectories(file_path):
    trajectories = []
    with open(file_path, 'r') as f:
        trajectory = []
        for line in f:
            if line == '\n':
                for i in range(1, len(trajectory)):
                    trajectory[i][1] += trajectory[i - 1][1]
                trajectories.append(trajectory)
                trajectory = []
            else:
                points = line.strip().split(',')
                ls = [int(points[4]), float(points[8])]
                trajectory.append(ls)

    return trajectories


def read_gps(file_path):
    gps_points = []
    with open(file_path, 'r') as f:
        for line in f:
            points = line.strip().split(' ')
            ls = [float(points[0]), float(points[1])]
            gps_points.append(ls)
    # normalize the latitude and longitude to [0, 1] respectively
    gps_points = np.array(gps_points)
    lat_min, lat_max = np.min(gps_points[:, 0]), np.max(gps_points[:, 0])
    lon_min, lon_max = np.min(gps_points[:, 1]), np.max(gps_points[:, 1])
    gps_points[:, 0] = (gps_points[:, 0] - lat_min) / (lat_max - lat_min)
    gps_points[:, 1] = (gps_points[:, 1] - lon_min) / (lon_max - lon_min)
    # transform all values to float
    gps_points = gps_points.astype(np.float32)
    return gps_points, lat_min, lat_max, lon_min, lon_max


def read_category(file_path):
    poi_cat = []
    with open(file_path, 'r') as f:
        for line in f:
            points = line.strip().split(' ')
            poi_cat.append(int(points[0]))
    return poi_cat


# get visiting distribution of each poi, return a ndarray of visiting distribution,
# each row is  of size seq_len, sum to 1, the return size is num_pois * seq_len
def get_pois_visiting_distribution(interpolated_trajs, num_pois, seq_len=336):
    visiting_distributions = np.zeros((num_pois, seq_len))
    for traj in interpolated_trajs:
        for i in range(1, len(traj)):
            # if the poi is different from the previous one, then add 1 to the visiting distribution
            if traj[i] != traj[i - 1]:
                visiting_distributions[traj[i - 1]][i] += 1
    # normalize the visiting distribution
    for i in range(num_pois):
        visiting_distributions[i] /= np.sum(visiting_distributions[i])
    return visiting_distributions


class POI_traj_Dataset():
    def __init__(self, traj_path, gps_path, cat_path):
        self.real_trajectories = read_real_trajectories(traj_path)
        self.inter_trajectories = interpolate(self.real_trajectories)

        # gps
        self.poi_gps, self.lat_min, self.lat_max, self.lon_min, self.lon_max = read_gps(gps_path)
        self.gps_trajectories = self.transform2GPS_traj(self.inter_trajectories, self.poi_gps)

        # category
        self.poi_cat = read_category(cat_path)
        self.cat_trajectories = self.transform2Cat_traj(self.inter_trajectories, self.poi_cat)

        # visiting distribution
        self.visiting_distributions = get_pois_visiting_distribution(self.inter_trajectories, len(self.poi_gps))
        self.vd_trajectories = self.transform2Vd_traj(self.inter_trajectories, self.visiting_distributions)

        self.masks = self.generate_mask(self.inter_trajectories)
        self.N_poi = len(self.poi_gps)

    def __len__(self):
        return len(self.real_trajectories)

    def __getitem__(self, idx):
        return self.real_trajectories[idx], self.gps_points, self.poi_cat

    def transform2GPS_traj(self, trajs, poi_gps):
        gps_trajs = []
        for traj in trajs:
            gps_traj = []
            for poi in traj:
                gps_traj.append(poi_gps[poi])
            gps_trajs.append(gps_traj)
        gps_trajs = np.array(gps_trajs)
        return gps_trajs

    def transform2Cat_traj(self, trajs, poi_cat):
        cat_trajs = []
        for traj in trajs:
            cat_traj = []
            for poi in traj:
                cat_traj.append(poi_cat[poi])
            cat_trajs.append(cat_traj)
        cat_trajs = np.array(cat_trajs)
        return cat_trajs

    def transform2Vd_traj(self, trajs, visiting_distributions):
        vd_trajs = []
        for traj in trajs:
            vd_traj = []
            for poi in traj:
                vd_traj.append(visiting_distributions[poi])
            vd_trajs.append(vd_traj)
        vd_trajs = np.array(vd_trajs)
        return vd_trajs

    def generate_mask(self, trajs):
        masks = []
        for traj in trajs:
            mask = []
            cur = 0
            for i in range(len(traj)):
                if i == 0 or traj[i] == traj[i - 1]:
                    mask.append(cur)
                else:
                    cur = 1 if cur == 0 else 0
                    mask.append(cur)
            masks.append(mask)
        masks = np.array(masks)
        return masks


if __name__ == '__main__':
    path = '../data/NYC/train_set.csv'
    gps_path = '../data/NYC/gps'
    cat_path = '../data/NYC/category'
    dataset = POI_traj_Dataset(path, gps_path, cat_path)
    print(len(dataset))
