import collision
import pathfinder

import dill
import math
import json
import argparse
from collections import defaultdict, namedtuple
import numpy as np
import copy
from statistics import mean

TrackRow = namedtuple('Row', ['frame', 'pedestrian', 'x', 'y', 'prediction_number', 'scene_id'])
TrackRow.__new__.__defaults__ = (None, None, None, None, None, None)
SceneRow = namedtuple('Row', ['scene', 'pedestrian', 'start', 'end', 'fps', 'tag'])
GridPoint = namedtuple('Point', ['frame', 'pedestrian', 'x', 'y', 'pre_x', 'pre_y',
                                 'next_x', 'next_y', 'prediction_number', 'scene_id'])
GridPoint.__new__.__defaults__ = (None, None, None, None, None, None, None, None, None, None)
distance_threshold = 0.2


class Indexer(object):
    """Read trajnet files and put tracks into grid

    :param grid_dim: defines grid dimension  n x n
    """

    def __init__(self, input_file, grid_dim=100):

        self.grid_dim = grid_dim
        self.grid = [[[] for x in range(self.grid_dim)] for y in range(self.grid_dim)]

        self.tracks = []
        self.scenes_by_frame = defaultdict(list)

        self.x_max = self.y_max = float('-inf')
        self.x_min = self.y_min = float('inf')

        self.read_file(input_file)

        self.y_dim = abs(self.y_max) + abs(self.y_min)
        self.x_dim = abs(self.x_max) + abs(self.x_min)

        self.fill_grid()

    # Reads file and extract environment dimension
    # creates tracks and scene lists
    def read_file(self, input_file):
        self.scene_frames_start_end = {}  # scene_id:[frame_start, frame_end]
        with open(input_file, 'r') as f:
            for line in f:
                line = json.loads(line)

                track = line.get('track')
                if track is not None:
                    row = TrackRow(track['f'], track['p'], track['x'], track['y'], \
                                   track.get('prediction_number'), track.get('scene_id'))

                    self.x_max = row.x if row.x > self.x_max else self.x_max
                    self.x_min = row.x if row.x < self.x_min else self.x_min
                    self.y_max = row.y if row.y > self.y_max else self.y_max
                    self.y_min = row.y if row.y < self.y_min else self.y_min

                    self.tracks.append(row)
                    continue

                scene = line.get('scene')
                if scene is not None:
                    self.scene_frames_start_end[scene['id']] = [scene['s'], scene['e']]
                    row = SceneRow(scene['id'], scene['p'], scene['s'], scene['e'], \
                                   scene['fps'], scene['tag'])
                    for i in range(row.start, row.end + 1):
                        self.scenes_by_frame[i].append(row.scene)

        self.x_max = math.ceil(self.x_max)
        self.x_min = math.floor(self.x_min)
        self.y_max = math.ceil(self.y_max)
        self.y_min = math.floor(self.y_min)

    # fills grid with the tracks
    def fill_grid(self):
        for i in range(len(self.tracks)):
            frame = self.tracks[i].frame
            pedestrian = self.tracks[i].pedestrian
            x = self.tracks[i].x
            y = self.tracks[i].y
            prediction_number = self.tracks[i].prediction_number
            scene_id = self.tracks[i].scene_id
            pre_x = pre_y = next_x = next_y = None

            # If a previous point on the track of this point exists, save the coordinates
            if (self.tracks[i - 1].frame + 1) == frame \
                    and self.tracks[i - 1].pedestrian == pedestrian:
                pre_x = self.tracks[i - 1].x
                pre_y = self.tracks[i - 1].y

            # If a next point on the track of this point exists, save the coordinates
            if i < len(self.tracks) - 1:
                if (self.tracks[i + 1].frame - 1) == frame \
                        and self.tracks[i + 1].pedestrian == pedestrian:
                    next_x = self.tracks[i + 1].x
                    next_y = self.tracks[i + 1].y

            point = GridPoint(frame, pedestrian, x, y, pre_x, pre_y, next_x, next_y, prediction_number, scene_id)
            self.grid[self.convert_x(point.x)][self.convert_y(point.y)].append(point)

    # returns x-coordinate inside the grid
    def convert_x(self, x):
        tmp_x = x + abs(self.x_min)
        return math.floor((tmp_x / self.x_dim) * self.grid_dim)

    # returns y-coordinate inside the grid
    def convert_y(self, y):
        tmp_y = y + abs(self.y_min)
        return math.floor((tmp_y / self.y_dim) * self.grid_dim)

    # returns all scene ids from a certain frame
    def get_scenes_at_frame(self, frame):
        return self.scenes_by_frame[frame]

    # returns whole grid
    def get_grid(self):
        return self.grid

    # returns all frames from a grid panel which includes the coordinates x,y
    def get_frames_at_xy(self, x, y):
        return self.grid[self.convert_x(x)][self.convert_y(y)]

    def get_all_frames_by_ped_id(self, ped_id):
        all_ped_frames = []
        for i in range(len(self.tracks)):
            if self.tracks[i].pedestrian == ped_id:
                all_ped_frames.append(self.tracks[i])
        return all_ped_frames

    def get_ped_traj(self, all_ped_frames):
        xy = []
        ped_traj = []
        for i, p in enumerate(all_ped_frames):
            if [p.x, p.y] not in xy:
                xy.append([p.x, p.y])
                ped_traj.append(p)
                # print(i, p)
        return ped_traj

    def get_ped_traj_xy(self, all_ped_frames):
        xy = []
        ped_traj = []
        for i, p in enumerate(all_ped_frames):
            if [p.x, p.y] not in xy:
                xy.append([p.x, p.y])
                ped_traj.append(p)
                # print(i, p)
        return xy

    def get_all_frames_by_scene_id(self, scene_id):
        all_frames_by_scene_id = []
        unique_pedestrians = []
        for i in range(len(self.tracks)):
            if self.tracks[i].frame >= self.scene_frames_start_end[scene_id][0] and self.tracks[i].frame <= \
                    self.scene_frames_start_end[scene_id][1]:
                all_frames_by_scene_id.append(self.tracks[i])
                unique_pedestrians.append(self.tracks[i].pedestrian)
        unique_pedestrians = list(set(unique_pedestrians))
        return all_frames_by_scene_id, unique_pedestrians

    def get_scene_prediction_datastracture(self, all_frames_by_scene_id):
        scene_prediction_datastracture = {}
        # iterate over all frames to group pedestrians together
        for f in all_frames_by_scene_id:
            if f.pedestrian not in scene_prediction_datastracture.keys():
                scene_prediction_datastracture[f.pedestrian] = {}
                scene_prediction_datastracture[f.pedestrian][f.frame] = f
            else:
                scene_prediction_datastracture[f.pedestrian][f.frame] = f
        # iterate over pedestrians
        for p in scene_prediction_datastracture.keys():
            all_ped_frames = self.get_all_frames_by_ped_id(p)
            ped_traj = self.get_ped_traj(all_ped_frames)
            all_per_frames_in_scene = list(scene_prediction_datastracture[p].keys())

            common_frame_id = list(np.intersect1d([ff.frame for ff in all_ped_frames], all_per_frames_in_scene))
            common_frame_id = common_frame_id[0]
            for f in all_frames_by_scene_id:  # find common frame in the scene data
                if f.frame == common_frame_id and p == f.pedestrian:
                    x = f.x  # scene data
                    y = f.y  # scene data
                    fr = f.frame  # scene data
            for i, f in enumerate(ped_traj):  # find common frame in the ped traj data
                if f.x == x and f.y == y:
                    index_frame = fr - i
            for i in range(len(ped_traj)):
                ped_traj[i] = copy.deepcopy(ped_traj[i]._replace(frame=index_frame))
                index_frame += 1
            for i, f in enumerate(ped_traj):
                if f.frame in list(scene_prediction_datastracture[p].keys()):
                    if f.x == scene_prediction_datastracture[p][f.frame].x and f.y == scene_prediction_datastracture[p][
                        f.frame].y:
                        pass
                    else:
                        print('ERROR: X, Y values for pedestrian don\'t match X/Y values for frame')
                else:
                    scene_prediction_datastracture[p][f.frame] = f
        return scene_prediction_datastracture

    def predict_mean_movement(self, line_ped_x, line_ped_y, line_ped_x_mean, line_ped_y_mean,
                                frame_id, ped_id,scene_id):
        """
        Computes the average movement from the previous steps and predicts one more step based on that
        returns: newly computed lines (with appended step if no collision)
        """
        mvmt_x = []
        mvmt_y = []

        # compute average movement for each step/frame
        for j in range(len(line_ped_x) - 1):
            mvmt_x.append(line_ped_x[j + 1] - line_ped_x[j])
            mvmt_y.append(line_ped_y[j + 1] - line_ped_y[j])

        # store last step for grid update
        last_step_x = line_ped_x[-1]
        last_step_y = line_ped_y[-1]

        # next step is the last step + mean of all movement-values
        next_step_x = last_step_x + mean(mvmt_x)
        next_step_y = last_step_y + mean(mvmt_y)

        # extend line to be plotted with this
        line_ped_x.append(next_step_x)
        line_ped_y.append(next_step_y)
        # additional line for plotting only the extension by mean
        line_ped_x_mean.append(next_step_x)
        line_ped_y_mean.append(next_step_y)
        # extend the grid with the newest step
        new_track = TrackRow(frame_id, ped_id, next_step_x, next_step_y, None, scene_id)
        self.extend_grid(track=new_track, pre_x=last_step_x, pre_y=last_step_y)

        return line_ped_x, line_ped_y, line_ped_x_mean, line_ped_y_mean

    def extend_grid(self, track, pre_x, pre_y):
        """
        Extend the grid by the TrackRow track
        """
        next_x = next_y = None
        point = GridPoint(track.frame, track.pedestrian, track.x, track.y, pre_x, pre_y, next_x, next_y,
                          track.prediction_number, track.scene_id)
        self.grid[self.convert_x(point.x)][self.convert_y(point.y)].append(point)


def check_distance(pedestrian):
    min_distance = 1000000
    new_pedestrian = copy.deepcopy(pedestrian)
    # for each frame
    for frame in range(len(pedestrian[0][0])):
        # for each pedestrian
        for ped in range(len(pedestrian)):
            # for other pedestrians
            for ped_compare in range(len(pedestrian)):
                if ped != ped_compare:
                    # check for distance
                    distance = calculate_distance(new_pedestrian[ped][0][frame], new_pedestrian[ped][1][frame],
                                                  new_pedestrian[ped_compare][0][frame], new_pedestrian[ped_compare][1][frame])
                    if distance < min_distance:
                        min_distance = distance
                    if distance < distance_threshold:
                        # movement vector for new_pedestrian
                        vector_ped_x = new_pedestrian[ped][0][frame] - new_pedestrian[ped][0][frame - 1]
                        vector_ped_y = new_pedestrian[ped][1][frame] - new_pedestrian[ped][1][frame - 1]

                        # collision avoidance vector
                        vector_ped_compare_x = new_pedestrian[ped][0][frame] - new_pedestrian[ped_compare][0][frame]
                        vector_ped_compare_y = new_pedestrian[ped][1][frame] - new_pedestrian[ped_compare][1][frame]

                        # compute weight for avoidance maneuver
                        weight = 1 - ((distance - 0.0) / (distance_threshold - 0.0))
                        #print("Weight:", weight)
                        #print("Distance", distance)

                        # norm
                        vector_ped_compare_x /= distance * 2
                        vector_ped_compare_y /= distance * 2

                        # new position = position before + movement vector + collision avoidance vector
                        new_pedestrian[ped][0][frame] = pedestrian[ped][0][frame - 1] + vector_ped_x + (
                                    vector_ped_compare_x * weight)
                        new_pedestrian[ped][1][frame] = pedestrian[ped][1][frame - 1] + vector_ped_y + (
                                    vector_ped_compare_y * weight)

                        # print('frame #' + str(frame))
                        # print('ped #' + str(ped))
                        # print('ped_compare #' + str(ped_compare))
                        # print(str(pedestrian[ped][0][frame]) + '; ' + str(pedestrian[ped][1][frame]))
                        # print(str(pedestrian[ped_compare][0][frame]) + '; ' + str(pedestrian[ped_compare][1][frame]))
                        # print(str(distance))

    print("Min distance:" , min_distance)
    return new_pedestrian


def calculate_distance(point_1_x, point_1_y, point_2_x, point_2_y):
    return math.sqrt(math.pow(point_2_x - point_1_x, 2) + math.pow(point_2_y - point_1_y, 2))


# def norm_vec(vec_x, vec_y):
#    return math.sqrt(math.pow(vec_x,2) + math.pow(vec_y,2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ndjson', help='trajnet dataset file')
    parser.add_argument('--load', default=None, help='deserialize dataset file')
    parser.add_argument('--n', type=int, default=100, help='grid dimension n x n')
    parser.add_argument('--scene_id', type=int, default=2396, help="scene you want to look at")
    args = parser.parse_args()
    print(args.ndjson)

    # deserialization
    if args.load is not None:
        indexer = Indexer(args.ndjson, args.n)
        indexer_pred = dill.load(open(str(args.load.split("/")[-1]).replace(".ndjson", "") + ".p", "rb"))
    else:
        indexer = Indexer(args.ndjson, args.n)
    grid = indexer.get_grid()
    '''print([indexer.x_max, indexer.x_min, indexer.y_max, indexer.y_min])
    for p in indexer.get_frames_at_xy(indexer.tracks[5].x, indexer.tracks[5].y):
        print(p)'''

    scene_id = args.scene_id
    all_frames_by_scene_id, unique_pedestrians = indexer.get_all_frames_by_scene_id(scene_id)

    scene_prediction_datastracture = indexer.get_scene_prediction_datastracture(all_frames_by_scene_id)

    # scene_ped_curves = {}
    # for ped in unique_pedestrians:
    #     all_ped_frames = indexer.get_all_frames_by_ped_id(ped)
    #     ped_traj = indexer.get_ped_traj(all_ped_frames, ped)
    #     scene_ped_curves[ped] = ped_traj
    # for i in scene_ped_curves.keys():
    #     ax = plt.plot([p.x for p in scene_ped_curves[i]], [p.y for p in scene_ped_curves[i]], '*')

    # matplotlib.use('MacOSX')
    import matplotlib.pyplot as plt

    # for ped in range(10):
    #     all_ped_frames = indexer.get_all_frames_by_ped_id(ped)
    #     ped_traj = indexer.get_ped_traj(all_ped_frames, ped)
    #     ax = plt.plot([p.x for p in all_ped_frames], [p.y for p in all_ped_frames], '*')
    print(scene_prediction_datastracture.keys())
    pedestrians = []
    collisions = []
    input_line = []
    for p in scene_prediction_datastracture.keys():

        line_ped_x = []
        line_ped_y = []
        line_ped_x_in = []
        line_ped_y_in = []
        line_ped_x_mean = []
        line_ped_y_mean = []
        collision_x = []
        collision_y = []
        # iterate over all frames that the scene has
        for i in range(indexer.scene_frames_start_end[scene_id][0], indexer.scene_frames_start_end[scene_id][1] + 1):
            try:
                if i < 9 + indexer.scene_frames_start_end[scene_id][0]:
                    # append the values found in the database
                    print("Frame: ", scene_prediction_datastracture[p][i].frame)
                    print("X Value: ", scene_prediction_datastracture[p][i].x)
                    line_ped_x.append(scene_prediction_datastracture[p][i].x)
                    line_ped_y.append(scene_prediction_datastracture[p][i].y)
                    line_ped_x_in.append(scene_prediction_datastracture[p][i].x)
                    line_ped_y_in.append(scene_prediction_datastracture[p][i].y)
                else:
                    line_ped_x, line_ped_y, line_ped_x_mean, line_ped_y_mean, \
                    collision_x, collision_y = indexer.predict_mean_movement(line_ped_x, line_ped_y, line_ped_x_mean,
                                                                             line_ped_y_mean, ped_id=p,
                                                                             scene_id=scene_id,
                                                                             frame_id=i, collision_x=collision_x,
                                                                             collision_y=collision_y)
            except:  # no database entries found

                # TODO added pathfinder
                found_steps = len(line_ped_x)
                # from the last valid frame compute the path
                # TODO indexer_for_prediction (can be the same as indexer from which the current position comes from)
                # TODO can be loaded using serialization with dill
                indexer_for_prediction = indexer
                pathfinder_object = pathfinder.Pathfinder(scene_prediction_datastracture[p][i - 1], indexer, indexer_for_prediction)
                path_prediction = pathfinder_object.get_path(found_steps)
                # TODO this has all possible future coordinates of winner pedestrian, need to filter the needed ones
                # TODO and handle if we stil need more
                if path_prediction:
                    print("Pathfinder path: ", path_prediction)
                else:
                    # pathfinder function could not find a suitable path to copy, do something else
                    print("pathfinder function could not find a suitable path to copy, do something else")
                    pass



                # try to predict the movement by taking mean movement of pedestrian before
                line_ped_x, line_ped_y, line_ped_x_mean, line_ped_y_mean = indexer.predict_mean_movement(line_ped_x,
                                                                    line_ped_y, line_ped_x_mean, line_ped_y_mean,
                                                                    ped_id=p, scene_id=scene_id, frame_id=i)

        pedestrians.append([line_ped_x, line_ped_y])
        input_line.append([line_ped_x_in, line_ped_y_in])

    # compute the avoidance paths
    old_pedestrians = copy.deepcopy(pedestrians)
    new_pedestrians = check_distance(pedestrians)
    pedestrians == new_pedestrians
    i = 2
    while new_pedestrians != pedestrians or i >= 5:
        pedestrians = copy.deepcopy(new_pedestrians)
        new_pedestrians = check_distance(pedestrians)
        print(f"Done >> Iteration {i}")
        i += 1

    #assert old_pedestrians != new_pedestrians
    # plot the pedestrians
    for i in range(len(new_pedestrians)):
        #if new_pedestrians[i] != pedestrians[i]:
        #ax = plt.plot(input_line[i][0], input_line[i][1], linewidth=3)
        ax = plt.plot(new_pedestrians[i][0], new_pedestrians[i][1])
        ax = plt.plot(old_pedestrians[i][0], old_pedestrians[i][1], linestyle="dashed")

    # ax[0].axes.set_aspect('equal')
    plt.title([args.ndjson.split('/')[-1], "scene_id", scene_id])
    plt.savefig('collision_avoidance.png')
    plt.show()

    print('DONE')
    # collect scene frames
    # collect ped trajectories
    # assign start in ped trajectories
    # extend scene trajectories


if __name__ == '__main__':
    main()
