from collections import defaultdict, namedtuple
import numpy as np
import math

# TODO or should radius be dynamic? avg steplength in the trajectory?

class Pathfinder():
    def __init__(self, current_position, indexer):
        """
        :param: current_position = GridPoint (namedTuple)
        :param: indexer = required indexer object for grid computations
        :param: timesteps = how many steps to look into the past for
        :param: radius
        :param: alpha = coefficient for distance measure - should be shape (timesteps, 1)
        :param: beta = coefficient for scalar product - should be shape (timesteps, 1)
        :param: history
        :param: destination
        """

        self.current_position = current_position
        self.indexer = indexer
        self.timesteps = 3
        self.radius = 1.5
        self.alpha = [1, 2, 3] # the last entry lies furthest in the past
        self.beta = [1, 2, 3]
        self.history = get_history(current_position, indexer) # the last entry lies furthest in the past
        self.destination = None

    def calculate_loss(self, t, direction, compare_x, compare_y, compare_direction):
        """
        Calculates loss between position of current pedestrian and the pedestrian to compare
        :param: t = current timestep
        :param: direction
        :param: compare_x
        :param: compare_y
        :param: compare_direction
        :returns: loss as float or None
        """
        if direction is not None and compare_direction is not None:
            scalar_product = np.dot(direction, compare_direction)
            # if the pedestrians walk in opposite directions, dismiss
            if 0 >= scalar_product >= -1:
                return None

            # L2 Norm distance
            distance = self.get_distance(self.history[t], [compare_x, compare_y])
            return self.alpha[t] * (distance/self.radius) + self.beta[t] * scalar_product
        else:
            return None

    def get_all_pedestrians_in_radius(self, t):
        """
        Collects all pedestrians that have a history inside the radius of the pedestrian at question at timepoint t
        :param: t = current timestep in the past
        :returns: peds_in_radius as Points
        """
        print(self.history)
        i = self.indexer.convert_x(self.history[t][0])
        j = self.indexer.convert_y(self.history[t][1])
        peds_in_radius = []
        grid = self.indexer.get_grid()
        # iterates current and neighbouring grid cells if existent
        i = np.clip(i, self.indexer.x_min+1, self.indexer.x_max-1)
        j = np.clip(j, self.indexer.y_min+1, self.indexer.y_max-1)
        for point in grid[i-1:i+1][j-1:j+1]:
            if point.next_x is not None and get_distance(self.history[t], [point.x, point.y]) < self.radius:
                peds_in_radius.append(point)
        return peds_in_radius


    # TODO
    # TODO find out how many steps there are in the future
    def get_path(self):
        """
        Finds the most suitable candidate according to a loss function from which the steps are copied
        :returns: list of all future x-y-coordinates of best candidate or None
        """
        print("current_position: ", self.current_position)
        candidates = []
        # for every timestep in the past collect all pedestrian points in a certain radius to current_position
        for t in range(self.timesteps):
            candidates_at_t = self.get_all_pedestrians_in_radius(t)
            print("peds_in_radius: ", candidates_at_t)
            candidates.append(candidates_at_t) # len = self.timesteps

        # if we have found candidates in every timestep in the past
        if len(candidates) == self.timesteps:
            candidates_final = []
            # test for every pedestrian in the fist timestep
            for i, ped in enumerate(candidates[0]):
                # if ped id also present in all other timesteps
                in_all_timesteps = []
                for t in range(1, self.timesteps):
                    point_same_ped_id, is_in_timestep = ped_is_in(ped, candidates[t])
                    in_all_timesteps.append(is_in_timestep)
                in_all_timesteps = np.all(in_all_timesteps)
                if in_all_timesteps:
                    # collect those as final candidates
                    candidates_final.append(ped)
            print("candidates_final: ", candidates_final)

            # for every final candidate calculate the loss over all timesteps
            for ped in candidates_final:
                loss = 0
                ped_history = get_history(ped, self.indexer)
                index_of_current_timestep = np.argwhere(ped_history == [ped.x, ped.y])
                for i in range(index_of_current_timestep, index_of_current_timestep+self.timesteps):
                    loss = loss + self.calculate_loss(i, get_history(ped.x, ped.y, ped.pre_x, ped.pre_y), ped_history[i,0], ped_history[i,1],
                                                     get_direction(ped_history[i,0], ped_history[i,1], ped_history[i+1,0], ped_history[i+1,1]))
                


        else:
            return None


def get_history(current_position, indexer):
    all_ped_frames = indexer.get_all_frames_by_ped_id(current_position.pedestrian)
    return indexer.get_ped_traj_xy(all_ped_frames)[::-1]
    # ich gehe davon aus, dass ich hier eine Historie kriege, die mit den aktuellen Koordinaten von current_position
    # aufhoert. D.h. umgedreht ist der erste Eintrag der aktuelle und alles danach sind Schritte in die Vergangenheit

def get_distance(coordinates, other_coordinates):
    return math.sqrt(math.pow(coordinates[0] - other_coordinates[0], 2) + math.pow(coordinates[1] - other_coordinates[1], 2))


def get_direction(x, y, pre_x, pre_y):
    """
    Calculates direction from last to current position in unit length

    :returns: direction or None
    """
    try:
        direction = np.array([x, y]) - np.array([pre_x, pre_y])
    except:
        return None
    return np.linalg.norm(direction)

def is_same_tuple(this, other):
    """
    Compares two namedTuples by their attribute frame and pedestrian id to ensure equality
    :returns: Boolean
    """
    return this[0] == other[0] and this[1] == other[1]

def is_same_ped(this, other):
    """
    Compares two namedTuples by their attribute pedestrian id
    :returns: Boolean
    """
    return this[1] == other[1]

def ped_is_in(point, list_of_points):
    """
    Checks whether the ped id of point is also present in list_of_points
    :returns: Boolean, entry
    """
    for entry in enumerate(list_of_points):
        if is_same_ped(point, entry) and not is_same_tuple(point, entry):
            return True, entry
    return False, None