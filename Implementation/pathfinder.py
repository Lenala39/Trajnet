from collections import defaultdict, namedtuple
import numpy as np
import math

# TODO or should radius be dynamic? avg steplength in the trajectory?
# TODO sometimes possible None values still need to be caught
# TODO don't iterate lists or grid cells if they are possibly empty

class Pathfinder():
    def __init__(self, current_position, indexer_current, indexer_db):
        """
        :param: current_position = GridPoint (namedTuple)
        :param: indexer_current = indexer object for grid computations, source of current_position
        :param: indexer_db = indexer object for grid computations, source of desired predictions (can also be == indexer_current)
        :param: timesteps = how many steps to look into the past for
        :param: radius
        :param: alpha = coefficient for distance measure - should be length timesteps
        :param: beta = coefficient for scalar product - should be length timesteps
        :param: history
        :param: destination
        """

        self.current_position = current_position
        self.indexer_current = indexer_current
        self.indexer_db = indexer_db
        self.timesteps = 3
        self.radius = 1.5
        self.alpha = list(range(1, self.timesteps+1)) # the last entry lies furthest in the past
        self.beta = list(range(1, self.timesteps+1))

        tmp = get_history(current_position, indexer_current)
        current_position_index = tmp.index([current_position.x, current_position.y])
        tmp = tmp[tmp.index([current_position.x, current_position.y]):] # check: does that really start WITH current coordinates?
        self.history = tmp # the last entry lies furthest in the past
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
            distance = get_distance(self.history[t], [compare_x, compare_y])
            # defined loss function:
            return self.alpha[t] * (distance/self.radius) + self.beta[t] * scalar_product
        else:
            return None

    def get_all_pedestrians_in_radius(self, t):
        """
        Collects all pedestrians that have a history inside the radius of the pedestrian at question at timepoint t
        :param: t = current timestep in the past
        :returns: peds_in_radius as Points
        """
        i = self.indexer_current.convert_x(self.history[t][0])
        j = self.indexer_current.convert_y(self.history[t][1])
        peds_in_radius = []
        grid = self.indexer_db.get_grid()
        # iterates current and, if existent, neighbouring grid cells
        i = np.clip(i, 1, self.indexer_db.grid_dim-1)
        j = np.clip(j, 1, self.indexer_db.grid_dim-1)

        for ii in range(i-1, i+1):
            for jj in range(j-1, j+1):
                for point in grid[ii][jj]:
                    if point.next_x is not None and get_distance(self.history[t], [point.x, point.y]) < self.radius:
                        peds_in_radius.append(point)
        return peds_in_radius


    # TODO find out how many steps there are in the future?
    def get_path(self, found_steps):
        """
        Finds the most suitable candidate according to a loss function from which the steps are copied
        :param found_steps: int of how many steps have been found already
        :returns: list of all future x-y-coordinates of best candidate or None
        """
        print("current_position: ", self.current_position)
        candidates = []
        # for every timestep in the past (increasing t) collect all pedestrian points in a certain radius to current_position
        for t in range(self.timesteps):
            candidates_at_t = self.get_all_pedestrians_in_radius(t)
            if not len(candidates_at_t) == 0:
                candidates.append(candidates_at_t) # len = self.timesteps

        # if we have found candidates in every timestep in the past
        if len(candidates) == self.timesteps:
            candidates_final = []
            # test for every pedestrian in the fist timestep
            for i, ped in enumerate(candidates[0]):
                # check if ped id also present in all other timesteps
                in_all_timesteps = []
                for t in range(1, self.timesteps):
                    #print(print('candidates[t]: ', candidates[t]))
                    is_in_timestep, point_same_ped_id = ped_is_in(ped, candidates[t])
                    in_all_timesteps.append(is_in_timestep)
                # if present at every single timestep
                in_all_timesteps = np.all(in_all_timesteps)
                if in_all_timesteps:
                    # collect those as final candidates
                    candidates_final.append(ped)
            print("candidates_final: ", candidates_final)

            losses_final = []
            # for every final candidate calculate the summed loss over all timesteps
            if candidates_final:
                for ped in candidates_final:
                    loss_sum = 0
                    ped_history = get_history(ped, self.indexer_db) # we need all timesteps

                    #index_of_current_timestep = np.argwhere(ped_history == list([ped.x, ped.y])) # find right starting point in history
                    index_of_current_timestep = ped_history.index(list([ped.x, ped.y]))
                    # sum up loss of candidate over every timestep
                    for i in range(index_of_current_timestep, index_of_current_timestep+self.timesteps):
                        loss = self.calculate_loss(i-index_of_current_timestep, get_direction(ped.x, ped.y, ped.pre_x, ped.pre_y), ped_history[i][0], ped_history[i][1],
                                                   get_direction(ped_history[i][0], ped_history[i][1], ped_history[i+1][0], ped_history[i+1][1]))
                        #print("loss: ", loss)
                        if loss is not None:
                            loss_sum = loss_sum + loss
                    losses_final.append(loss_sum)

                if losses_final:
                    # find candidate with smallest loss
                    print("losses_final: ", losses_final)
                    index_winner = np.argmin(np.array(losses_final))
                    if index_winner:
                        print("index_winner: ", index_winner)
                        winner = candidates_final[index_winner]
                        winner_history = get_history(winner, self.indexer_db)
                        print("winner_history: ", winner_history)

                        # return every future step of most suitable track starting with first prediction step
                        start_index = winner_history.index(list((winner.x, winner.y)))
                        #return winner_history[:np.argwhere(winner_history == [winner.x, winner.y])][::-1]
                        print(np.array(winner_history)[start_index:][::-1])
                        return np.array(winner_history)[start_index:][::-1]
                    else:
                        return None
                else:
                    return None
        else:
            return None

def get_history(current_position, indexer):
    """
    :param: current_position
    :param: indexer: has to be the database where current_position comes from
    :returns: reversed history of pedestrian from current position point
    """
    all_ped_frames = indexer.get_all_frames_by_ped_id(current_position.pedestrian)
    return indexer.get_ped_traj_xy(all_ped_frames)[::-1] # reverse, so the higher the index, the more into the past


def get_distance(coordinates, other_coordinates):
    """
    :param: coordinates:
    :param: other_coordinates:
    """
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
    #return np.linalg.norm(direction)
    return normalize(direction)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

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
    for _, entry in enumerate(list_of_points):
        if is_same_ped(point, entry) and not is_same_tuple(point, entry):
            return True, entry
    return False, None