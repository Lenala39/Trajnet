from __future__ import division
import math


class Collision():
    def __init__(self):
        pass

    def check_for_collisions(self, next_step_x, next_step_y, last_step_x, last_step_y, scene_id, indexer):
        """
        Checks whether the computed next step will create a collision
        :param: next_step_x
        :param: next_step_y
        :param: last_step_x
        :param: last_step_y
        :param: indexer = required indexer object for grid computations
        :returns: bool for collision or not
        """
        # get frames that cross in the next step that should be taken
        grid_frames = indexer.get_frames_at_xy(next_step_x, next_step_y)
        scene_frames = indexer.get_all_frames_by_scene_id(scene_id=scene_id)[0]
        grid_frames_in_scene = []
        for g_frame in grid_frames:
            for s_frame in scene_frames:
                if g_frame.frame == s_frame.frame and g_frame.pedestrian == s_frame.pedestrian:
                    grid_frames_in_scene.append(g_frame)


        cell_tracks = {}
        # iterate over all frames in Grid-Cell
        for frame in grid_frames_in_scene:
            if not (frame.pedestrian in cell_tracks.keys()):
                # get all frames of the pedestrian (not just in grid cell)
                pedestrian_frames = indexer.get_all_frames_by_ped_id(frame.pedestrian)
                # filter so that only one frame before or after or the same frame remain in list
                try:
                    pedestrian_frames = [fr for fr in pedestrian_frames if (
                            (fr.frame == frame.frame + 1) or (fr.frame == frame.frame))]
                except IndexError:
                    pass
                cell_tracks[frame.pedestrian] = indexer.get_ped_traj_xy(pedestrian_frames)

        # iterate over the paths of the pedestrians in the grid cell
        for ped, track in cell_tracks.items():
            new_line = ([next_step_x, next_step_y], [last_step_x, last_step_y])

            # create line that crosses in the cell
            try:
                line2 = (track[1], track[0])
                collide, intersection_point = self.line_intersection(line2, new_line)
            except IndexError:
                collide = False

            if collide:
                return True
            else:
                continue

        return False

    def line_intersection(self, existing_line, new_line):
        """"
        Computes the intersection between two lines
        :param: existing line = line from database in grid cell
        :param: new_line = newly computed line
        :returns: True, (intersection_x, intersection_y) if intersection
        :returns: False, (None, None) if no intersection
        """
        # get differences in x- and y-values
        xdiff = (existing_line[0][0] - existing_line[1][0], new_line[0][0] - new_line[1][0])
        ydiff = (existing_line[0][1] - existing_line[1][1], new_line[0][1] - new_line[1][1])

        # determinant of a, b
        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        # if determinant 0: no intersection
        div = det(xdiff, ydiff)
        if div == 0:
            return False, (None, None)

        # else: compute intersection point
        d = (det(*existing_line), det(*new_line))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        intersection_point = (x, y)

        # compute if collision/intersection is on new segment of line
        collide = self.intersection_in_new_segment(existing_line, intersection_point)
        # only if collision on new segment, return true
        if collide:
            return True, intersection_point
        else:
            return False, (None, None)

    def intersection_in_new_segment(self, new_line, intersection_point):
        """
        Computes whether intersection in current grid cell
        returns: bool
        """
        def distance(point_1, point_2):
            return math.sqrt((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2)

        point_1 = new_line[0]
        point_2 = new_line[1]
        # if intersection_point is between the endpoints of the segment of new_line
        # then the distance between (start_new to intersection) + (intersection to end_new) == (start_new, end_new)
        return distance(point_1, intersection_point) + distance(point_2, intersection_point) == distance(point_1, point_2)


