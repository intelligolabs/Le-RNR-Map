import torch
import copy
import numpy as np
import quaternion as q
import random
import matplotlib.pyplot as plt
import clip
from PIL import Image

def sim_map_to_sim_continuous(coords, map_obj_origin):
    """Converts ground-truth 2D Map coordinates to absolute Habitat
    simulator position and rotation.
    """
    # agent_state = self._env.sim.get_agent_state(0)
    y, x = coords
    min_x, min_y = map_obj_origin / 100.0

    cont_x = x / 20. + min_x
    cont_y = y / 20. + min_y
    # agent_state.position[0] = cont_y
    # agent_state.position[2] = cont_x
    new_pos = np.array([cont_y, 0, cont_x])
    return new_pos

def sim_continuous_to_sim_map(sim_loc, map_obj_origin):
    """Converts absolute Habitat simulator pose to ground-truth 2D Map
    coordinates.
    """
    x, y, o = sim_loc
    min_x, min_y = map_obj_origin / 100.0
    x, y = int((-x - min_x) * 20.), int((-y - min_y) * 20.)

    o = np.rad2deg(o) + 180.0
    return y, x, o

    
def get_habitat_coordinate_from_x_y_coordinate(goal_x,
                                               goal_y,
                                               orig_Rt,
                                               canonical_angle=np.pi,
                                               map_size=128,
                                               coordinate_scale=32,
                            ):
    '''
    aim of this function is to revert the process of localization.
    Given a goal_x, goal_y, we want to retrieve this position in habitat-lab coordinate reference system
    '''
    x_temp = ((goal_x - map_size / 2) * (coordinate_scale / 2)) / (map_size / 2)
    y_temp = ((goal_y - map_size / 2) * (coordinate_scale / 2)) / (map_size / 2)
    
    rotation_inverse = q.as_rotation_matrix(q.from_euler_angles([0.0, canonical_angle, 0.0]))
    R_t_inverse = np.eye(4)
    R_t_inverse[:3, :3] = rotation_inverse
    R_t_inverse[:3, 3] = np.array([x_temp, 0, y_temp])

    R_t_2 = np.linalg.inv(R_t_inverse)
    R_t_tmp = np.matmul(R_t_2, orig_Rt)
    R_t_final = np.linalg.inv(R_t_tmp)
    pred_position = R_t_final[:3, 3]
    pred_rotation = R_t_final[:3, :3]
    return pred_position, q.from_rotation_matrix(pred_rotation)



