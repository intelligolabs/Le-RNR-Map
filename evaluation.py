import os
import pickle
import sys
import time
import numpy as np
import bz2
import _pickle as cPickle
import gzip
import quaternion
import matplotlib.pyplot as plt
import skimage
import torch


import habitat
from habitat import get_config
from habitat.sims import make_sim
from habitat.datasets import make_dataset
from tqdm import tqdm
import gzip
import json
import clip


import cv2
import numpy as np
import skfmm
import skimage
from numpy import ma

from lernr.utils import (
    get_habitat_coordinate_from_x_y_coordinate,
    sim_continuous_to_sim_map,
    sim_map_to_sim_continuous,
)


def get_mask(sx, sy, scale, step_size):
    size = int(step_size // scale) * 2 + 1
    mask = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if ((i + 0.5) - (size // 2 + sx)) ** 2 + (
                (j + 0.5) - (size // 2 + sy)
            ) ** 2 <= step_size**2 and ((i + 0.5) - (size // 2 + sx)) ** 2 + (
                (j + 0.5) - (size // 2 + sy)
            ) ** 2 > (
                step_size - 1
            ) ** 2:
                mask[i, j] = 1

    mask[size // 2, size // 2] = 1
    return mask


def get_dist(sx, sy, scale, step_size):
    size = int(step_size // scale) * 2 + 1
    mask = np.zeros((size, size)) + 1e-10
    for i in range(size):
        for j in range(size):
            if ((i + 0.5) - (size // 2 + sx)) ** 2 + (
                (j + 0.5) - (size // 2 + sy)
            ) ** 2 <= step_size**2:
                mask[i, j] = max(
                    5,
                    (
                        ((i + 0.5) - (size // 2 + sx)) ** 2
                        + ((j + 0.5) - (size // 2 + sy)) ** 2
                    )
                    ** 0.5,
                )
    return mask


class FMMPlanner:
    def __init__(self, traversible, scale=1, step_size=5):
        self.scale = scale
        self.step_size = step_size
        if scale != 1.0:
            self.traversible = cv2.resize(
                traversible,
                (traversible.shape[1] // scale, traversible.shape[0] // scale),
                interpolation=cv2.INTER_NEAREST,
            )
            self.traversible = np.rint(self.traversible)
        else:
            self.traversible = traversible

        self.du = int(self.step_size / (self.scale * 1.0))
        self.fmm_dist = None

    def set_goal(self, goal, auto_improve=False):
        traversible_ma = ma.masked_values(self.traversible * 1, 0)
        goal_x, goal_y = int(goal[0] / (self.scale * 1.0)), int(
            goal[1] / (self.scale * 1.0)
        )

        if self.traversible[goal_x, goal_y] == 0.0 and auto_improve:
            goal_x, goal_y = self._find_nearest_goal([goal_x, goal_y])

        traversible_ma[goal_x, goal_y] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd = ma.filled(dd, np.max(dd) + 1)
        self.fmm_dist = dd
        return

    def set_multi_goal(self, goal_map):
        traversible_ma = ma.masked_values(self.traversible * 1, 0)
        traversible_ma[goal_map == 1] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd = ma.filled(dd, np.max(dd) + 1)
        self.fmm_dist = dd
        return

    def get_short_term_goal(self, state):
        scale = self.scale * 1.0
        state = [x / scale for x in state]
        dx, dy = state[0] - int(state[0]), state[1] - int(state[1])
        mask = get_mask(dx, dy, scale, self.step_size)
        dist_mask = get_dist(dx, dy, scale, self.step_size)

        state = [int(x) for x in state]

        dist = np.pad(
            self.fmm_dist,
            self.du,
            "constant",
            constant_values=self.fmm_dist.shape[0] ** 2,
        )
        subset = dist[
            state[0] : state[0] + 2 * self.du + 1, state[1] : state[1] + 2 * self.du + 1
        ]

        assert (
            subset.shape[0] == 2 * self.du + 1 and subset.shape[1] == 2 * self.du + 1
        ), "Planning error: unexpected subset shape {}".format(subset.shape)

        subset *= mask
        subset += (1 - mask) * self.fmm_dist.shape[0] ** 2

        if subset[self.du, self.du] < 0.25 * 100 / 5.0:  # 25cm
            stop = True
        else:
            stop = False

        subset -= subset[self.du, self.du]
        ratio1 = subset / dist_mask
        subset[ratio1 < -1.5] = 1

        (stg_x, stg_y) = np.unravel_index(np.argmin(subset), subset.shape)

        if subset[stg_x, stg_y] > -0.0001:
            replan = True
        else:
            replan = False

        return (
            (stg_x + state[0] - self.du) * scale,
            (stg_y + state[1] - self.du) * scale,
            replan,
            stop,
        )

    def _find_nearest_goal(self, goal):
        traversible = (
            skimage.morphology.binary_dilation(
                np.zeros(self.traversible.shape), skimage.morphology.disk(2)
            )
            != True
        )
        traversible = traversible * 1.0
        planner = FMMPlanner(traversible)
        planner.set_goal(goal)

        mask = self.traversible

        dist_map = planner.fmm_dist * mask
        dist_map[dist_map == 0] = dist_map.max()

        goal = np.unravel_index(dist_map.argmin(), dist_map.shape)

        return goal


def show_me_at(sim, x, y, map_obj_origin):
    coords = (y, x)
    new_pos = sim_map_to_sim_continuous(coords=coords, map_obj_origin=map_obj_origin)
    obs = sim.get_observations_at(new_pos, rot)
    rgb = obs["rgb"]
    return rgb


def get_goals_from_lernr_map(
    lernr_map: torch.Tensor,
    lernr_mask: torch.Tensor,
    goal_name: str,
    houseWords: list,
    origRT,
    topk: int = 5,
    model=None,
    device=None,
):
    """
    Args:
        lernr_map (torch.Tensor): _description_
        goal_name (str): _description_
        topk (int, optional): _description_. Defaults to 5.

    Returns:
        List[Tuple[int,int]]: coordinates in the map of the topk goals
    """
    h, w = lernr_map.shape[2], lernr_map.shape[3]
    B = lernr_map.shape[0]

    locations = []
    lernr_map_softmax = None

    # query word

    TOP_K_MATCHES = topk
    
    wList = houseWords.split(",")
    text_words = wList

    text_words.insert(0, f"{goal_name}")

    words = clip.tokenize(text_words).to(device)
    with torch.no_grad():
        text_features = model.encode_text(words)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

    map_reshaped = lernr_map.permute(0, 2, 3, 1)
    resText = (
        torch.matmul(
            map_reshaped.reshape(1, -1, 512).float(),
            torch.transpose(text_features.float(), 1, 0),
        )
        * 100
    )
    lernr_map_softmax = resText.softmax(dim=-1)

    lernr_map_softmax = torch.reshape(lernr_map_softmax[0, :, 0].to(device), (h, w))

    # Only the similarities in the map are taken into account
    mask = lernr_mask.squeeze()
    mask = torch.flatten(mask)

    map = torch.flatten(lernr_map_softmax)
    map_points = mask <= 1
    indices = map_points.nonzero()
    m = torch.flatten(map)
    m[indices] = 0

    lernr_map_softmax = torch.reshape(m, (h, w))

    


    max_values, max_indices = torch.topk(lernr_map_softmax.view(B, -1), k=TOP_K_MATCHES)
    max_rows = max_indices // lernr_map.shape[3]
    max_cols = max_indices % lernr_map.shape[3]

    

    for GX, GY in zip(max_cols.squeeze(), max_rows.squeeze()):
        locations.append([GX.item(), GY.item()])

    habitat_locs = []
    for loc in locations:
        p, r = get_habitat_coordinate_from_x_y_coordinate(loc[1], loc[0], origRT)
        habitat_locs.append(p)
    
    return habitat_locs, locations, lernr_map_softmax.detach().cpu()


def debug(goal_map_location, lernr_goal_pos):
    x, y = goal_map_location
    ###### DEBUG
    plt.figure(figsize=(10, 10))
    plt.suptitle(f"scene: {scene_name}, goal: {goal_name}")
    plt.subplot(2, 2, 1)
    plt.title("GT")
    plt.imshow(map_dsts)
    plt.scatter(x, y, marker="*", c="r", s=20)
    plt.subplot(2, 2, 2)
    plt.title("GT rgb")
    rgb = show_me_at(sim, x, y, map_obj_origin)
    plt.imshow(rgb)
    plt.subplot(2, 2, 3)
    plt.title("lernr")
    x, y = lernr_goal_pos
    plt.imshow(lernr_map_softmax)
    plt.scatter(x, y, marker="*", c="r", s=20)
    plt.subplot(2, 2, 4)
    plt.title("lernr rgb")
    obs = sim.get_observations_at(top1_hab, rot)
    rgb = obs["rgb"]
    plt.imshow(rgb)

    plt.savefig("current.png")
    plt.show()
    # plt.pause(5)
    ######


def unproj_gibson(coords, map_obj_origin):
    # unproject from 2D to SIM 3D
    pos = [0, 0, 0]
    min_x, min_y = map_obj_origin / 100.0
    x, y = coords[0:2]
    hab_loc = (-(y / 20) - min_y), (-(x / 20) - min_x)
    pos[2] = -hab_loc[0]
    pos[0] = -hab_loc[1]
    return pos


def proj_gibson(pos, map_obj_origin):
    # project from SIM 3D to 2D
    x = -pos[2]
    y = -pos[0]
    min_x, min_y = map_obj_origin / 100.0
    map_y = int((-y - min_y) * 20.0)
    map_x = int((-x - min_x) * 20.0)
    map_loc = [map_x, map_y]
    return map_loc


DATASET_PATH = "/media/data/all_dataset"


if __name__ == "__main__":
    
    #Negative prompt
    if len(sys.argv) > 1:
        negative_prompts = sys.argv[1]
    else:
        negative_prompts = "things, stuff, textures, objects"
 
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    # container for results
    SR = []  # TODO
    SPL = []  # TODO

    # CREATE CONFIGURATION
    config = get_config()
    cfg = config
    cfg.defrost()
    habitat_api_path = os.path.join(os.path.dirname(habitat.__file__), "../")
    cfg.DATASET.SCENES_DIR = os.path.join(habitat_api_path, cfg.DATASET.SCENES_DIR)
    cfg.DATASET.DATA_PATH = os.path.join(
        habitat_api_path, cfg.DATASET.DATA_PATH.replace("habitat-test-scenes", "gibson")
    )

    cfg.DATASET.SCENES_DIR = (
        f"{DATASET_PATH}/gibson/scene_dataset/gibson_habitat"
    )
    cfg.DATASET.DATA_PATH = f"{DATASET_PATH}/gibson/object_nav_jsons/objectnav/gibson/v1.1/{{split}}/{{split}}.json.gz"
    cfg.SIMULATOR.SCENE_DATASET = f"{DATASET_PATH}/gibson/scene_dataset/gibson_habitat/"
    
    cfg.DATASET.TYPE = "PointNav-v1"
    cfg.SIMULATOR.RGB_SENSOR.HEIGHT = 128
    cfg.SIMULATOR.RGB_SENSOR.WIDTH = 128
    cfg.SIMULATOR.DEPTH_SENSOR.HEIGHT = 128
    cfg.SIMULATOR.DEPTH_SENSOR.WIDTH = 128
    cfg.TASK.SENSORS = cfg.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    cfg.freeze()

    dataset = make_dataset("PointNav-v1")
    cfg.defrost()
    cfg.DATASET.SPLIT = "val"
    cfg.freeze()
    val_scenes = dataset.get_scenes_to_load(cfg.DATASET)
    all_scenes = val_scenes
    print(f"Total {len(all_scenes)} scenes found")
    print(all_scenes)

    for i, scene in enumerate(all_scenes):
        if "episodes" in scene:
            continue
        print(i, scene)

    success_th = 1.0
    success_rate = 0
    total = 0
    distances = {}
    # TRAVERSING ALL THE VALIDATION SCENE
    for scene in tqdm(all_scenes, desc="Scene in val"):
        if "episodes" in scene:
            continue

        if "Collierville" in scene:
            continue
        
        # Filter scene
        #if "Wiconisco" not in scene:
        #    continue 

        total += 1
        print(scene)
        cfg.defrost()
        cfg.SIMULATOR.SCENE = os.path.join(cfg.DATASET.SCENES_DIR, scene)
        cfg.freeze()

        sim = make_sim(id_sim=cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR)

        # NOW LOAD THE EPISODES - json file
        file = scene.replace(".glb", "")
        episode_path = f"{DATASET_PATH}/gibson/object_nav_jsons/objectnav/gibson/v1.1/val/content/{file}_episodes.json.gz"
        with gzip.open(episode_path, "rb") as file:
            content = file.read().decode("utf-8")
            episodes_content = json.loads(content)

        # NOW LOAD THE GOAL INFO - json file
        with gzip.open(
            f"{DATASET_PATH}/gibson/object_nav_jsons/objectnav/gibson/v1.1/val/content/{scene}.json.gz",
            "rb",
        ) as file:
            content = file.read().decode("utf-8")
            GOAL_INFO = json.loads(content)
        print("GOAl info", GOAL_INFO)  

        with bz2.BZ2File(
            f"{DATASET_PATH}/gibson/object_nav_jsons/objectnav/gibson/v1.1/val/val_info.pbz2",
            "rb",
        ) as f:
            dataset_info = cPickle.load(f)
            # contains also the semantic map

        for episode in episodes_content["episodes"]:
            floor_idx = episode["floor_id"]
            scene_name = episode["scene_id"].replace(".glb", "").split("/")[-1]
            goal_name = episode["object_category"]
            goal_idx = episode["object_id"]

            pos = episode["start_position"]
            rot = quaternion.from_float_array(episode["start_rotation"])

            # Load scene info
            scene_info = dataset_info[scene_name]
            sem_map = scene_info[floor_idx]["sem_map"]
            map_obj_origin = scene_info[floor_idx]["origin"]

            # Setup ground truth planner
            object_boundary = 1 
            map_resolution = 5  
            selem = skimage.morphology.disk(2)
            traversible = skimage.morphology.binary_dilation(sem_map[0], selem) != True
            traversible = 1 - traversible
            planner = FMMPlanner(traversible)
            selem = skimage.morphology.disk(
                int(object_boundary * 100.0 / map_resolution)
            )
            goal_map = (
                skimage.morphology.binary_dilation(sem_map[goal_idx + 1], selem) != True
            )
            goal_map = 1 - goal_map
            planner.set_multi_goal(goal_map)

            

            # Get starting loc in GT map coordinates
            x = -pos[2]
            y = -pos[0]
            min_x, min_y = map_obj_origin / 100.0
            map_loc = int((-y - min_y) * 20.0), int((-x - min_x) * 20.0)

            gt_planner = planner
            starting_loc = map_loc
            object_boundary = object_boundary
            goal_idx = goal_idx
            goal_name = goal_name
            map_obj_origin = map_obj_origin

            starting_distance = (
                gt_planner.fmm_dist[starting_loc] / 20.0 + object_boundary
            )

            map_dsts = planner.fmm_dist
            h, w = map_dsts.shape[0:2]



            # Load lernr map
            with open(f"lernr_maps/{scene_name}_map_dict.pkl", "rb") as f:
                lernr_data = pickle.load(f)

            print("Loading map:", f"lernr_maps/{scene_name}_map_dict.pkl")
            lernr_map = lernr_data["map"]
            lernr_mask = lernr_data["mask"]
            if "origin_Rt" not in lernr_data:
                orig_Rt = lernr_data["origin_pose"]
            else:
                orig_Rt = lernr_data["origin_Rt"]

            topk = 25
            habitat_locs, locations, lernr_map_softmax = get_goals_from_lernr_map(
                lernr_map[:, 32:, :, :],
                lernr_mask,
                goal_name,
                negative_prompts,
                orig_Rt,
                topk,
                model,
                device,
            )

            top1_hab = habitat_locs[0]
            top1_lernr = locations[0]

            
            
            # goal found from RNR projected to gibson map
            projected = proj_gibson(top1_hab, map_obj_origin)

            found = goal_map[projected[1], projected[0]]
            dist = 0

            if not found:
                # non zero indices
                nz = np.nonzero(goal_map)
                # get closest non zero indexes (row columns)
                closest = np.argmin(
                    np.linalg.norm(
                        np.array(nz) - np.array(projected)[:, None], axis=0
                    )
                )
                goal_map_location = [nz[1][closest], nz[0][closest]]
                goal_hab = sim_map_to_sim_continuous(
                    coords=[goal_map_location[1], goal_map_location[0]],
                    map_obj_origin=map_obj_origin,
                )
                dist = np.linalg.norm(np.array(top1_hab) - np.array(goal_hab))
                if dist < 0:
                    print("IMPOSSIBRU")

            debug_mode = False
            if debug_mode:
                plt.figure(figsize=(10, 10))
                plt.suptitle(
                    f"Goal name: {goal_name} | DTS: {dist} | Success: {found}"
                )
                plt.subplot(1, 2, 1)
                plt.imshow(goal_map)
                if not found:
                    plt.scatter(
                        goal_map_location[0],
                        goal_map_location[1],
                        marker="*",
                        c="g",
                        s=50,
                    )
                plt.scatter(projected[0], projected[1], marker="*", c="r", s=50)
                plt.subplot(1, 2, 2)
                plt.imshow(map_dsts)
                if not found:
                    plt.scatter(
                        goal_map_location[0],
                        goal_map_location[1],
                        marker="*",
                        c="g",
                        s=50,
                    )
                plt.scatter(projected[0], projected[1], marker="*", c="r", s=50)
                plt.show()

            if scene_name not in distances:
                distances[scene_name] = []
            distances[scene_name].append(dist)
            print("Scene name: ", scene_name, "Goal name", goal_name, "Distance ", dist)


        if scene_name not in distances:
            distances[scene_name] = []
        dists = np.asarray(distances[scene_name])

        total = len(dists)
        success_rate = np.sum(np.array(dists) < success_th) / total
        dts = dists - success_th
        dts[dts < 0] = 0  # equivalent of max(0, d - th)

        print("Scene", scene_name, "Success", success_rate, "DTS (m)", np.mean(dts))

        sim.close()

    fname = f"{int(time.time())}_results.txt"
    fp = open(fname, "w")

    # For each scene
    print("---" * 25)
    for scene_name in distances.keys():
        dists = np.asarray(distances[scene_name])

        total = len(dists)
        success_rate = np.sum(np.array(dists) < success_th) / total
        dts = dists - success_th
        dts[dts < 0] = 0  # equivalent of max(0, d - th)

        mean_dts = np.mean(dts)
        print("Scene", scene_name, "Success", success_rate, "DTS (m)", mean_dts)
        print(
            "Scene", scene_name, "Success", success_rate, "DTS (m)", mean_dts, file=fp
        )

        if scene_name == "Corozal":
            if success_rate >= 0.685: # or mean_dts < 2.569954:
                with open("winners.txt", "a") as ffp:
                    print(f"WINNER for {scene_name}!", fname, negative_prompts, file=ffp)

    # Gibson val set
    total = sum([len(distances[scene_name]) for scene_name in distances.keys()])
    success_rate = 0
    dts = 0
    for scene_name in distances.keys():
        for d in distances[scene_name]:
            if d < success_th:
                success_rate += 1
            dts += max(0, d - success_th)

    print("Negative prompts: ", negative_prompts, file=fp)
    print(
        f"Gibson results: {success_rate/total:.3f} | DTS (m) {dts/total:.4f}", file=fp
    )

    print("Negative prompts: ", negative_prompts)
    print(f"Gibson results: {success_rate/total:.3f} | DTS (m) {dts/total:.4f}")