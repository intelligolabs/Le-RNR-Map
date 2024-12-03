# %%
import torch
import matplotlib.pyplot as plt
import cv2
import quaternion as q
import numpy as np
from src.models.autoencoder.autoenc import Embedder
import imageio
from src.utils.render_utils import add_title, add_agent_view_on_w
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
import clip
from PIL import Image
from tqdm import tqdm
from lernr.smooth_utils.walkthrough_utils import get_smooth_trajectory
import pickle
from lernr.language import get_goal_points_single, get_goal_points_multi, save_heat_map, get_true_rot, get_start_position_rotation

# HABITAT LAB & SIM SETUP
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--scene", type=str, default="Cantwell")
parser.add_argument("--query_words", type=str, default="window")
parser.add_argument("--negative_prompt", type=str, default="stuff, things, objects, textures")
parser.add_argument("--multi_search", action="store_true")
parser.add_argument("--make_video", action="store_true")
parser.add_argument("--smooth", action="store_true")
parser.add_argument("--erased_area", type=int, default=500)
parser.add_argument("--th", type=float, default=0.6)
args = parser.parse_args()


scene = args.scene
make_video = args.make_video
multi_search = args.multi_search
TEXT_SEARCH = args.query_words
houseWords = args.negative_prompt
smooth = args.smooth
erase = args.erased_area
th = args.th

print("#"*40)
print("Scene: ", scene)
print("Multi search: ", multi_search)
print("Query words: ", TEXT_SEARCH)
print("Negative prompt: ", houseWords)
print("Make video: ", make_video)
print("Smooth: ", smooth)
if multi_search:
    print("Ereased area: ", erase)
    print("Treshold : ", th)
print("#"*40)


os.environ["GLOG_minloglevel"] = "3"
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

import habitat
from habitat import get_config
from habitat.sims import make_sim
from habitat.datasets import make_dataset

# %%
# HABITAT LAB & SIM SETUP
import os

os.environ["GLOG_minloglevel"] = "3"
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

import habitat
from habitat import get_config
from habitat.sims import make_sim
from habitat.datasets import make_dataset

config = get_config()
cfg = config
cfg.defrost()

#Dataset folder path
cfg.DATASET.SCENES_DIR = "/media/data/all_dataset/gibson/scene_dataset/gibson_habitat"

cfg.SIMULATOR.RGB_SENSOR.HEIGHT = 1024
cfg.SIMULATOR.RGB_SENSOR.WIDTH = 1024
cfg.SIMULATOR.DEPTH_SENSOR.HEIGHT = 1024
cfg.SIMULATOR.DEPTH_SENSOR.WIDTH = 1024
cfg.TASK.SENSORS = cfg.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
cfg.freeze()



#map path
map_path = f"./lernr_maps/{scene}_map_dict.pkl"
#check if the map of this scene is present
if not os.path.isfile(map_path):
    print(f"No map named {scene} founded")
    exit()
#loading map
with open(map_path, 'rb') as f:
    map_dict = pickle.load(f)


lernr_map = map_dict['map']
lernr_mask = map_dict['mask']
top_down_map_reconstruction_feature = lernr_map[:,0:32,:,:]
top_down_map_clip_feature = lernr_map[:,32:,:,:]
map_size = lernr_map.shape[-1]

print("lernr_map: ", lernr_map.shape)
print("lernr_mask: ", lernr_mask.shape)
print("top_down_map_reconstruction_feature: ", top_down_map_reconstruction_feature.shape)
print("top_down_map_clip_feature: ", top_down_map_clip_feature.shape)

try:
    sim.close()
except:
    pass
cfg.defrost()
cfg.SIMULATOR.SCENE = os.path.join(cfg.DATASET.SCENES_DIR, "{}.glb".format(scene))
cfg.freeze()
past_room = scene
sim = make_sim(id_sim=cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR)

# LOADING MODELS
device = "cuda:0"
embedder = Embedder(
    pretrained_ckpt="pretrained/autoenc_large.ckpt",
    img_res=lernr_map.shape[-1],
    w_size=map_size,
    coordinate_scale=32,
    w_ch=32,
    nerf_res=128,
    voxel_res=128,
)
embedder = embedder.to(device).eval()

action_mapping = {0: "stop", 1: "move_forward", 2: "turn left", 3: "turn right"}
OUT_RES = 64
device = "cuda:0" if torch.cuda.is_available() else "cpu"

#loading clip model
model, preprocess = clip.load("ViT-B/32", device=device)

B = top_down_map_clip_feature.shape[0]

# query words
query_list = TEXT_SEARCH.split(",")
#starting rotation of the navigation
start_rotation = q.quaternion(0.446406539135454, -0, 0.894830264249655, 0)

#origin pose of the navigation
orig_Rt = map_dict['origin_Rt']

#Getting a random starting position
start_position, _, map_coords = get_start_position_rotation(mask=lernr_mask.squeeze(), orig_Rt=orig_Rt, sim=sim, map_size=lernr_map.shape[-1])
sim.set_agent_state(start_position, start_rotation)

# getting the goal points to perform navigation
if not multi_search:
    goal_points, goal_pos, heat_maps = get_goal_points_single(query_list=query_list, houseWords=houseWords, clip_model=model, top_down_map_clip_feature=top_down_map_clip_feature, lernr_mask=lernr_mask, device=device, orig_Rt=orig_Rt, sim=sim, map_coords=map_coords)
else:
    goal_points, goal_pos, heat_maps = get_goal_points_multi(TEXT_SEARCH=query_list[0], houseWords=houseWords, clip_model=model, top_down_map_clip_feature=top_down_map_clip_feature, lernr_mask=lernr_mask, device=device, orig_Rt=orig_Rt, sim=sim, map_coords=map_coords, erase=erase, th=th)

#cleaning folder heat_maps
if multi_search:
    PATH = "multi_search"
else:
    PATH = "single_search"

try:
    files = os.listdir(f"./heat_maps/{PATH}")
    for file in files:
        file_path = os.path.join(f"./heat_maps/{PATH}", file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
except OSError:
    print("Error occurred while deleting files.")

#Saving the heat map of the search
save_heat_map(softmax_maps=heat_maps, goal_pos=goal_pos, query_list=query_list, houseWords=houseWords, multi_search=multi_search)

# navigation video creation
if make_video:
    follower = ShortestPathFollower(sim, 0.1, return_one_hot=False)
    
    origin = torch.eye(4).unsqueeze(0).to(device)
    diffs = []
    coordinate_scale = embedder.coordinate_scale
    map_size = embedder.w_size
    patch_size = map_size // 4
    angle_bin = 18

    print("map size:", map_size)
    print("patch size:", patch_size)
    print("angle bin", angle_bin)


    fig, ax = plt.subplots()
    done = False
    images = []
    VERBOSE = False
    VIS_RES = 256

    K = torch.eye(3)
    K[0, 0] = (embedder.img_res / 2.0) / np.tan(np.deg2rad(90.0) / 2)
    K[1, 1] = -(embedder.img_res / 2.0) / np.tan(np.deg2rad(90.0) / 2)
    K = K.unsqueeze(0).to(device)
    w, w_mask = None, None
    old_position = start_position
    #take the goal point
    goal_point = goal_points.pop(0)
    pose_trajectory = []
    Rt_t_goal = []
    goal_rt = []
    positions = []
    all_obj_pos = []
    c_rotation=None
    c_position=None
    goal_found = False
    query_index = 0

    while not done:
        if goal_found == True:
            pose_trajectory = []
            r = q.from_rotation_matrix(c_rotation)[0]
            sim.set_agent_state(c_position, r)
            goal_found=False
        
        state = sim.get_agent_state()
        #pose of the agent
        Rt_t = np.eye(4)
        Rt_t[:3, 3] = state.position
        Rt_t[:3, :3] = q.as_rotation_matrix(state.rotation)
        Rt_t = np.linalg.inv(Rt_t)
        Rt_t = Rt_t @ np.linalg.inv(orig_Rt)
        
        positions.append(state.position)

        Rt_t = torch.from_numpy(Rt_t).unsqueeze(0).float().to(device)
        pose_trajectory.append(Rt_t.unsqueeze(1))

        # get action that allows to arrive at destination
        best_action = follower.get_next_action(goal_point)

        if VERBOSE:
            print("action:", action_mapping[best_action])
        
        if best_action is None or best_action == 0:
            #object found
            print("Object found!")
            print("###############################")
            #It turns back of "previous_cells" cells 
            previous_cells = 5
            #delete last previous_cells cells
            for i in range(previous_cells):
                positions.pop()
                pose_trajectory.pop()
            #rotation towards the object
            if not multi_search:
                rt_GOAL = get_true_rot(word=query_list[query_index], position=positions[-1], sim=sim, device=device, model=model, preprocess=preprocess, orig_Rt=orig_Rt)
            else:
                rt_GOAL = get_true_rot(word=query_list[0], position=positions[-1], sim=sim, device=device, model=model, preprocess=preprocess, orig_Rt=orig_Rt)
            R_t_tmp = np.matmul(rt_GOAL.cpu().numpy(), orig_Rt)
            R_t_final = np.linalg.inv(R_t_tmp)
            #getting the last position and rotation in order to setting the sim state
            c_position = R_t_final[0][:3,3]
            c_rotation = R_t_final[:3,:3]

            goal_rt.append(rt_GOAL)
            query_index+=1

            #adding the goal pose to the list of poses
            pose_trajectory.append(rt_GOAL.unsqueeze(1))

            all_obj_pos.append(pose_trajectory)
            goal_found = True
            if len(goal_points) == 0:
                break
            
            goal_point = goal_points.pop(0)
            best_action = follower.get_next_action(goal_point)
            
        old_position = state.position
        sim.step(best_action)
        
    #process the list of poses
    k = 0
    new_Rts=torch.zeros((0,1,4,4))
    for pos in all_obj_pos:
        if k!=0:
            for i in range(1):
                pos.insert(0, new_Rts[-1,:,:,:].unsqueeze(0).to(device))
        pose_trajectory = torch.cat(pos, dim=1)
        # jitter camera pose a tiny amount to make sure each pose is unique
        # (to avoid problems with trajectory smoothing)
        pose_trajectory = pose_trajectory + torch.rand_like(pose_trajectory) * 1e-5 

        n_keypoints = len(pose_trajectory[0])
        new_Rts_tmp = pose_trajectory[0].unsqueeze(1).cpu()
        limit = 22
        #if smooth, the poses are interpolated
        if(n_keypoints >= limit and smooth):
            new_Rts_tmp = get_smooth_trajectory(Rt=pose_trajectory[0], n_frames= 5 * n_keypoints, subsample=7)

        rt = goal_rt[k].unsqueeze(1).cpu()
        goal_poses = torch.zeros((100,1,4,4)) + rt
        object_poses = torch.concatenate((new_Rts_tmp, goal_poses))
        new_Rts = torch.concatenate((new_Rts, object_poses))
        k+=1


    n_steps = len(new_Rts)
   
    images = []
    #navigation video generation
    for i in tqdm(range(n_steps)):
        with torch.no_grad():
            # getting the observation from Rt
            R_t_2 = (new_Rts[i:i+1].cpu().numpy())
            R_t_tmp = np.matmul(R_t_2, orig_Rt)
            R_t_final = np.linalg.inv(R_t_tmp)
            c_position = R_t_final[0,0][:3, 3]
            c_rotation = R_t_final[0,0][:3, :3]
            c_rotation = -q.from_rotation_matrix(c_rotation)
            current_observation = sim.get_observations_at(c_position, c_rotation)

            img = Image.fromarray(current_observation["rgb"]).resize((map_size, map_size))
            img = np.array(img)
            rgb = add_title(img, "Obs.")

            # nerf reconstruction from latent code
            recon_rgb, recon_depth = embedder.generate(
                    top_down_map_reconstruction_feature, {"Rt": new_Rts[i:i+1].to(device), "K": K.unsqueeze(1)}, out_res=64
                )
            recon_rgb = (
                    (recon_rgb.squeeze().permute(1, 2, 0).detach().cpu() * 255)
                    .numpy()
                    .astype(np.uint8)
                )

            recon_rgb = cv2.resize(recon_rgb, (rgb.shape[0], rgb.shape[1]))
            
            recon_rgb = add_title(recon_rgb, "Rec. obs")

            w_im = top_down_map_reconstruction_feature.mean(0).mean(0).detach().cpu().numpy()
            w_im = ((w_im - w_im.min()) / (w_im.max() - w_im.min()) * 255).astype(np.uint8)
            w_im = cv2.applyColorMap(w_im, cv2.COLORMAP_VIRIDIS)[:, :, ::-1]
            last_w_im = w_im

            w_im = add_agent_view_on_w(
                w_im,
                new_Rts[i:i+1],
                embedder.coordinate_scale,
                embedder.w_size,
                agent_size=4,
                view_size=15,
                target_position=goal_pos
            )
            w_im = np.fliplr(w_im)
            w_im = add_title(w_im, "Le-RNR-Map")

            # get the agent position in top-down map
            if isinstance(new_Rts[i:i+1], torch.Tensor):
                rt_temp = new_Rts[i:i+1].squeeze().detach().cpu().numpy()
            x, _, y = np.linalg.inv(rt_temp)[:3, 3]
            agent_y = int(x / (coordinate_scale / 2.0) * map_size / 2 + map_size / 2)
            agent_x = int(y / (coordinate_scale / 2.0) * map_size / 2 + map_size / 2)

           
            view_im = np.concatenate(
                [
                    np.concatenate([rgb, recon_rgb,w_im], 1)[
                        :, :, ::-1
                    ]
                ],
                1,
            )

            view_im = add_title(view_im, f"Query - {TEXT_SEARCH}")
            final_img = cv2.cvtColor(view_im, cv2.COLOR_RGB2BGR)
            images.append(final_img)
            #plt.imsave("view_im.jpg", final_img)
            if VERBOSE:
                print("images for gif:", len(images))


    print("###### gif creation ######")

    # %%
    import os

    # check if file exists
    if os.path.exists("video/query_map/query_map.gif"):
        os.remove("video/query_map/query_map.gif")
    if smooth:
        imageio.mimwrite("video/query_map/query_map.gif", images, duration=20, loop=0)
    else:
        imageio.mimwrite("video/query_map/query_map.gif", images, duration=10, loop=0)

    from IPython.display import Image

    Image("video/image-based-localization.gif")
    print("##" * 40)
    print("GIF created!")
