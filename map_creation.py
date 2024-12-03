import torch
import matplotlib.pyplot as plt
import cv2
import quaternion as q
import numpy as np
from src.models.autoencoder.autoenc import Embedder
from src.utils.render_utils import add_agent_view_on_w, add_title
import imageio
import clip
from PIL import Image
import pickle
import time
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
import habitat
from habitat import get_config
from habitat.sims import make_sim
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--scene", type=str, default="Cantwell")
parser.add_argument("--n_goals", type=int, default=25)
parser.add_argument("--make_video", action="store_true")
parser.add_argument("--img_res", type=int, default=128)
parser.add_argument("--map_size", type=int, default=128)

args = parser.parse_args()


scene = args.scene
NUM_GOALS = args.n_goals
make_video = args.make_video
img_res = args.img_res
w_size = args.map_size

print("#"*20)
print("Scene: ", scene)
print("N_goals: ", NUM_GOALS)
print("img_res: ", img_res)
print("map_size: ", w_size)
print("make_video: ", make_video)
print("#"*20)
os.environ['GLOG_minloglevel'] = "3"
os.environ['MAGNUM_LOG'] = "quiet"
os.environ['HABITAT_SIM_LOG'] = "quiet"

config = get_config()
cfg = config
cfg.defrost()

#Dataset folder path
cfg.DATASET.SCENES_DIR = '/media/data/all_dataset/gibson/scene_dataset'

cfg.SIMULATOR.RGB_SENSOR.HEIGHT = 1024
cfg.SIMULATOR.RGB_SENSOR.WIDTH = 1024
cfg.SIMULATOR.DEPTH_SENSOR.HEIGHT = 1024
cfg.SIMULATOR.DEPTH_SENSOR.WIDTH = 1024
cfg.TASK.SENSORS = cfg.SIMULATOR.AGENT_0.SENSORS = ['RGB_SENSOR', 'DEPTH_SENSOR']
cfg.SIMULATOR.AGENT_0.HEIGHT = 1.25
cfg.freeze()


device = 'cuda'
embedder = Embedder(pretrained_ckpt='pretrained/autoenc_large.ckpt',
                   img_res=img_res, w_size=w_size, coordinate_scale=32, w_ch=32, nerf_res=128, voxel_res=128)
embedder = embedder.to(device).eval()

try: sim.close()
except: pass
cfg.defrost()
cfg.SIMULATOR.SCENE = os.path.join(cfg.DATASET.SCENES_DIR,'gibson_habitat/{}.glb'.format(scene))
cfg.freeze()
past_room = scene

#check scene
if not os.path.isfile(cfg.SIMULATOR.SCENE):
    print("No scene named: ", scene)
    exit()
#simulator
sim = make_sim(id_sim=cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR)

# loading Clip model
model, preprocess = clip.load("ViT-B/32", device=device)

follower = ShortestPathFollower(sim, 0.5, return_one_hot=False)

#starting rotation
start_rot = q.quaternion(0.446406539135454, -0, 0.894830264249655, 0)

#starting position
start_position = sim.sample_navigable_point()
print("Starting position: ", start_position)
start_rotation = start_rot

#starting pose
orig_Rt = np.eye(4)
orig_Rt[:3,3] = start_position
orig_Rt[:3,:3] = q.as_rotation_matrix(start_rotation)
orig_Rt = np.linalg.inv(orig_Rt)

print("Starting pose: ", orig_Rt)
goal_points = []
# defining goal points 
for i in range(NUM_GOALS):
    while True:
        random_point = sim.sample_navigable_point()
        if abs(random_point[1]-start_position[1]) > 0.5: continue
        break
    goal_points.append(random_point)

sim.set_agent_state(start_position, start_rotation)
done = False
goal_point = goal_points.pop(0)

K = torch.eye(3)
K[0,0] = (embedder.img_res/2.) / np.tan(np.deg2rad(90.0) / 2)
K[1,1] = -(embedder.img_res/2.) / np.tan(np.deg2rad(90.0) / 2)
K = K.unsqueeze(0).to(device)

w_nerf, w_clip, w_mask = None, None, None
imgs = []
i = 0

n_images = 0
print("Navigation started")
start_time = time.time()
while not done:
    state = sim.get_agent_state()
    n_images+=1
    
    #getting the observation
    obs = sim.get_observations_at(state.position, state.rotation)
    #state position
    Rt_t = np.eye(4)
    Rt_t[:3,3] = state.position
    Rt_t[:3,:3] = q.as_rotation_matrix(state.rotation)
    Rt_t = np.linalg.inv(Rt_t)
    Rt_t = Rt_t @ np.linalg.inv(orig_Rt)
    #rgb observation resized
    rgb_resized = Image.fromarray(obs['rgb'])
    rgb_resized = rgb_resized.resize((img_res,img_res))
    rgb_resized = np.array(rgb_resized)
    #depth observation resized
    depth_resized = cv2.resize(obs['depth'],(img_res,img_res))
    depth_resized = depth_resized.reshape((img_res,img_res,1))
    #to torch
    rgb_t = torch.from_numpy(rgb_resized/255.).unsqueeze(0).permute(0,3,1,2).to(device)
    depth_t = torch.from_numpy(depth_resized).unsqueeze(0).permute(0,3,1,2).to(device)
    Rt_t = torch.from_numpy(Rt_t).unsqueeze(0).float().to(device)
    #clip features
    rgb_clip = preprocess(Image.fromarray(obs['rgb'])).unsqueeze(0).to(device)
    image_features = model.encode_image(rgb_clip)
    image_features = image_features / image_features.norm(dim=1, keepdim=True)

    with torch.no_grad():
        output = embedder.calculate_mask_func(depth_t*10.0, Rt_t, K)
        sorted_indices, seq_unique_list, seq_unique_counts, _ = output
        input_dict = {'rgb': rgb_t.unsqueeze(1),
                      'depth': depth_t.unsqueeze(1),
                    'sorted_indices': sorted_indices.unsqueeze(1),
                    'seq_unique_counts': seq_unique_counts.unsqueeze(1),
                      'seq_unique_list': seq_unique_list.unsqueeze(1)}
        
        #maps generation
        w_nerf, w_mask = embedder.embed_obs_enc(input_dict, past_w=w_nerf, past_w_num_mask=w_mask)
        w_clip, w_mask = embedder.embed_obs_clip(input_dict, past_w=w_clip, past_w_num_mask=w_mask, clip_feat=image_features)
    
        if make_video:
            #nerf reconstruction
            recon_rgb, _ = embedder.generate(w_nerf, {'Rt': Rt_t.unsqueeze(1), 'K':K.unsqueeze(1)}, out_res=64)

            orig_rgb = add_title(rgb_resized, 'observation')
            recon_rgb = (recon_rgb.squeeze().permute(1,2,0).detach().cpu() * 255).numpy().astype(np.uint8)
            recon_rgb = cv2.resize(recon_rgb, (orig_rgb.shape[0], orig_rgb.shape[1]))
            recon_rgb = add_title(recon_rgb, 'recon obs.')

            w_im = w_clip.mean(0).mean(0).detach().cpu().numpy()
            w_im = ((w_im - w_im.min())/(w_im.max()-w_im.min()) * 255).astype(np.uint8)
            w_im = cv2.applyColorMap(w_im, cv2.COLORMAP_VIRIDIS)[:,:,::-1]
            last_w_im = w_im
            w_im = add_agent_view_on_w(w_im, Rt_t, embedder.coordinate_scale, embedder.w_size, agent_size=4, view_size=15)
            w_im = np.fliplr(w_im)
            
            w_im = cv2.resize(w_im,(orig_rgb.shape[0],orig_rgb.shape[1]))
            w_im = add_title(w_im, 'lernr_map')

            view_im = np.concatenate([orig_rgb, recon_rgb, w_im],1)
        
            imgs.append(view_im)

    
    action = follower.get_next_action(goal_point)
        
    while action is None or action == 0:
        if len(goal_points) == 0:
            done = True
            break
        goal_point = goal_points.pop(0)
        print(len(goal_points), "goal points left")
        action = follower.get_next_action(goal_point)
    if not done:
        sim.step(action)

print("Generation takes:  %s seconds" % (time.time() - start_time))
print("Images seen: ", n_images)
#lernr map creation
lernr_map = torch.concatenate((w_nerf, w_clip), 1)
#dictionary creation
map_dict = {'map': lernr_map, 'mask': w_mask, 'origin_Rt': orig_Rt}

# saving map dictionary
with open(f'lernr_maps/{scene}_map_dict.pkl', 'wb') as fp:
    pickle.dump(map_dict, fp)
    print('dictionary saved successfully to file')
print("make_video: ", make_video)
if make_video:
    if os.path.exists("video/map_generation/map_generation.gif"):
        os.remove("video/map_generation/map_generation.gif")
    print("Gif creation...")
    imageio.mimwrite("video/map_generation/map_generation.gif", imgs, duration=10, loop=0)
    from IPython.display import Image

    Image("video/map_generation/map_generation.gif")
    print("##" * 40)
    print("GIF created!")
