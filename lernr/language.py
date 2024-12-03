import torch
import copy
import numpy as np
import quaternion as q
import random
import matplotlib.pyplot as plt
import clip
from PIL import Image
from lernr.utils import get_habitat_coordinate_from_x_y_coordinate
import os

#getting the nearest navigable cell
def get_goal_cell(target_x=None, target_y=None, map_coords=None, orig_Rt=None, sim=None, device=None, map_size=128):
    
    goal = torch.tensor([int(target_y), int(target_x)]).to(device)

    dist = (map_coords-goal).pow(2).sum(1).sqrt()
    _, ind = torch.sort(dist)
    nearest_cell = map_coords[ind]
    
    GOAL_X, GOAL_Y = (
        nearest_cell[0,1].cpu().numpy(),
        nearest_cell[0,0].cpu().numpy(),
    )

    query_position, pred_rotation = get_habitat_coordinate_from_x_y_coordinate(
    GOAL_Y, GOAL_X, orig_Rt, map_size=map_size
    )
    index=1
    while not sim.is_navigable(query_position):
        GOAL_X, GOAL_Y = (
            nearest_cell[index,1].cpu().numpy(),
            nearest_cell[index,0].cpu().numpy(),
        )
        query_position, pred_rotation = get_habitat_coordinate_from_x_y_coordinate(
            GOAL_Y, GOAL_X, orig_Rt, map_size=map_size
        )
        index = index+1 
        
    pred_position_ = query_position
  
    return pred_position_, pred_rotation

#getting the random start position
def get_start_position_rotation(mask=None, orig_Rt=None, sim=None, map_size=128):
    
    map_points = mask > 1
    indices = map_points.nonzero()

    listCoord = indices.tolist()
    random.shuffle(listCoord)
    
    s_position, s_rotation = get_habitat_coordinate_from_x_y_coordinate(
    listCoord[0][0], listCoord[0][1], orig_Rt, map_size=map_size
    )

    start_position = s_position
    start_rotation = s_rotation

    index = 1
    while not sim.is_navigable(start_position) or index > len(listCoord):
        s_position, s_rotation = get_habitat_coordinate_from_x_y_coordinate(
        listCoord[index][0], listCoord[index][1], orig_Rt, map_size=map_size
        )
        index = index + 1
        start_position = s_position
        start_rotation = s_rotation
    
    
    return start_position, start_rotation, indices

#
def get_soft_map(TEXT_SEARCH=None, houseWords=None, top_down_map_clip_feature=None, lernr_mask=None, clip_model=None, device=None, map_size=128):
    wList = houseWords.split(",")
    text_words = wList

    text_words.insert(0, TEXT_SEARCH)

    words = clip.tokenize(text_words).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(words)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

    map_reshaped = top_down_map_clip_feature.permute(0, 2, 3, 1)
    resText = (
        torch.matmul(
            map_reshaped.reshape(1, -1, 512).float(),
            torch.transpose(text_features.float(), 1, 0),
        )
        * 100
    )
    res_softmax = resText.softmax(dim=-1)

    softmax_map = torch.reshape(res_softmax[0, :, 0].to(device), (map_size, map_size))
    
    #Only the similarities in the map are taken into account
    mask = lernr_mask.squeeze()
    mask = torch.flatten(mask)
    
    flat_map = torch.flatten(softmax_map)
    map_points = (mask <=1)
    indices = map_points.nonzero()
    flat_map[indices] = 0
    
    softmax_map = torch.reshape(flat_map,(map_size,map_size))

    return softmax_map

#getting the coordinate of the goal
def get_goal_map_coordinate(TEXT_SEARCH=None, houseWords=None, top_down_map_clip_feature=None, top_down_map_clip_mask=None ,device=None, model=None, map_size=128):
    
    B = top_down_map_clip_feature.shape[0]

    # query word
    
    TOP_K_MATCHES = 100

    wList = houseWords.split(",")
    text_words = wList

    text_words.insert(0, TEXT_SEARCH)

    words = clip.tokenize(text_words).to(device)
    with torch.no_grad():
        text_features = model.encode_text(words)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

    map_reshaped = top_down_map_clip_feature.permute(0, 2, 3, 1)
    resText = (
        torch.matmul(
            map_reshaped.reshape(1, -1, 512).float(),
            torch.transpose(text_features.float(), 1, 0),
        )
        * 100
    )
    res_softmax = resText.softmax(dim=-1)

    sotfmax_map = torch.reshape(res_softmax[0, :, 0].to(device), (map_size, map_size))
    
    #Only the similarities in the map are taken into account
    mask = top_down_map_clip_mask.squeeze()
    mask = torch.flatten(mask)
    
    #mask1 = torch.flatten(mask)
    flat_map = torch.flatten(sotfmax_map)
    map_points = (mask <=1)
    indices = map_points.nonzero()
    flat_map[indices] = 0
    sotfmax_map = torch.reshape(flat_map,(map_size, map_size))

    
    fig_map, ax_map = plt.subplots()
    ax_map.imshow(sotfmax_map.cpu())
    ax_map.invert_xaxis()
    

    max_values, max_indices = torch.topk(sotfmax_map.view(B, -1), k=TOP_K_MATCHES)
    max_rows = max_indices // top_down_map_clip_feature.shape[3]
    max_cols = max_indices % top_down_map_clip_feature.shape[3]

    GOAL_X, GOAL_Y = max_cols[0, 0].cpu().numpy(), max_rows[0, 0].cpu().numpy()
    return GOAL_X, GOAL_Y, sotfmax_map

#get new goal after have seen the previous goal (multi search)
def get_goal_map_coordinate_not_seen(TEXT_SEARCH=None, houseWords=None, top_down_map_clip_feature=None, lernr_mask=None ,device=None, model=None, seen_point=None, map_coords=None, softmax_map=None, erase=500, th=0.6):
    
    found = True

    TOP_K_MATCHES = 100
    B = top_down_map_clip_feature.shape[0]

    if softmax_map is None:
        softmax_map=get_soft_map(TEXT_SEARCH=TEXT_SEARCH, houseWords=houseWords, top_down_map_clip_feature=top_down_map_clip_feature, lernr_mask=lernr_mask, clip_model=model, device=device, map_size=top_down_map_clip_feature.shape[-1])
        
    #first time is none
    if seen_point is not None:
        goal = torch.tensor([int(seen_point[1]), int(seen_point[0])]).to(device)

        dist = (map_coords-goal).pow(2).sum(1).sqrt()

        _, ind = torch.sort(dist)
        #deleting area 
        closest = ind[:erase]
        zeros = map_coords[closest]
        softmax_map[zeros[:,0], zeros[:,1]] = 0

    if softmax_map.max()<=th:
        found = False

    
    fig_map, ax_map = plt.subplots()
    ax_map.imshow(softmax_map.cpu())
    ax_map.invert_xaxis()
    #fig_map.savefig("images/softmax_map.png")

    max_values, max_indices = torch.topk(softmax_map.view(B, -1), k=TOP_K_MATCHES)
    max_rows = max_indices // top_down_map_clip_feature.shape[3]
    max_cols = max_indices % top_down_map_clip_feature.shape[3]

    GOAL_X, GOAL_Y = max_cols[0, 0].cpu().numpy(), max_rows[0, 0].cpu().numpy()

    return GOAL_X, GOAL_Y, found, softmax_map

#getting the true rotation pose for the goal with clip
def get_true_rot(word=None, position=None, sim=None, device=None, model=None, preprocess=None, orig_Rt=None):
    
    ang = np.arange (0,2 * np.pi,0.25)
    best_angle = []
    res = []
    for angl in ang:
        rotation = q.from_euler_angles([0., angl, 0.])
        Rt = np.eye(4)
        Rt[:3, 3] = position
        Rt[:3, :3] = q.as_rotation_matrix(rotation)
        Rt = np.linalg.inv(Rt)
        Rt = Rt @ np.linalg.inv(orig_Rt)
        Rt = torch.from_numpy(Rt).unsqueeze(0).float().to(device)
        best_angle.append(Rt)

        obs = sim.get_observations_at(position, rotation)
        
        image = preprocess(Image.fromarray(obs['rgb'])).unsqueeze(0).to(device)
        text = clip.tokenize(word).to(device)


        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            # normalized features
            image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)
            text_features_norm = text_features / text_features.norm(dim=1, keepdim=True)

            import torch.nn.functional as F
            #cosine similarity
            res.append(F.cosine_similarity(image_features_norm, text_features_norm)*100)
            
    res_tens = torch.tensor(res)
    _, ind = res_tens.sort(descending=True)
    return best_angle[ind[0]]

#getting the goal points of the single search
def get_goal_points_single(query_list=None, houseWords=None, clip_model=None, top_down_map_clip_feature=None, lernr_mask=None, device=None, orig_Rt=None, sim=None, map_coords=None):
    goal_pos=[]
    goal_points=[]
    heat_maps = []
    for query_word in query_list:
        #getting the map cell coordinate of the goal
        GOAL_X, GOAL_Y, softmax_map = get_goal_map_coordinate(TEXT_SEARCH=query_word, houseWords=houseWords, model=clip_model, top_down_map_clip_feature=top_down_map_clip_feature, top_down_map_clip_mask=lernr_mask, device=device, map_size=top_down_map_clip_feature.shape[-1])
        #position of the nearest cell navigable
        pred_position_, _ = get_goal_cell(target_x=GOAL_X, target_y=GOAL_Y, orig_Rt=orig_Rt, map_coords=map_coords, sim=sim,device=device, map_size=top_down_map_clip_feature.shape[-1])
        goal_points.append(pred_position_)
        goal_pos.append([GOAL_X, GOAL_Y])
        heat_maps.append(softmax_map)
    return goal_points, goal_pos, heat_maps

#getting the goal points of the multi search
def get_goal_points_multi(TEXT_SEARCH='', houseWords=None, clip_model=None, top_down_map_clip_feature=None, lernr_mask=None, device=None, orig_Rt=None, sim=None, map_coords=None, erase=500, th=0.6):

    goal_points=[]
    goal_pos=[]
    found=True
    soft_map = None
    seen_point = None
    
    softmax_map = None
    while found==True:
        
        GOAL_X, GOAL_Y, found, soft_map = get_goal_map_coordinate_not_seen(TEXT_SEARCH=TEXT_SEARCH, houseWords=houseWords, top_down_map_clip_feature=top_down_map_clip_feature, 
        lernr_mask=lernr_mask, device=device,
        model=clip_model, map_coords=map_coords, seen_point=seen_point, softmax_map=soft_map, erase=erase, th=th)
        
        if softmax_map is None:
            softmax_map=torch.clone(soft_map)
        if found==True:
            pred_position_, _ = get_goal_cell(target_x=GOAL_X, target_y=GOAL_Y, orig_Rt=orig_Rt, map_coords=map_coords, sim=sim,device=device, map_size=top_down_map_clip_feature.shape[-1])
            goal_points.append(pred_position_)
            seen_point = [GOAL_X, GOAL_Y]
            goal_pos.append([GOAL_X, GOAL_Y])

    return goal_points, goal_pos, softmax_map

#saving resulting heat maps
def save_heat_map(softmax_maps=None, goal_pos=None, query_list=None, houseWords=None, multi_search=None):
    if not multi_search:
        q_count=0
        for softmax_map in softmax_maps:
            fig = plt.figure()
            plt.imshow(softmax_map.cpu(),aspect='auto')
            
            
            plt.scatter(goal_pos[q_count][0], goal_pos[q_count][1], marker='*', color='red', s=300)
            
            
            plt.title("Query: "+ query_list[q_count] + "\n negative prompt: " + houseWords)
            plt.axis('off')
            plt.gca().invert_xaxis()
            
            
            fig.savefig(f"./heat_maps/single_search/{query_list[q_count]}_heat_map.jpg",bbox_inches='tight')
            
            q_count+=1
    else:
        fig = plt.figure()
        plt.imshow(softmax_maps.cpu(),aspect='auto')  
       
        for goal in goal_pos:
            plt.scatter(goal[0], goal[1], marker='*', color='red', s=300)
        plt.title("query: "+ query_list[0] + "\n negative Prompt: " + houseWords)
        plt.axis('off')
        plt.gca().invert_xaxis()
        
        
        fig.savefig(f"./heat_maps/multi_search/{query_list[0]}_heat_map.jpg",bbox_inches='tight')
    print("Heat map saved")
    return None