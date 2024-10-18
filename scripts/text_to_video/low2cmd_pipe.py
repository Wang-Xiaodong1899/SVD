import json
import torch
import torch.nn as nn
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from einops import rearrange, repeat

DATAROOT = "/root/autodl-tmp/nuscenes/all"

nusc_can = NuScenesCanBus(dataroot=DATAROOT)

def process_state(speeds, steering_angles, left_signals, right_signals):
    turns = determine_turns(left_signals, right_signals)
    
    if len(turns) == 0:
        turns = determine_turns_by_steering_angle(steering_angles)
    
    if type(turns) == list:
        state = ", ".join(turns)
    else:
        # determine speed
        speed_state = determine_speed_state(speeds)
        state = turns + speed_state
        if speed_state == "wait":
            state = "wait"
    
    return state

def determine_turns(left_signals, right_signals):
    turns = []
    last_turn = None
    
    for i in range(len(left_signals)):
        current_turn = None
        if left_signals[i] == 1:
            current_turn = "turn left"
        elif right_signals[i] == 1:
            current_turn = "turn right"

        if current_turn and current_turn != last_turn:
            turns.append(f"{current_turn}")
            last_turn = current_turn

    if not turns:
        return [] # others
    return turns

def determine_turns_by_steering_angle(steering_angles):

    left_turns = sum(angle > 10 for angle in steering_angles)
    right_turns = sum(angle < -10 for angle in steering_angles)
    
    if left_turns > len(steering_angles) / 2:
        return "follow the road left, "
    elif right_turns > len(steering_angles) / 2:
        return "follow the road right, "
    else:
        return "go straight, "


def determine_speed_state(speeds):
    diffs = [speeds[i+1] - speeds[i] for i in range(len(speeds)-1)]
    if all(v == 0 for v in speeds):
        return "wait"
    elif all(v < 10 for v in speeds):
        return "drive slowly"
    elif all(v > 30 for v in speeds):
        return "drive fast"
    else:
        if all(diff > 0 for diff in diffs) or (all(diff >= 0 for diff in diffs) and diffs[-1] == 0):
            return "speed up"
        elif all(diff < 0 for diff in diffs) or (all(diff <= 0 for diff in diffs) and diffs[-1] == 0):
            if all(v > 20 for v in speeds):
                return "maintain speed"
            else:
                return "slow down"
        else:
            return "maintain speed"

scenes_dict = create_splits_scenes()

split = "train"

scenes = scenes_dict[split]

json_file = f"nusc_action_{split}.json"

data = {}

for sce in tqdm(scenes):
    try:
        veh_moni = nusc_can.get_messages(sce, 'vehicle_monitor')
    except:
        print(f"{sce} does not have any CAN bus data!")
        continue
    data[sce] = {}
    for s in range(0, len(veh_moni)-8):
        selected_veh_moni = veh_moni[s: s+8]
        speeds = [item["vehicle_speed"] for item in selected_veh_moni]
        steering_angles = [item["steering"] for item in selected_veh_moni]
        left_signals = [item["left_signal"] for item in selected_veh_moni]
        right_signals = [item["right_signal"] for item in selected_veh_moni]

        state = process_state(speeds, steering_angles, left_signals, right_signals)
        # print(state)
        # import pdb; pdb.set_trace()
        data[sce][s] = state

with open(json_file, 'w') as f:
    json.dump(data, f, indent='\t')
