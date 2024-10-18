
# ego_speed = [
#                 self.can_bus_extension.query_and_interpolate(
#                     "vehicle_monitor", scene_name, i[0]["timestamp"] if
#                     self.is_multimodal else i["timestamp"], "vehicle_speed")
#                 for i in item["segment"]
#             ]

# ego_steering = [
#                 self.can_bus_extension.query_and_interpolate(
#                     "steeranglefeedback", scene_name, i[0]["timestamp"] if
#                     self.is_multimodal else i["timestamp"], "value")
#                 for i in item["segment"]
#             ]

# ego_orient = [
#                 self.can_bus_extension.query_and_interpolate(
#                     "pose", scene_name, i[0]["timestamp"] if
#                     self.is_multimodal else i["timestamp"], "orientation")
#                 for i in item["segment"]
#             ]

import torch
import torch.nn as nn
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from einops import rearrange, repeat

DATAROOT = "/root/autodl-tmp/nuscenes/all"

nusc_can = NuScenesCanBus(dataroot=DATAROOT)

while(1):
    # sce = "scene-0094"
    # sce = "scene-0061"
    sce = input("scene: ")

    veh_moni = nusc_can.get_messages(sce, 'vehicle_monitor')

    # import pdb; pdb.set_trace()

    # ['available_distance', 'battery_level', 'brake', 'brake_switch', 
    # 'gear_position', 'left_signal', 'rear_left_rpm', 'rear_right_rpm', 
    # 'right_signal', 'steering', 'steering_speed', 'throttle', 'utime', 
    # 'vehicle_speed', 'yaw_rate']

    # scene-0061
    # left_signal
    # [0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
    # right_signal
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # scene-0095
    # gear_position (always 7 in driving, 0 in parking)
    # [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]

    # for veh_item in veh_moni:
    #     steer = veh_item['steering']
    #     speed = veh_item['vehicle_speed']
    # print(steer)
    # print(speed)
    print([item["vehicle_speed"] for item in veh_moni])
    print([item["steering"] for item in veh_moni])
    print([item["left_signal"] for item in veh_moni])
    print([item["right_signal"] for item in veh_moni])
    print([item["brake"] for item in veh_moni]) #[0, 126]
    print([item["gear_position"] for item in veh_moni])

    # define status
    # status = ["turn left", "turn right", ]
    # s, len = 0, 8
    # selected_veh_moni = veh_moni[s:, s+len]

    # output_status = []
    # for item in selected_veh_moni:
    #     if 1 in item["left_signal"] and "turn left" not in output_status:
    #         output_status.append("turn left")
    #     elif 1 in item["right_signal"] and "turn right" not in output_status:
    #         output_status.append("turn right")
        
