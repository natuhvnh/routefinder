import pandas as pd
import numpy as np
from tensordict import TensorDict
from math import ceil
import torch
import torchrl.data.tensor_specs
from torch.distributions.uniform import Uniform

torchrl.data.tensor_specs.CompositeSpec = torchrl.data.tensor_specs.Composite
torchrl.data.tensor_specs.BoundedTensorSpec = torchrl.data.tensor_specs.Bounded
torchrl.data.tensor_specs.UnboundedDiscreteTensorSpec = (
    torchrl.data.tensor_specs.UnboundedDiscrete
)
torchrl.data.tensor_specs.UnboundedContinuousTensorSpec = (
    torchrl.data.tensor_specs.UnboundedContinuous
)
from routefinder.models.model import RouteFinderBase, RouteFinderMoE
from routefinder.models.baselines.mvmoe.model import MVMoE
from routefinder.models.baselines.mtpomo.model import MTPOMO
from routefinder.envs.mtvrp import MTVRPEnv, MTVRPGenerator
import json
import time
import pickle
from datetime import datetime
import statistics
import requests
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import cycle
from rl4co.utils.ops import unbatchify, gather_by_index
import data_process


#
def evaluate(
    model,
    td,
    env,
    num_augment=8,
    num_starts=None,
):

    with torch.inference_mode():
        with torch.amp.autocast("cpu"):
            n_start = env.get_num_starts(td) if num_starts is None else num_starts

            if num_augment > 1:
                td = model.augment(td)

            # Evaluate policy
            out = model.policy(
                td, env, phase="test", num_starts=n_start, return_actions=True
            )

            # Unbatchify reward to [batch_size, num_augment, num_starts].
            reward = unbatchify(out["reward"], (num_augment, n_start))

            if n_start > 1:
                # max multi-start reward
                max_reward, max_idxs = reward.max(dim=-1)
                out.update({"max_reward": max_reward})

                if out.get("actions", None) is not None:
                    # Reshape batch to [batch_size, num_augment, num_starts, ...]
                    actions = unbatchify(out["actions"], (num_augment, n_start))
                    out.update(
                        {
                            "best_multistart_actions": gather_by_index(
                                actions, max_idxs, dim=max_idxs.dim()
                            )
                        }
                    )
                    out["actions"] = actions

            # Get augmentation score only during inference
            if num_augment > 1:
                # If multistart is enabled, we use the best multistart rewards
                reward_ = max_reward if n_start > 1 else reward
                max_aug_reward, max_idxs = reward_.max(dim=1)
                out.update({"max_aug_reward": max_aug_reward})

                if out.get("actions", None) is not None:
                    actions_ = (
                        out["best_multistart_actions"] if n_start > 1 else out["actions"]
                    )
                    out.update({"best_aug_actions": gather_by_index(actions_, max_idxs)})

            return out


#
# def extract_routes_from_actions(actions, drop_empty_routes=True):
#     """
#     Parse giant-tour actions into per-route lists. Robust to both conventions:
#     - depot markers placed before each route (0, a, b, 0, c, d, 0, ...)
#     - depot markers placed after each route (a, b, 0, c, d, 0, ...)
#     Args:
#         actions: 1D or 2D Tensor/list. If 2D, shape [batch, seq_len].
#         drop_empty_routes: if True, omit routes that contain no customers (i.e., [0]).
#     Returns:
#         If input was 1D: list[list[int]] of routes (each starts with 0 followed by clients).
#         If input was 2D: list of batches -> list[list[list[int]]].
#     """
#     single = False
#     if isinstance(actions, torch.Tensor):
#         actions = actions.cpu()
#     if not isinstance(actions, (list, tuple)) and not hasattr(actions, "__iter__"):
#         raise TypeError("actions must be a tensor or an iterable")
#     # Normalize to list-of-lists
#     if isinstance(actions, torch.Tensor) and actions.dim() == 1:
#         actions = actions.unsqueeze(0)
#         single = True
#     elif isinstance(actions, torch.Tensor) and actions.dim() == 2:
#         pass
#     elif isinstance(actions, (list, tuple)) and (
#         len(actions) and not isinstance(actions[0], (list, tuple))
#     ):
#         # single sequence provided as list -> wrap
#         actions = [list(actions)]
#         single = True
#     elif isinstance(actions, (list, tuple)):
#         # assume list of sequences
#         pass
#     else:
#         raise TypeError("Unsupported actions format")
#     routes_batch = []
#     for seq in actions:
#         # get plain python list of ints
#         if isinstance(seq, torch.Tensor):
#             seq = seq.tolist()
#         seq = [int(x) for x in seq]
#         # Split on zeros: segments are sequences between zeros.
#         segments = []
#         curr = []
#         for val in seq:
#             if val == 0:
#                 # zero acts as separator/end-of-route marker
#                 segments.append(curr)
#                 curr = []
#             else:
#                 curr.append(val)
#         # Add trailing segment (if sequence did not end with 0)
#         if curr or not segments:
#             segments.append(curr)
#         # Convert segments -> routes that start with depot 0
#         routes = []
#         for seg in segments:
#             if len(seg) == 0:
#                 route = [0]  # empty route (no customers)
#             else:
#                 route = [0] + seg
#             if drop_empty_routes and route == [0]:
#                 continue
#             routes.append(route)
#         routes_batch.append(routes)
#     routes = routes_batch[0] if single else routes_batch
#     for i in routes:
#         for r in i:
#             r.append(0)
#     return routes


#
def extract_routes_from_actions(actions, drop_empty_routes=True, is_open=False):
    """
    Parse giant-tour actions into per-route lists. 
    
    Args:
        actions: 1D or 2D Tensor/list. If 2D, shape [batch, seq_len].
        drop_empty_routes: if True, omit routes that contain no customers (i.e., [0] or [0, 0]).
        is_open: if True, routes end at the last customer (e.g., [0, 1, 2]). 
                 if False, routes return to the depot (e.g., [0, 1, 2, 0]).
    Returns:
        If input was 1D: list[list[int]] of routes.
        If input was 2D: list of batches -> list[list[list[int]]].
    """
    single = False
    if isinstance(actions, torch.Tensor):
        actions = actions.cpu()
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
            single = True
    elif isinstance(actions, (list, tuple)):
        if len(actions) > 0 and not isinstance(actions[0], (list, tuple)):
            actions = [list(actions)]
            single = True
    else:
        raise TypeError("actions must be a tensor or an iterable")
    routes_batch = []
    for seq in actions:
        if isinstance(seq, torch.Tensor):
            seq = seq.tolist()
        seq = [int(x) for x in seq]
        # Split on zeros
        segments = []
        curr = []
        for val in seq:
            if val == 0:
                segments.append(curr)
                curr = []
            else:
                curr.append(val)
        if curr:
            segments.append(curr)
        # Convert segments to routes
        routes = []
        for seg in segments:
            # Skip empty routes if requested
            if len(seg) == 0 and drop_empty_routes:
                continue
            # Start at depot
            route = [0] + seg
            # If it's a closed route, return to depot
            if not is_open:
                route.append(0)
            routes.append(route)
        routes_batch.append(routes)
    return routes_batch[0] if single else routes_batch


#
if __name__ == "__main__":
    # data_path = "3t_data/b2d8731b-4d6f-4e1d-80cf-c46ef29908e9.json" # 100
    # data_path = "3t_data/184d181a-0630-49a7-b8c5-2f94f634ee3e.json" # 76
    data_path = "3t_data/208244c0-8aee-4bc0-971e-ae8909428301.json" # 46
    with open(data_path, "r") as file:
        data = json.load(file)
    variant = data["variant"]
    biggest_equipment = data["biggest_equipment"]
    unit_type = data["unit_type"]
    num_stack = data["num_stack"]
    req_id = data["req_id"]
    #
    equipment = pd.DataFrame(data["equipment_list"])
    equipment = equipment[equipment.name == biggest_equipment]
    if unit_type == 'Pallet':
        capacity = equipment[equipment.name == biggest_equipment]['palletSpacesUK'].values[0]
    elif unit_type == "Volume":
        capacity = equipment[equipment.name == biggest_equipment]['volume'].values[0]
    #
    df = pd.DataFrame(data["order"])
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)
    df["dest_name"] = df["dest_name"] + "_" + df["dest_id"] + "_" + df["order_number"]
    df["collection_date"] = pd.to_datetime(df["collection_date"], format="%Y-%m-%d")
    df["volume"] = (
        df["quantity"] * df["height"] / 1000 * df["length"] / 1000 * df["width"] / 1000
    )
    df["stackable"] = df.apply(
        lambda row: (
            1 if (row["stack_on_top"] == "Y" and row["stack_on_other"] == "Y") else 0
        ),
        axis=1,
    )
    (
        locations,
        order_id,
        full_load_route,
        dest_names,
        coordinates,
        weights,
        volumes,
        pallets,
        distances,
        time_windows,
        durations,
        speed,
    ) = data_process.process_account(
        df,
        num_stack=num_stack,
        equipment=equipment,
        biggest_equipment=biggest_equipment,
        unit_type=unit_type,
        variant=variant,
        req_id=req_id,
    )
    #
    scale_factor = 0.2  # 3, 0.2
    min_coord = np.array(coordinates).min(axis=0)
    max_coord = np.array(coordinates).max(axis=0)
    coord_range = max_coord - min_coord
    distance_scaler = coord_range.max()
    coordinates_scaled = (np.array(coordinates) - min_coord) / distance_scaler
    coordinates_scaled = np.expand_dims(coordinates_scaled, axis=0)
    coordinates = np.expand_dims(coordinates, axis=0)
    #
    max_time_window = 1440
    time_scaler = max_time_window / scale_factor  # 1440/3
    service_time = np.array([10] * (len(weights) - 1))
    service_time = torch.Tensor(service_time.reshape((1, -1))) / time_scaler
    tw_start = (
        torch.Tensor(
            np.array(locations["delivery_start_minutes_of_day"].tolist()).reshape((1, -1))
        )
        / time_scaler
    )  # [1, n]
    tw_end = (
        torch.Tensor(
            np.array(locations["delivery_end_minutes_of_day"].tolist()).reshape((1, -1))
        )
        / time_scaler
    )  # [1, n]
    depot_tw, depot_service_time = torch.tensor([[0.0, scale_factor]]), torch.zeros(
        (1, 1), device=service_time.device
    )
    tw = torch.stack([tw_start.squeeze(0), tw_end.squeeze(0)], dim=1)
    tw = torch.cat([depot_tw, tw], dim=0)
    service_time = torch.cat([depot_service_time, service_time], dim=1)
    if unit_type == "Volume":
        node_demand = np.array(volumes[1:])
        node_demand = torch.Tensor(node_demand.reshape((1, -1))) / capacity
    elif unit_type == "Pallet":
        node_demand = np.array(pallets[1:])
        node_demand = torch.Tensor(node_demand.reshape((1, -1))) / (capacity + 0.1) # [1, n]
    speed = speed / 60 * time_scaler
    distances = np.array(distances)
    durations = np.array(durations)
    distances_limit = 900
    distance_matrix_scaler = 3
    distances = distances * distance_matrix_scaler / distances_limit
    durations = durations / time_scaler
    td_instance = TensorDict(
        {
            "locs": torch.from_numpy(coordinates_scaled[0]).float(),
            "demand_linehaul": node_demand[0],
            "capacity_original": torch.tensor([capacity]),
            "service_time": service_time[0],
            "speed": torch.tensor([speed]).float(),
            "time_windows": tw,
            "distance_limit": torch.tensor([distance_matrix_scaler]).float(),
            "distance_matrix": torch.from_numpy(distances).float(),
            "duration_matrix": torch.from_numpy(durations).float(),
            "open_route": torch.tensor([True], dtype=torch.bool),
        },
        batch_size=[],
    )[None]
    #
    device = "cpu"
    # Choose your model
    PATH = "checkpoints/100/rf-transformer.ckpt"
    model = RouteFinderBase.load_from_checkpoint(PATH, map_location="cpu", weights_only=False)
    model.eval().to(device)
    #
    policy = model.policy
    policy = policy.to(device).eval()
    # Create env
    generator = MTVRPGenerator(num_loc=100, variant_preset="all")
    # env = MTVRPEnv(generator, check_solution=False)
    env = MTVRPEnv(generator, check_solution=True).to(device) # Add .to(device) here
    #
    td_instance = td_instance.to(device)
    td_reset = env.reset(td_instance).to(device)

    start = time.time()
    actions = evaluate(model, td_reset.clone(), env)["best_aug_actions"]
    with open('output/actions.pkl', 'wb') as f:
        pickle.dump(actions, f)
    inference_time = time.time() - start

    # Obtain reward from the environment with new locs
    # td_reset["locs"] = coordinates_scaled[0][None]  # unnormalized
    # td_reset["locs"] = torch.from_numpy(coordinates_scaled[0][None]).float().to(device)
    reward = env.get_reward(td_reset, actions)

    cost = ceil(-reward.item())
    inference_time = time.time() - start
    print(cost)
    print(inference_time)
    #
    is_open = td_instance["open_route"].item()
    routes = extract_routes_from_actions(actions, is_open=is_open)
    print(routes)
    print(len(routes[0]))
