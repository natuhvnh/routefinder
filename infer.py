import pandas as pd
import numpy as np
from tensordict import TensorDict
import math
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
# from routefinder.models.baselines.mvmoe.model import MVMoE
# from routefinder.models.baselines.mtpomo.model import MTPOMO
from routefinder.envs.mtvrp import MTVRPEnv, MTVRPGenerator
import time
import argparse
import pickle
from io import BytesIO
from rl4co.utils.ops import unbatchify, gather_by_index
import data_process
from packing import items_packing
import opt_utils
from opt_cf import env_config as opt_config

def parse_args():
    parser = argparse.ArgumentParser(description="Data prep job")
    parser.add_argument("--input_path", type=str, required=True, help="Input path")
    return parser.parse_args()


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


def compute_schedule(
    route,                  # list of location indices (including depot)
    travel_time_matrix,     # 2D list or numpy array
    service_durations,      # list indexed by location
    time_windows,           # list of (start, end)
    start_time=0
):
    schedule = []
    location_arrival_time = []
    current_time = start_time
    for i in range(len(route)):
        loc = route[i]
        if i == 0:
            # Depot start
            arrival = current_time
        else:
            prev = route[i - 1]
            travel_time = travel_time_matrix[prev][loc]
            arrival = current_time + travel_time
        tw_start, tw_end = time_windows[loc]
        # Waiting if early
        wait = max(0, tw_start - arrival)
        # Start service
        start_service = arrival + wait
        # Departure time
        service_duration = service_durations if loc != 0 else 0
        departure = start_service + service_duration
        # Time warp (late arrival)
        time_warp = max(0, arrival - tw_end)
        schedule.append({
            "location": loc,
            "arrival": arrival,
            "wait": wait,
            "start_service": start_service,
            "departure": departure,
            "time_warp": time_warp,
        })
        location_arrival_time.append(arrival)
        # Move time forward
        current_time = departure
    return schedule, location_arrival_time


def get_route_delivery(
    df,
    equipment,
    routes,
    order_number_id,
    num_stack,
    unit_type,
    distances,
    durations,
    pallets,
    weights,
    volumes,
    time_windows,
):
    start_time = time.perf_counter()
    loc_service_time = opt_config.client_service_duration
    route_delivery = []
    df_dict = df.set_index("id").to_dict(orient="index")
    #
    for index, route in enumerate(routes):
        _, location_arrival_time = compute_schedule(
            route,
            durations,
            opt_config.client_service_duration,
            time_windows,
            start_time=0,
        )
        print(f"Loading check for route {index}.")
        order_id = []
        pallet_delivery = 0
        weight_delivery = 0
        volume_delivery = 0
        duration = 0
        distance = 0
        # Accumulate Order IDs
        for v in route:
            if v != 0:
                orderids = order_number_id[v - 1][1]
                order_id.extend(orderids)
        loading_items = {"num_stack": num_stack, "unit_type": unit_type, "items": []}
        for i in order_id:
            if i in df_dict:
                data = df_dict[i]
                loading_items["items"].append(
                    {
                        "id": i,
                        "height": data["height"],
                        "width": data["width"],
                        "length": data["length"],
                        "weight": data["weight"],
                        "quantity": data["quantity"],
                        "stack_on_top": data["stack_on_top"],
                        "stack_on_other": data["stack_on_other"],
                    }
                )

        # Calculate Route Metrics
        for i in range(len(route) - 1):
            distance += distances[route[i]][route[i + 1]]
            duration += durations[route[i]][route[i + 1]] + loc_service_time
            pallet_delivery += pallets[route[i + 1]]
            weight_delivery += weights[route[i + 1]]
            volume_delivery += volumes[route[i + 1]]

        # Determine order column
        if unit_type == "Pallet":
            order_column = "palletSpacesUK"
        elif unit_type == "Weight":
            order_column = "maximumPayloadKg"
        elif unit_type == "Volume":
            order_column = "volume"
        else:
            order_column = "volume"
        sorted_equipment = equipment.sort_values(by=[order_column], ascending=False)
        ldm = float("inf")
        for i, row in enumerate(sorted_equipment.itertuples()):
            v_name = row.name
            v_weight = row.maximumPayloadKg
            v_pallet = row.palletSpacesUK
            v_volume = row.volume
            v_whd = [
                row.internalWidthMillimeter,
                row.internalHeightMillimeter,
                row.internalLengthMillimeter,
            ]
            if (
                pallet_delivery <= v_pallet
                and weight_delivery <= v_weight
                and volume_delivery <= v_volume
            ):
                best_ldm, init_ldm = items_packing(
                    loading_items["items"],
                    v_whd,
                    v_weight,
                    num_stack,
                    unit_type,
                    visualize=False,
                )
                if i == 0:
                    vehicle_weight_utilization = round(weight_delivery / v_weight, 3)
                    vehicle_volume_utilization = round(volume_delivery / v_volume, 3)
                    vehicle_pallet_utilization = round(pallet_delivery / v_pallet, 3)
                    ldm = round(best_ldm / 1000, 2)
                    vehicle_name = v_name
                else:
                    if not math.isinf(best_ldm):
                        vehicle_weight_utilization = round(weight_delivery / v_weight, 3)
                        vehicle_volume_utilization = round(volume_delivery / v_volume, 3)
                        vehicle_pallet_utilization = round(pallet_delivery / v_pallet, 3)
                        ldm = round(best_ldm / 1000, 2)
                        vehicle_name = v_name
            else:
                # print(f"Can not loading for route {index}, equipment {v_name}")
                break
        #
        route_delivery.append(
            {
                "weight_delivery": weight_delivery,
                "volume_delivery": round(volume_delivery, 2),
                "pallet_delivery": pallet_delivery,
                "vehicle_weight_utilization": vehicle_weight_utilization,
                "vehicle_volume_utilization": vehicle_volume_utilization,
                "vehicle_pallet_utilization": vehicle_pallet_utilization,
                "ldm": ldm,
                "distance": distance,
                "location_arrival_time": location_arrival_time,
                "service_time": location_arrival_time[-1] - location_arrival_time[0],
                "vehicle_name": vehicle_name,
            }
        )
    loading_runtime = time.perf_counter() - start_time
    print(f"\nTotal Loading Check Time: {loading_runtime:.4f}s")
    return route_delivery


def get_route_output(
    req_id,
    directory_ref,
    col_date,
    routes,
    locations,
    route_delivery,
    variant,
    dest_names,
    order_number_id,
    address_guids,
    distances,
    durations,
    full_load_route,
    runtime,
):
    all_routes = []
    for i, route in enumerate(routes):
        visits = []
        visit_index = []
        orderids = []
        all_order_number = []
        all_order_id = []
        matrix_component = []
        last_visit_name = ""
        #
        for cust in route[1:]:
            if cust != 0:
                visit = {}
                visit_name = dest_names[cust].split("_")[0]
                directoryReference = dest_names[cust].split("_")[1]
                order_number = order_number_id[cust - 1][0]
                orderids = order_number_id[cust - 1][1]
                visit_index.append(cust)
                all_order_number.extend(order_number)
                all_order_id.extend(orderids)
                if visit_name == last_visit_name:
                    visits[-1]["order_number"].append(order_number)
                    visits[-1]["orderIds"].extend(orderids)
                else:
                    visit["visit_name"] = visit_name
                    visit["directoryReference"] = directoryReference
                    visit["orderIds"] = orderids
                    visit["order_number"] = order_number
                    visits.append(visit)
                last_visit_name = visit_name
        source_info = {
            "visit_name": dest_names[0].split("_")[0],
            "directoryReference": dest_names[0].split("_")[1],
            "orderIds": all_order_id,
            "order_number": all_order_number,
        }
        visits.insert(0, source_info)
        visit_index.insert(0, 0)
        if "o" not in variant:
            visits.append(source_info)
            visit_index.append(0)
        #
        for j in range(len(visit_index) - 1):
            component = {
                "sourceId": address_guids[visit_index[j]],
                "destinationId": address_guids[visit_index[j + 1]],
                "drivingDistanceInMeters": int(
                    distances[visit_index[j]][visit_index[j + 1]] * 1000
                ),
                "travelTimeInSeconds": int(
                    durations[visit_index[j]][visit_index[j + 1]] * 60
                ),
            }
            matrix_component.append(component)
        #
        r_visit = {
            "route_id": i,
            "visits": visits,
            "distanceMatrix": matrix_component,
            "note": "routefinder",
        }
        r_delivery = route_delivery[i]
        route_detail = r_visit | r_delivery
        all_routes.append(route_detail)
    if len(full_load_route) > 0:
        all_routes.extend(full_load_route)

    # save actions
    buffer = BytesIO()
    locations.to_parquet(buffer, index=False)
    opt_utils.upload_file_to_blob(
        opt_config.connection_string,
        opt_config.output_container_name,
        f"locations/{req_id}.parquet",
        buffer.getvalue(),
    )

    opt_utils.upload_file_to_blob(
        opt_config.connection_string,
        opt_config.output_container_name,
        f"routes/{req_id}.pkl",
        pickle.dumps(routes),
    )
    output = {}
    output["status"] = True
    output["id"] = req_id
    output["directory_reference"] = directory_ref
    output["col_date"] = col_date
    output["routes"] = all_routes
    output["run_time"] = runtime
    return output


if __name__ == "__main__":
    start_time = time.perf_counter()
    # args = parse_args()
    # input_path = args.input_path
    input_path = "c468df24-0163-4125-907e-68d32291418d"
    query = f"""
        SELECT *
        FROM c
        WHERE c.req_id = "{input_path}"
    """
    data = opt_utils.query_cosmos("orders", "hgs-input", query)
    data = data[0]
    #
    variant = data["variant"]
    biggest_equipment = data["biggest_equipment"]
    unit_type = data["unit_type"]
    num_stack = data["num_stack"]
    multi_visit_penalty = data["multi_visit_penalty"]
    req_id = data["req_id"]
    col_date = data["col_date"]
    directory_ref = data["directory_reference"]
    #
    equipment = pd.DataFrame(data["equipment_list"])
    equipment["floor_area"] = (
        equipment["internalLengthMillimeter"]
        / 1000
        * equipment["internalWidthMillimeter"]
        / 1000
        * 0.95
    )  # 5% reserved for loading
    if unit_type == "Volume":
        equipment["palletSpacesUK"] = equipment[
            "floor_area"
        ]  # for boxes, optimize based on floor area instead of pallets
        equipment["volume"] = equipment["volume"] * 0.9  # 10% reserved for loading
    #
    df = pd.DataFrame(data["order"])
    if variant == "vrpbtw":
        df = data_process.process_backhaul_order(df)
    else:
        df = df.assign(backhaul=1)
    df = data_process.process_order(df)
    (
        locations,
        order_number_id,
        full_load_route,
        dest_names,
        coordinates,
        weights,
        volumes,
        pallets,
        distances,
        time_windows,
        durations,
        address_guids,
        speed,
    ) = data_process.process_account(
        df,
        num_stack=num_stack,
        equipment=equipment,
        biggest_equipment=biggest_equipment,
        unit_type=unit_type,
        variant=variant,
        req_id=req_id,
        directory_ref=directory_ref,
    )
    #
    if unit_type == "Pallet":
        capacity = equipment[equipment.name == biggest_equipment][
            "palletSpacesUK"
        ].values[0]
    elif unit_type == "Volume":
        capacity = equipment[equipment.name == biggest_equipment]["volume"].values[0]
    scale_factor = 0.2  # 3, 0.2
    min_coord = np.array(coordinates).min(axis=0)
    max_coord = np.array(coordinates).max(axis=0)
    coord_range = max_coord - min_coord
    distance_scaler = coord_range.max()
    coordinates_scaled = (np.array(coordinates) - min_coord) / distance_scaler
    coordinates_scaled = np.expand_dims(coordinates_scaled, axis=0)
    # coordinates = np.expand_dims(coordinates, axis=0)
    #
    max_time_window = 1440
    time_scaler = max_time_window / scale_factor  # 1440/3
    service_time = np.array([opt_config.client_service_duration] * (len(weights) - 1))
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
        node_demand = torch.Tensor(node_demand.reshape((1, -1))) / (capacity + 0.1)  # [1, n]
    speed = speed / 60 * time_scaler
    distances_limit = 900
    distance_matrix_scaler = 3
    distances_scaled = np.array(distances) * distance_matrix_scaler / distances_limit
    durations_scaled = np.array(durations) / time_scaler
    is_open = True if "o" in variant else False
    td_instance = TensorDict(
        {
            "locs": torch.from_numpy(coordinates_scaled[0]).float(),
            "demand_linehaul": node_demand[0],
            "capacity_original": torch.tensor([capacity]),
            "service_time": service_time[0],
            "speed": torch.tensor([speed]).float(),
            "time_windows": tw,
            "distance_limit": torch.tensor([distance_matrix_scaler]).float(),
            "distance_matrix": torch.from_numpy(distances_scaled).float(),
            "duration_matrix": torch.from_numpy(durations_scaled).float(),
            "open_route": torch.tensor([is_open], dtype=torch.bool),
        },
        batch_size=[],
    )[None]
    #
    if variant == "ovrptw":
        variant_preset = "ovrpltw"
    elif variant == "vrptw":
        variant_preset = "vrpltw"
    device = "cpu"
    PATH = "checkpoints/100/rf-transformer.ckpt"
    model = RouteFinderBase.load_from_checkpoint(
        PATH, map_location=device, weights_only=False
    )
    model.eval().to(device)
    policy = model.policy
    policy = policy.to(device).eval()
    generator = MTVRPGenerator(num_loc=100, variant_preset=variant_preset)
    env = MTVRPEnv(generator, check_solution=True).to(device)  # Add .to(device) here
    td_instance = td_instance.to(device)
    td_reset = env.reset(td_instance).to(device)
    actions = evaluate(model, td_reset.clone(), env)["best_aug_actions"]
    routes = extract_routes_from_actions(actions, is_open=is_open)
    #
    route_delivery = get_route_delivery(
        df,
        equipment,
        routes[0],
        order_number_id,
        num_stack,
        unit_type,
        distances,
        durations,
        pallets,
        weights,
        volumes,
        time_windows,
    )
    runtime = time.perf_counter() - start_time
    output = get_route_output(
        req_id,
        directory_ref,
        col_date,
        routes[0],
        locations,
        route_delivery,
        variant,
        dest_names,
        order_number_id,
        address_guids,
        distances,
        durations,
        full_load_route,
        runtime,
    )
    print(f"\nTotal Run Time: {runtime}s")
    opt_utils.cosmos_upsert_data('hgs-output', 'route', output)
# opt_utils.build_route(output)

