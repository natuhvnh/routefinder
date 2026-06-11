"""
Convert real-world VRP data into TensorDict format for RouteFinder.
"""

import numpy as np
import torch
from tensordict import TensorDict


def real_world_to_tensordict(
    coordinates: np.ndarray,
    distance_matrix: np.ndarray,
    time_windows: np.ndarray,
    service_times: np.ndarray = None,
    demands: np.ndarray = None,
    backhauls: np.ndarray = None,
    vehicle_capacity: float = 100.0,
    distance_limit: float = float("inf"),
    open_route: bool = False,
    num_vehicles: int = None,
    depot_idx: int = 0,
) -> TensorDict:
    """
    Convert real-world VRP instance into TensorDict format.

    This function assumes that `distance_matrix`, `time_windows`, and `service_times`
    have already been normalized using `normalize_real_instance()`.

    Parameters
    ----------
    coordinates : np.ndarray
        shape (n_locations, 2) with [x, y] coordinates. First row should be depot.
    distance_matrix : np.ndarray
        **normalized** distance matrix, shape (n_locations, n_locations).
    time_windows : np.ndarray
        **normalized** time windows, shape (n_locations, 2) with [early, late] times.
    service_times : np.ndarray, optional
        **normalized** service times, shape (n_locations,).
        Default: all zeros.
    demands : np.ndarray, optional
        linehaul (delivery) demands, shape (n_locations,).
        Default: uniform [1, 10).
    backhauls : np.ndarray, optional
        backhaul (pickup) demands, shape (n_locations,).
        Default: all zeros (no backhauls).
    vehicle_capacity : float, default 100.0
        capacity of each vehicle (real units, will be scaled).
    distance_limit : float, default inf
        maximum distance per vehicle (normalized units).
    open_route : bool, default False
        if True, vehicles don't need to return to depot.
    num_vehicles : int, optional
        number of vehicles. If None, automatically inferred from data.
    depot_idx : int, default 0
        index of depot in coordinates/distance_matrix.

    Returns
    -------
    td : TensorDict
        A TensorDict with batch_size=(1,) and all required fields for MTVRPEnv.
        Fields: locs, demand_linehaul, demand_backhaul, time_windows, service_time,
                vehicle_capacity, capacity_original, distance_limit, open_route,
                backhaul_class, speed.
    """

    # Ensure numpy arrays
    coordinates = np.asarray(coordinates, dtype=np.float32)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float32)
    time_windows = np.asarray(time_windows, dtype=np.float32)

    n_locations = len(coordinates)

    # Validate that depot is at index 0
    if depot_idx != 0:
        # Reorder so depot is first
        depot_coords = coordinates[depot_idx].copy()
        coordinates = np.vstack([depot_coords, np.delete(coordinates, depot_idx, axis=0)])
        # Reorder distance matrix
        idx_order = [depot_idx] + [i for i in range(n_locations) if i != depot_idx]
        distance_matrix = distance_matrix[np.ix_(idx_order, idx_order)]
        time_windows = time_windows[idx_order]

    # Service times
    if service_times is None:
        service_times = np.zeros(n_locations, dtype=np.float32)
    else:
        service_times = np.asarray(service_times, dtype=np.float32)

    # Demands (linehaul)
    if demands is None:
        demands = np.random.uniform(1, 10, size=n_locations - 1).astype(np.float32)
        demands = np.concatenate([[0], demands])  # depot has 0 demand
    else:
        demands = np.asarray(demands, dtype=np.float32)

    # Backhauls (pickup)
    if backhauls is None:
        backhauls = np.zeros(n_locations, dtype=np.float32)
    else:
        backhauls = np.asarray(backhauls, dtype=np.float32)

    # Infer number of vehicles
    if num_vehicles is None:
        # Simple heuristic: num_vehicles = ceil(total_demand / capacity)
        total_demand = np.sum(demands)
        num_vehicles = max(1, int(np.ceil(total_demand / vehicle_capacity)))

    # Convert to torch tensors
    locs = torch.from_numpy(coordinates).float()  # [n_locations, 2]
    dist_matrix = torch.from_numpy(distance_matrix).float()  # [n_locations, n_locations]
    demand_linehaul = torch.from_numpy(demands[1:]).float()  # [n_locations-1] (exclude depot)
    demand_backhaul = torch.from_numpy(backhauls[1:]).float()  # [n_locations-1]
    tw = torch.from_numpy(time_windows).float()  # [n_locations, 2]
    st = torch.from_numpy(service_times).float()  # [n_locations]

    # Create TensorDict with batch_size = (1,)
    td = TensorDict(
        {
            "locs": locs.unsqueeze(0),  # [1, n_locations, 2]
            "demand_linehaul": demand_linehaul.unsqueeze(0),  # [1, n_locations-1]
            "demand_backhaul": demand_backhaul.unsqueeze(0),  # [1, n_locations-1]
            "time_windows": tw.unsqueeze(0),  # [1, n_locations, 2]
            "service_time": st.unsqueeze(0),  # [1, n_locations]
            "vehicle_capacity": torch.tensor([[vehicle_capacity]], dtype=torch.float32),  # [1, 1]
            "capacity_original": torch.tensor([[vehicle_capacity]], dtype=torch.float32),  # [1, 1]
            "distance_limit": torch.tensor([[distance_limit]], dtype=torch.float32),  # [1, 1]
            "open_route": torch.tensor([[open_route]], dtype=torch.bool),  # [1, 1]
            "backhaul_class": torch.tensor([[1]], dtype=torch.int32),  # [1, 1] (1=classic, 2=mixed)
            "speed": torch.tensor([[1.0]], dtype=torch.float32),  # [1, 1] (implicit speed in normalized units)
        },
        batch_size=(1,),
    )

    return td


# Example usage
if __name__ == "__main__":
    # Example: 5 locations (1 depot + 4 customers)
    coords = np.array([
        [0, 0],      # depot
        [1, 0.5],    # customer 1
        [0.5, 1],    # customer 2
        [1, 1],      # customer 3
        [0.2, 0.8],  # customer 4
    ], dtype=np.float32)

    # Euclidean distances (already normalized)
    dist = np.array([
        [0, 1.0, 1.1, 1.42, 0.83],
        [1.0, 0, 0.7, 0.5, 1.0],
        [1.1, 0.7, 0, 0.7, 0.3],
        [1.42, 0.5, 0.7, 0, 1.0],
        [0.83, 1.0, 0.3, 1.0, 0],
    ], dtype=np.float32)

    # Time windows (normalized: max ~1.414)
    tw = np.array([
        [0, 10],      # depot: always open
        [1, 8],       # customer 1
        [2, 9],       # customer 2
        [1, 10],      # customer 3
        [2, 8],       # customer 4
    ], dtype=np.float32)

    # Service times (normalized)
    st = np.array([0, 0.5, 0.3, 0.4, 0.2], dtype=np.float32)

    # Create TensorDict
    td = real_world_to_tensordict(
        coordinates=coords,
        distance_matrix=dist,
        time_windows=tw,
        service_times=st,
        demands=np.array([0, 2, 3, 1, 2]),
        vehicle_capacity=10.0,
        distance_limit=5.0,
    )

    print("TensorDict created:")
    print(td)
    print("\nBatch size:", td.batch_size)
    print("Locations shape:", td["locs"].shape)
    print("Time windows shape:", td["time_windows"].shape)
