import pandas as pd
import numpy as np
import pickle
import time
import requests
import statistics


def chunk_list(lst, size):
    """Split list into chunks of given size."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


#
def get_ggmap_matrix_batched(points, mode="driving"):
    """
    Returns full distance and duration matrices by batching sub-requests.
    points: list of (lat, lng)
    """
    n = len(points)
    # choose batch size B, e.g. 10
    B = 10
    # initialize matrices
    distances = [[None] * n for _ in range(n)]
    durations = [[None] * n for _ in range(n)]

    # Precompute str list
    point_strs = [f"{lat},{lng}" for (lng, lat) in points]

    # For each origin batch
    speeds = []
    for i_batch, origin_idxs in enumerate(chunk_list(range(n), B)):
        origins = "|".join(point_strs[j] for j in origin_idxs)
        # For each destination batch
        for j_batch, dest_idxs in enumerate(chunk_list(range(n), B)):
            destinations = "|".join(point_strs[k] for k in dest_idxs)

            url = "https://maps.googleapis.com/maps/api/distancematrix/json"
            params = {
                "origins": origins,
                "destinations": destinations,
                "key": "",
                "mode": mode,
                # optionally departure_time etc.
            }
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "OK":
                raise RuntimeError("Distance Matrix API error: " + str(data))

            for i_rel, row in enumerate(data["rows"]):
                i = origin_idxs[i_rel]
                for j_rel, elem in enumerate(row["elements"]):
                    j = dest_idxs[j_rel]
                    if elem.get("status") == "OK":
                        distance = elem["distance"]["value"] / 1000  # kilometers
                        duration = elem["duration"]["value"]  # seconds
                        if distance > 0:
                            speed = distance / (duration / 3600)
                            speeds.append(speed)
                        distances[i][j] = distance
                        durations[i][j] = duration
                    else:
                        distances[i][j] = None
                        durations[i][j] = None

    return distances, durations, statistics.mean(speeds)


#
def get_tomtom_matrix_async(points, travelMode="car", batch_size=30):
    base_url = "https://api.tomtom.com/routing/matrix/2"
    api_key = ""
    n = len(points)
    print(f"Call Tomtom API with {n} locations")
    # Initialize the master N x N matrices
    distances = [[None] * n for _ in range(n)]
    durations = [[None] * n for _ in range(n)]

    # Prepare all destinations (we keep these constant for every batch)
    dest_locations = [{"point": {"latitude": p[1], "longitude": p[0]}} for p in points]

    # Loop through the points in batches
    for start_idx in range(0, n, batch_size):
        end_idx = min(start_idx + batch_size, n)
        print(f"\n--- Processing Batch: Origins {start_idx} to {end_idx-1} ---")

        # Prepare the subset of origins for this batch
        batch_points = points[start_idx:end_idx]
        origin_locations = [
            {"point": {"latitude": p[1], "longitude": p[0]}} for p in batch_points
        ]

        payload = {
            "origins": origin_locations,
            "destinations": dest_locations,
            "options": {"travelMode": travelMode},
        }

        # 1. SUBMIT THE JOB (POST)
        submit_url = f"{base_url}/async?key={api_key}"
        response = requests.post(submit_url, json=payload)
        response.raise_for_status()

        job_id = response.json().get("jobId")
        print(f"Job submitted. ID: {job_id}")

        # 2. POLL FOR STATUS (GET)
        status_url = f"{base_url}/async/{job_id}?key={api_key}"
        while True:
            status_resp = requests.get(status_url)
            status_data = status_resp.json()
            status = status_data.get("state")

            if status == "Completed":
                print(f"Batch {start_idx//batch_size + 1} completed!")
                break
            elif status == "Failed":
                print(f"\nBatch failed: {status_data}")
                # Depending on your needs, you might want to 'continue' or 'break' here
                break
            else:
                print(f"Status: {status}... waiting 5 seconds.", end="\r")
                time.sleep(5)

        # 3. FETCH RESULTS (GET)
        result_url = f"{base_url}/async/{job_id}/result?key={api_key}"
        result_resp = requests.get(result_url)
        result_data = result_resp.json()

        # 4. PARSE DATA into master matrix
        # Note: The 'originIndex' in the response is relative to the batch.
        # Add start_idx to map it back to the absolute index in your master list.
        for element in result_data.get("data", []):
            rel_origin_idx = element.get("originIndex")
            dest_idx = element.get("destinationIndex")

            abs_origin_idx = start_idx + rel_origin_idx

            if "routeSummary" in element:
                summary = element["routeSummary"]
                distances[abs_origin_idx][dest_idx] = (
                    summary["lengthInMeters"] / 1000
                )  # km
                durations[abs_origin_idx][dest_idx] = summary["travelTimeInSeconds"]
    total_distance = np.sum(distances)  # km
    total_time_hours = np.sum(durations) / 3600  # hours
    speed = total_distance / total_time_hours
    # Final return now happens AFTER all batches are processed
    return distances, durations, speed


#
def get_proxy_volume(g, num_stack, unit_type, equipment, biggest_equipment):
    equipment = equipment[equipment.code == biggest_equipment]
    max_height = equipment["internalHeightMillimeter"].iloc[0].item() / 1000
    #
    total_quantity = g["quantity"].sum()
    pallet_max_length = g["length"].max() / 1000
    pallet_max_width = g["width"].max() / 1000
    pallet_avg_height = (
        np.sum((np.array(g["height"]) / 1000 * np.array(g["quantity"])))
        / g["quantity"].sum()
    )
    pallet_total_volume = g["volume"].sum()
    if statistics.mode(g["stackable"]) == 0:
        num_stack = 1
    num_stack = min(num_stack, int(max_height / pallet_avg_height))
    #
    if unit_type in ["Weight"]:
        total_height = 0
        proxy_volume = pallet_total_volume
        for i in range(len(g["height"])):
            unit_height = g["height"].tolist()[i] / 1000 * g["quantity"].tolist()[i]
            total_height += unit_height
        proxy_pse = total_height / max_height
    elif unit_type in ["Pallet"]:
        num_pse = total_quantity // num_stack
        remainder_pse = total_quantity % num_stack
        pse_volume = pallet_max_length * pallet_max_width * max_height * num_pse
        remainder_volume = (
            pallet_max_length * pallet_max_width * max_height * remainder_pse
        ) / num_stack
        proxy_volume = pse_volume + remainder_volume
        proxy_pse = num_pse + remainder_pse / num_stack
    elif unit_type in ["Volume"]:
        proxy_volume = pallet_total_volume
        non_stackable_indices = [
            i for i, val in enumerate(g["stackable"].tolist()) if val == 0
        ]
        if len(non_stackable_indices) > 0:
            total_area = np.sum(
                np.array(g["width"].tolist())[non_stackable_indices]
                / 1000
                * np.array(g["length"].tolist())[non_stackable_indices]
                / 1000
                * np.array(g["quantity"].tolist())[non_stackable_indices]
            )
        else:
            total_area = 0
        proxy_pse = total_area  # for box optimize struck floor area instead of pse
    return pd.Series({"proxy_volume": proxy_volume, "proxy_pse": proxy_pse})


#
def capacity_check(
    locations,
    distances,
    durations,
    equipment,
    biggest_equipment,
    variant,
    capacity_check=False,
):
    # equipment = pd.read_parquet("data/equipment_processed_full.parquet")
    equipment = equipment[equipment.code == biggest_equipment]
    max_utilization = 0.95
    max_weight = equipment["maximumPayloadKg"].iloc[0].item() * max_utilization
    max_volume = equipment["volume"].iloc[0].item() * max_utilization
    max_pallet = equipment["palletSpacesUK"].iloc[0].item()
    #
    locations["over_weight"] = locations["weight"].apply(lambda x: x / max_weight)
    locations["over_volume"] = locations["proxy_volume"].apply(lambda x: x / max_volume)
    locations["over_pallet"] = locations["proxy_pse"].apply(lambda x: x / max_pallet)
    #
    routes = []
    if capacity_check:  # capacity already checked when order created
        # capacity check, do not change the key order
        capacity_check_column = {
            "over_volume": ["proxy_volume", max_volume],
            "over_weight": ["weight", max_weight],
            "over_pallet": ["proxy_pse", max_pallet],
        }
        # if unit_type == 'Weight':
        #     capacity_check_column = dict(sorted(capacity_check_column.items(), reverse=True))
        # elif unit_type == 'Pallet':
        #     capacity_check_column = dict(sorted(capacity_check_column.items(), reverse=False))
        location_index_adding = 2 if variant == "ovrptw" else 1
        while any(
            x > 1
            for x in [
                locations["over_weight"].max(),
                locations["over_volume"].max(),
                locations["over_pallet"].max(),
            ]
        ):
            for index, row in locations.iterrows():
                priority_capacity = (
                    row[["over_volume", "over_weight", "over_pallet"]]
                    .sort_values(ascending=False)
                    .index.tolist()
                )
                priority_capacity = priority_capacity[0]
                # print(capacity_check_column[priority_capacity[0]])
                # capacity_check_column = {k: capacity_check_column[k] for k in priority_capacity}
                v = capacity_check_column[priority_capacity]
                full_load_route = {}
                over_capacity = row[priority_capacity]
                if over_capacity > 1:
                    full_load_route["route_id"] = row.dest_name
                    full_load_route["vehicle_name"] = biggest_equipment
                    #
                    remain_volume = (
                        row["proxy_volume"]
                        * (row[v[0]] - int(over_capacity) * v[1])
                        / row[v[0]]
                    )
                    remain_pallet = (
                        row["proxy_pse"]
                        * (row[v[0]] - int(over_capacity) * v[1])
                        / row[v[0]]
                    )
                    remain_weight = (
                        row["weight"]
                        * (row[v[0]] - int(over_capacity) * v[1])
                        / row[v[0]]
                    )
                    locations.loc[index, "proxy_volume"] = remain_volume
                    locations.loc[index, "proxy_pse"] = remain_pallet
                    locations.loc[index, "weight"] = remain_weight
                    # check over capacity condition after orderlines splittingover_weight
                    locations.loc[index, "over_weight"] = (
                        0 if remain_weight <= max_weight else 1
                    )
                    locations.loc[index, "over_volume"] = (
                        0 if remain_volume <= max_volume else 1
                    )
                    locations.loc[index, "over_pallet"] = (
                        0 if remain_pallet <= max_pallet else 1
                    )
                    # add data to full load route
                    location_index = index + location_index_adding
                    # full_load_route["visits"] = [0, location_index]
                    full_load_route["visits"] = [
                        {
                            "visit_name": row["dest_name"].split("_")[0],
                            "directoryReference": row["dest_name"].split("_")[1],
                            "orderIds": [row["dest_name"].split("_")[2]],
                        }
                    ]
                    full_load_route["weight_delivery"] = int(
                        row["weight"] - remain_weight
                    )
                    full_load_route["volume_delivery"] = int(
                        row["proxy_volume"] - remain_volume
                    )
                    full_load_route["pallet_delivery"] = int(
                        row["proxy_pse"] - remain_pallet
                    )
                    full_load_route["distance"] = distances[0][index + 1]
                    full_load_route["vehicle_weight_utilization"] = round(
                        (row["weight"] - remain_weight) / max_weight, 2
                    )
                    full_load_route["vehicle_volume_utilization"] = round(
                        (row["proxy_volume"] - remain_volume) / max_volume, 2
                    )
                    full_load_route["vehicle_pallet_utilization"] = round(
                        (row["proxy_pse"] - remain_pallet) / max_pallet, 2
                    )
                    delivery_minute_of_day = int(
                        (
                            row["delivery_start_minutes_of_day"]
                            + row["delivery_end_minutes_of_day"]
                        )
                        / 2
                    )
                    full_load_route["location_arrival_time"] = [
                        delivery_minute_of_day - durations[0][index + 1],
                        delivery_minute_of_day,
                    ]
                    full_load_route["service_time"] = durations[0][index + 1]
                    full_load_route["note"] = priority_capacity
                    full_load_route["visit_name"] = [row["dest_name"]]
                    routes.append(full_load_route)
    return locations, routes


#
def process_account(
    df, num_stack, unit_type, equipment, biggest_equipment, variant, req_id
):
    proxy_volume = (
        df.groupby(["source_name", "dest_name"])
        .apply(
            get_proxy_volume,
            num_stack=num_stack,
            equipment=equipment,
            biggest_equipment=biggest_equipment,
            unit_type=unit_type,
        )
        .reset_index()
    )
    #
    order_id = df.groupby("order_number")["id"].apply(list).to_dict()
    #
    locations = (
        df.groupby(["source_name", "dest_name"])
        .agg(
            source_lat=("source_lat", "median"),
            source_long=("source_long", "median"),
            dest_lat=("dest_lat", "median"),
            dest_long=("dest_long", "median"),
            weight=("weight", "sum"),
            volume=("volume", "sum"),
            delivery_time_start=("delivery_time_start", "min"),
            delivery_time_end=("delivery_time_end", "max"),
            quantity=("quantity", "sum"),
            num_order=("id", "count"),
            height=("height", list),
            length=("length", list),
            width=("width", list),
            unit_id=("id", list),
        )
        .reset_index()
    )
    locations = locations.merge(proxy_volume, how="left", on=["source_name", "dest_name"])
    #
    # locations["delivery_time_end"] = (
    #     locations["delivery_time_end"].replace("06:00", "18:00")
    # )
    locations["delivery_time_start"] = (
        locations["delivery_time_start"].replace("nan", "00:00") + ":00"
    )
    locations["delivery_time_end"] = (
        locations["delivery_time_end"].replace("nan", "00:00") + ":00"
    )
    locations["delivery_time_end"] = locations["delivery_time_end"].replace(
        {"00:00:00": "23:59:00"}
    )
    locations["delivery_start_minutes_of_day"] = (
        pd.to_datetime(locations["delivery_time_start"], format="%H:%M:%S").dt.hour * 60
        + pd.to_datetime(locations["delivery_time_start"], format="%H:%M:%S").dt.minute
    )
    locations["delivery_end_minutes_of_day"] = (
        pd.to_datetime(locations["delivery_time_end"], format="%H:%M:%S").dt.hour * 60
        + pd.to_datetime(locations["delivery_time_end"], format="%H:%M:%S").dt.minute
    )
    #
    locations = locations[locations.dest_long > -5].reset_index(drop=True)
    #
    # solver_utils.location_visualization(locations)
    #
    depot = locations[["source_long", "source_lat"]].drop_duplicates().values.tolist()[0]
    dest_names = locations["dest_name"].tolist()
    source_name = df.loc[0, "source_name"] + "_" + df.loc[0, "source_id"]
    dest_names.insert(0, source_name)
    coordinates = [tuple(x) for x in locations[["dest_long", "dest_lat"]].values.tolist()]
    coordinates.insert(0, (depot[0], depot[1]))
    # distances, durations, speed = get_ggmap_matrix_batched(coordinates, mode="driving")
    # distances, durations, speed = get_tomtom_matrix_async(coordinates, batch_size=25)
    # with open("temp_data/distances.pkl", "wb") as file:
    #     pickle.dump(distances, file)
    # with open("temp_data/durations.pkl", "wb") as file:
    #     pickle.dump(durations, file)
    # with open("temp_data/speed.pkl", "wb") as file:
    #     pickle.dump(speed, file)
    #
    with open("temp_data/distances.pkl", "rb") as file:
        distances = pickle.load(file)
    with open("temp_data/durations.pkl", "rb") as file:
        durations = pickle.load(file)
    with open("temp_data/speed.pkl", "rb") as file:
        speed = pickle.load(file)
    durations = [[int(x / 60) for x in sublist] for sublist in durations]
    distances = [[int(x) for x in sublist] for sublist in distances]
    #
    locations, full_load_route = capacity_check(
        locations, distances, durations, equipment, biggest_equipment, variant
    )
    #
    weights = locations["weight"].tolist()
    weights.insert(0, 0)
    # weights = [int(i) for i in weights]
    #
    volumes = locations["proxy_volume"].tolist()
    volumes.insert(0, 0)
    # volumes = [int(i) for i in volumes]
    #
    pallets = locations["proxy_pse"].tolist()
    pallets.insert(0, 0)
    # pallets = [int(i) for i in pallets]
    #
    time_windows = locations[
        ["delivery_start_minutes_of_day", "delivery_end_minutes_of_day"]
    ].values.tolist()
    time_windows.insert(0, (0, 1440))
    print("=" * 10 + "PROCESS ACCOUNT ORDERS = DONE" + "=" * 10)
    return (
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
    )
