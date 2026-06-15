import pandas as pd
import numpy as np
import pickle
import time
import requests
import statistics
import opt_utils


def process_backhaul_order(df):
    source_name = statistics.mode(df["source_name"].tolist() + df["dest_name"].tolist())
    df_delivery = df[df.source_name == source_name]
    df_delivery = df_delivery.assign(backhaul=1)
    #
    df_pickup = df[~(df.source_name == source_name)]
    for row in df_pickup.itertuples():
        source_id = row.dest_id
        source_name = row.dest_name
        source_lat = row.dest_lat
        source_long = row.dest_long
        dest_id = row.source_id
        dest_name = row.source_name + " pickup"
        dest_lat = row.source_lat
        dest_long = row.source_long
        #
        df_pickup.loc[row.Index, "source_id"] = source_id
        df_pickup.loc[row.Index, "source_name"] = source_name
        df_pickup.loc[row.Index, "source_lat"] = source_lat
        df_pickup.loc[row.Index, "source_long"] = source_long
        df_pickup.loc[row.Index, "dest_id"] = dest_id
        df_pickup.loc[row.Index, "dest_name"] = dest_name
        df_pickup.loc[row.Index, "dest_lat"] = dest_lat
        df_pickup.loc[row.Index, "dest_long"] = dest_long
    df_pickup = df_pickup.assign(backhaul=-1)
    df = pd.concat([df_delivery, df_pickup], ignore_index=True)
    return df


def process_order(df):
    df = df[df.collection_date.notnull()]
    df = df[df.collection_date >= "2025-01-01"]
    df["height"] = np.where(
        df["dimensionUnit"].str.lower() == "cm", df["height"] * 10, df["height"]
    )
    df["length"] = np.where(
        df["dimensionUnit"].str.lower() == "cm", df["length"] * 10, df["length"]
    )
    df["width"] = np.where(
        df["dimensionUnit"].str.lower() == "cm", df["width"] * 10, df["width"]
    )
    df["dimensionUnit"] = "mm"
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)
    df["dest_name"] = df["dest_name"] + "_" + df["dest_id"]
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
    #
    print("=" * 10 + " GET ORDERS = DONE " + "=" * 10)
    return df


def pre_check(durations, locations, req_id, equipment, directory_ref):
    dest_name = locations["dest_name"].tolist()
    max_duration = max(durations[0])
    if max_duration > equipment["maximumDrivingTimeInMinutes"].max():
        max_duration_dest_name = dest_name[durations[0].index(max_duration) - 1]
        message = f"The travelling time to {max_duration_dest_name} = {max_duration} minutes, which is higher than the max service duration = {equipment['maximumDrivingTimeInMinutes'].max()} minutes"
        output = {
            "status": False,
            "id": req_id,
            "note": message,
            "directory_reference": directory_ref,
        }
        opt_utils.build_route(output)
        raise ValueError(message)
    #
    location_name = []
    travel_time = []
    service_time_window = []
    order_number = []
    order_id = []
    for index, row in locations.iterrows():
        travel_time_constraint = (
            row["delivery_end_minutes_of_day"] - row["delivery_start_minutes_of_day"]
        )
        travel_time_actual = durations[0][index + 1]
        if travel_time_actual > travel_time_constraint:
            location_name.append(row["dest_name"].split("_")[0])
            travel_time.append(str(travel_time_actual))
            service_time_window.append(
                f"{row['delivery_time_start']} and {row['delivery_time_end']}"
            )
            order_number.append(row["dest_name"].split("_")[-1])
            order_id.extend(row["unit_id"])
    if len(location_name) > 0:
        message = (
            f"The travelling time to {', '.join(location_name)}: {', '.join(travel_time)} minutes, respectively."
            f" Out of service time windows between {', '.join(service_time_window)}."
            f" Order number: {', '.join(order_number)}."
            f" Order id: {' '.join(order_id)}"
        )
        output = {
            "status": False,
            "id": req_id,
            "note": message,
            "directory_reference": directory_ref,
        }
        opt_utils.build_route(output)
        raise ValueError(message)
    return True


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


def process_account(
    df,
    num_stack,
    unit_type,
    equipment,
    biggest_equipment,
    variant,
    req_id,
    directory_ref,
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
            backhaul=("backhaul", lambda x: x.mode().iloc[0]),
            source_location_id=("source_location_id", lambda x: x.mode().iloc[0]),
            dest_location_id=("dest_location_id", lambda x: x.mode().iloc[0]),
            height=("height", list),
            length=("length", list),
            width=("width", list),
            unit_id=("id", list),
            order_number=("order_number", list),
        )
        .reset_index()
    )
    locations["order_number_id"] = locations[
        ["order_number", "unit_id"]
    ].values.tolist()
    locations = locations.drop(["order_number"], axis=1)
    locations = locations.merge(
        proxy_volume, how="left", on=["source_name", "dest_name"]
    )
    #
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
    order_number_id = locations["order_number_id"].tolist()
    #
    depot = (
        locations[["source_long", "source_lat"]].drop_duplicates().values.tolist()[0]
    )
    dest_names = locations["dest_name"].tolist()
    source_name = df.loc[0, "source_name"] + "_" + df.loc[0, "source_id"]
    dest_names.insert(0, source_name)
    coordinates = [
        tuple(x) for x in locations[["dest_long", "dest_lat"]].values.tolist()
    ]
    coordinates.insert(0, (depot[0], depot[1]))
    address_guids = locations["dest_location_id"].tolist()
    address_guids.insert(0, locations["source_location_id"].mode()[0])
    distances, durations = opt_utils.get_location_matrix(
        coordinates, address_guids, req_id, order_number_id, directory_ref
    )
    total_distance = np.sum(distances)  # km
    total_time_hours = np.sum(durations) / 60  # hours
    speed = total_distance / total_time_hours
    #
    locations, full_load_route = capacity_check(
        locations, distances, durations, equipment, biggest_equipment, variant
    )
    locations["weight"] = locations["weight"] * locations["backhaul"]
    locations["proxy_volume"] = locations["proxy_volume"] * locations["backhaul"]
    locations["proxy_pse"] = locations["proxy_pse"] * locations["backhaul"]
    #
    weights = locations["weight"].tolist()
    weights.insert(0, 0)
    #
    volumes = locations["proxy_volume"].tolist()
    volumes.insert(0, 0)
    #
    pallets = locations["proxy_pse"].tolist()
    pallets.insert(0, 0)
    #
    time_windows = locations[
        ["delivery_start_minutes_of_day", "delivery_end_minutes_of_day"]
    ].values.tolist()
    time_windows.insert(0, (0, 1440))
    print("=" * 10 + " PROCESS ACCOUNT ORDERS = DONE " + "=" * 10)
    return (
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
        speed
    )