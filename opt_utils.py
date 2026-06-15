import numpy as np
import requests
import json
import time
import io
from datetime import date
from redis.cluster import RedisCluster
from azure.storage.blob import BlobClient, BlobServiceClient
from azure.cosmos import CosmosClient
from opt_cf import env_config as opt_config


def query_cosmos(container_name, database_name, query):
    URL = opt_config.cosmos_url
    KEY = opt_config.cosmos_key
    client = CosmosClient(URL, KEY)
    database = client.get_database_client(database_name)
    container = database.get_container_client(container_name)
    #
    items = list(container.query_items(query, enable_cross_partition_query=True))
    return items


def build_route(data):
    url = opt_config.build_route_url
    headers = {"Ocp-Apim-Subscription-Key": opt_config.build_route_api_key}
    # If you have a request body, put it here. Currently your curl has --data empty.
    response = requests.post(
        url, headers=headers, json=data
    )  # use json= if sending JSON
    print("Status Code:", response.status_code)
    print("Response Text:", response.text)
    return


def get_location_matrix_redis(requests):
    """
    requests: list of tuples [("origin1", "dest1"), ("origin2", "dest5")]
    """
    r = RedisCluster(
        host=opt_config.redis_host,
        port=10000,
        password=opt_config.redis_key,
        ssl=True,
        ssl_check_hostname=False,
        decode_responses=True
    )
    pipe = r.pipeline(transaction=False)
    for origin, dest in requests:
        pipe.hget(f"loc:{origin}", dest)
    # This sends all requests to Azure in one go
    raw_results = pipe.execute()
    final_results = {}
    for i, raw_val in enumerate(raw_results):
        if raw_val:
            origin, dest = requests[i]
            if origin not in final_results:
                final_results[origin] = {}
            final_results[origin][dest] = json.loads(raw_val)
    return final_results


def missing_route_checking(address_guids, location_matrix):
    n = len(address_guids)
    if len(location_matrix) != n:
        return True
    #
    for k, v in location_matrix.items():
        if len(v) == n:
            continue
        else:
            return True
    return False


def get_tomtom_matrix_async(points, travelMode="car", batch_size=50):
    base_url = "https://api.tomtom.com/routing/matrix/2"
    api_key = opt_config.tomtom_api_key
    n = len(points)
    print(f"Call Tomtom API with {n} locations, chunking into {batch_size}x{batch_size}")    
    # Initialize the master N x N matrices
    distances = [[None] * n for _ in range(n)]
    durations = [[None] * n for _ in range(n)]
    # 1st Loop: Chunk the Origins
    for start_orig in range(0, n, batch_size):
        end_orig = min(start_orig + batch_size, n)
        batch_origins = points[start_orig:end_orig]
        origin_locations = [{"point": {"latitude": p[1], "longitude": p[0]}} for p in batch_origins]
        # 2nd Loop: Chunk the Destinations
        for start_dest in range(0, n, batch_size):
            end_dest = min(start_dest + batch_size, n)
            batch_dests = points[start_dest:end_dest]
            dest_locations = [{"point": {"latitude": p[1], "longitude": p[0]}} for p in batch_dests]
            print(f"\n--- Processing Batch: Origins {start_orig}-{end_orig-1} | Destinations {start_dest}-{end_dest-1} ---")
            payload = {
                "origins": origin_locations,
                "destinations": dest_locations,
                "options": {
                    "travelMode": travelMode, # Must be "car" or "truck"
                    "departAt": "any",
                    "routeType": "fastest",
                    "traffic": "historical"
                }
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
                    print(f"Batch completed!")
                    break
                elif status == "Failed":
                    print(f"\nBatch failed: {status_data}")
                    break 
                else:
                    print(f"Status: {status}... waiting 5 seconds.", end="\r")
                    time.sleep(5)
            # 3. FETCH RESULTS & PARSE (GET)
            if status == "Completed":
                result_url = f"{base_url}/async/{job_id}/result?key={api_key}"
                result_resp = requests.get(result_url)
                result_data = result_resp.json()
                for element in result_data.get("data", []):
                    rel_origin_idx = element.get("originIndex")
                    rel_dest_idx = element.get("destinationIndex")
                    # Map relative batch indices back to absolute indices in the master matrix
                    abs_origin_idx = start_orig + rel_origin_idx
                    abs_dest_idx = start_dest + rel_dest_idx
                    if "routeSummary" in element:
                        summary = element["routeSummary"]
                        distances[abs_origin_idx][abs_dest_idx] = summary["lengthInMeters"] / 1000 # km
                        durations[abs_origin_idx][abs_dest_idx] = summary["travelTimeInSeconds"]
    return distances, durations


def save_location_matrix_redis(data):
    """
    data: dict of dict {original1: {destination1: [travel_distance, travel_time, update_date]}}
    """
    r = RedisCluster(
        host=opt_config.redis_host,
        port=10000,
        password=opt_config.redis_key,
        ssl=True,
        ssl_check_hostname=False,
        decode_responses=True
    )
    pipe = r.pipeline()
    count = 0
    for origin, dest_updates in data.items():
        # Optimization: use mapping to update multiple destinations for a single origin in one go.
        serialized_updates = {
            dest: json.dumps(values) 
            for dest, values in dest_updates.items()
        }
        # update existing and add new destinations automatically
        pipe.hset(name=f"loc:{origin}", mapping=serialized_updates)
        count += len(dest_updates)
        if count >= 10000:
            pipe.execute()
            print(f"Processed {count} updates/expansions...")
            count = 0
    pipe.execute()
    print("All updates and expansions completed.")

def upload_route_matrix_to_blob(connection_string, container_name, blob_name, distances, durations):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, distances=distances, durations=durations)
    buffer.seek(0)
    # Create a BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    # Upload data
    container_client.upload_blob(blob_name, buffer, overwrite=True)
    return


def get_location_matrix(coordinates, address_guids, req_id, order_number_id, directory_ref):
    n = len(address_guids)
    guids_index_dict = {}
    for i, v in enumerate(address_guids):
        guids_index_dict[v] = i
    # Step 1: Get data from redis
    location_pairs = [(i, v) for i in address_guids for v in address_guids]
    location_matrix = get_location_matrix_redis(location_pairs)
    route_missing = missing_route_checking(address_guids, location_matrix)
    if route_missing:
        # Step 2: Call tomtom matrix API
        distances, durations = get_tomtom_matrix_async(coordinates)
        # Step 2.1: Check un-reach locations
        unreachable_location = any(None in sublist for sublist in durations)
        if unreachable_location:
            unreachable_location_index = [i for i, v in enumerate(durations[0]) if v is None]
            unreachable_location_name = [order_number_id[i-1][0] for i in unreachable_location_index]
            message = "Unreachable location: Order number " + ", ".join(str(x) for sublist in unreachable_location_name for x in sublist)
            output = {
                "status": False,
                "id": req_id,
                "note": message,
                'directory_reference': directory_ref
            }
            build_route(output)
            raise ValueError(message)
        # Step 3: Convert Tomtom output
        durations = [[int(x / 60) for x in sublist] for sublist in durations]
        distances = [[int(x) for x in sublist] for sublist in distances]
        # Step 4: Save new data to redis
        location_matrix = {}
        for i, o in enumerate(address_guids):
            loc_element = {}
            for v, d in enumerate(address_guids):
                loc_element[d] = [int(distances[i][v]), int(durations[i][v]), date.today().strftime("%Y-%m-%d")]
            location_matrix[o] = loc_element
        save_location_matrix_redis(location_matrix)
    else:
        print("Get all routes from cached data")
        distances = [[None] * n for _ in range(n)]
        durations = [[None] * n for _ in range(n)]
        for orig, v in location_matrix.items():
            for dest, t in v.items():
                distances[guids_index_dict[orig]][guids_index_dict[dest]] = t[0]
                durations[guids_index_dict[orig]][guids_index_dict[dest]] = t[1]
    # save matrices to blob
    upload_route_matrix_to_blob(
        opt_config.connection_string,
        opt_config.input_container_name,
        f"route_matrix/{req_id}.npy",
        np.array(distances),
        np.array(durations)
    )
    return distances, durations


def upload_file_to_blob(connection_string, container_name, blob_name, data):
    # Create a BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    # Upload data
    container_client.upload_blob(blob_name, data, overwrite=True)
    return


def build_route(data):
    url = opt_config.build_route_url
    headers = {"Ocp-Apim-Subscription-Key": opt_config.build_route_api_key}
    # If you have a request body, put it here. Currently your curl has --data empty.
    response = requests.post(
        url, headers=headers, json=data
    )  # use json= if sending JSON
    print("Status Code:", response.status_code)
    print("Response Text:", response.text)
    return