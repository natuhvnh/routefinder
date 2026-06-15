import os
from dotenv import load_dotenv
load_dotenv()


class SolverConfig:
    def __init__(self):
        # routing config
        self.vehicle_max_duration = 660
        self.client_service_duration = 10
        # map key
        self.google_api_key = ""
        self.openroute_api_key = ""
        # cosmos
        self.cosmos_url = os.getenv("cosmos_url")
        self.cosmos_key = os.getenv("cosmos_key")
        # blob save
        self.connection_string = os.getenv("blob_connection_string")
        self.account_url = os.getenv("account_url")
        self.input_container_name = os.getenv("input_container_name")
        self.input_blob_token = os.getenv("input_blob_token")
        self.output_container_name = os.getenv("output_container_name")
        self.output_blob_token = os.getenv("output_blob_token")
        # tomtom
        self.tomtom_url = os.getenv("tomtom_url")
        self.tomtom_api_key = os.getenv("tomtom_api_key")
        # build route api
        self.build_route_url = os.getenv("build_route_url")
        self.build_route_api_key = os.getenv("build_route_api_key")
        # redis
        self.redis_host = os.getenv("redis_host")
        self.redis_key = os.getenv("redis_key")

env_config = SolverConfig()
