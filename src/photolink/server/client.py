"""Singleton pattern for client."""
import bentoml
from photolink import get_application_path, get_config_file
from photolink.utils.function import read_config

_instance = None

def get_client():
    global _instance
    if _instance is None:
        _instance = BentoMLClient().client
    return _instance


class BentoMLClient:
    """Main client for the BentoML service. Singleton pattern."""

    def __init__(self):
        application_path = get_application_path()
        config_path = get_config_file(application_path)
        config_data = read_config(config_path)
        self.port = config_data["MODEL"]["PORT"]

    @property
    def client(self):
        self._instance = bentoml.AsyncHTTPClient(f"http://localhost:{self.port}")
        return self._instance