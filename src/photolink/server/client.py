"""Singleton pattern for client."""
import bentoml
from photolink import get_application_path, get_config_file
from photolink.utils.function import read_config
from PySide6.QtCore import QRunnable
import asyncio

class ThreadedPreprocess(QRunnable):
    """Create a BentoMLClient thread to pass jobs to the BentoML service. Think of this as a bridge."""
    def __init__(self, image_path, save_path, fail_path, keep_top_n,):
        super().__init__()
        self.image_path = image_path
        self.save_path = save_path
        self.fail_path = fail_path
        self.keep_top_n = keep_top_n

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(self.run_model())
        loop.close()

    async def run_model(self):
        client = get_client()
        result = await client.run_ml_model(self.image_path, self.save_path, self.fail_path, self.keep_top_n)
        return result


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