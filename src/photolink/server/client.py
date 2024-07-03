"""Singleton pattern for client."""
import bentoml
from photolink import get_application_path, get_config_file
from photolink.utils.function import read_config
from PySide6.QtCore import QRunnable
import asyncio

class ThreadedPreprocess(QRunnable):
    """Create a BentoMLClient thread to pass jobs to the BentoML service. Think of this as a bridge between the main app and the BentoML service."""
    def __init__(self, image_path, save_path, fail_path, keep_top_n, callback):
        super().__init__()
        self.image_path = image_path
        self.save_path = save_path
        self.fail_path = fail_path
        self.keep_top_n = keep_top_n
        self.callback = callback

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(self.run_model())
        loop.close()
        self.callback(result)

    async def run_model(self):
        client = BentoMLClient()
        result = await client.run_ml_model(self.image_path, self.save_path, self.fail_path, self.keep_top_n)
        return result

class BentoMLClient:
    """Main client for the BentoML service. Singleton pattern."""
    _instance = None

    def __new__(cls):
        """Singleton pattern for client using async http client."""
        if cls._instance is None:
            application_path = get_application_path()
            config_path = get_config_file(application_path)
            config_data = read_config(config_path)
            port = config_data["MODEL"]["PORT"]
            cls._instance = super(BentoMLClient, cls).__new__(cls)
            cls._instance.client = bentoml.AsyncHTTPClient.from_url(f"http://localhost:{port}")
        return cls._instance

    async def run_ml_model(self, image_path, save_path, fail_path, keep_top_n):
        """run_ml_model is same name as the API in the BentoML service."""
        data = {
            "image_path": image_path,
            "save_path": save_path,
            "fail_path": fail_path,
            "keep_top_n": keep_top_n
        }
        return await self.client.async_predict(data)

