import bentoml
from PySide6.QtCore import QThread, Signal
import subprocess
import requests
from photolink import get_application_path, get_config_file
from photolink.utils.function import read_config
import photolink.models.yunet as yunet
import photolink.models.sface as sface
from pathlib import Path
import os
import shutil
import pickle
import signal

# Global variables for pre-loaded models
YUNET_MODEL = None
SFACE_MODEL = None

def load_models():
    """Warmup the models."""
    global YUNET_MODEL, SFACE_MODEL
    if YUNET_MODEL is None:
        YUNET_MODEL = yunet.load_model()
    if SFACE_MODEL is None:
        SFACE_MODEL = sface.load_model() 


@bentoml.service(traffic={"timeout": 10})
class FaceMatchingService:

    def __init__(self):
        """Initialize the service. Load the models to speed up processing."""
        load_models()

    @bentoml.api
    def run_ml_model(image_path: str, save_path: str, fail_path: str, keep_top_n: int = 3):
        """Run the face matching model."""
        global YUNET_MODEL, SFACE_MODEL
        detection_result = YUNET_MODEL.run_face_detection(image_path)
        print(f"Pre-Processing:{image_path}", flush=True)
        failed_image = Path(fail_path) / os.path.basename(image_path)

        if "error" in detection_result:
            shutil.copy(image_path, failed_image)
            warning = f"Face detection error on source image {image_path}: {detection_result['error']}"

            return warning

        if "faces" not in detection_result:
            shutil.copy(image_path, failed_image)
            warning = f"Face not detected on source image {image_path}. Faces probably not there, or too small."

            return warning

        image = detection_result["image"]
        faces = detection_result["faces"]
        keep_top_n = int(keep_top_n)

        # Only go over the top n faces instead of going for all faces.
        if faces.shape[0] < keep_top_n:
            keep_top_n = faces.shape[0]

        # bb = [x, y, w, h, landmarks]
        faces = sorted(faces, key=lambda face: face[2] * face[3], reverse=True)[:keep_top_n]
        embedding_dict = SFACE_MODEL.run_embedding_conversion(image, faces)

        if "error" in embedding_dict:
            shutil.copy(image_path, failed_image)
            warning = f"Face recognition error on source image {image_path}: {embedding_dict['error']}"

            return warning

        save_embedding_name = Path(save_path) / Path(os.path.basename(image_path).split(".")[0] + ".pkl")

        # pickle dump all face embeddings found in the image.
        with open(save_embedding_name, "wb") as f:
            pickle.dump(embedding_dict, f)

        return True

class ServerThread(QThread):
    """Thread to start the BentoML service."""

    server_ready = Signal(bool)

    def __init__(self):
        super().__init__()
        self.application_path = get_application_path()
        self.config_path = get_config_file(self.application_path)
        self.config_data = read_config(self.config_path)
        self.port = self.config_data["MODEL"]["PORT"]

    def run(self):
        try:
            self.start_bentoml_service()
            server_up = self.check_server_health()
            self.server_ready.emit(server_up)
        except Exception as e:
            print(f"Failed to start server: {e}")
            self.server_ready.emit(False)

    def start_bentoml_service(self):
        # Start the BentoML service in a subprocess
        subprocess.Popen(["bentoml", "serve", "photolink.server.service:FaceMatchingService", "--port", f"{self.port}"])
        # time.sleep(5)  # Give some time for the server to start
        print("BentoML service started")

    def check_server_health(self):
        try:
            response = requests.get(f"http://localhost:{self.port}/healthz")  # Modify URL as needed
            print(f"Server health check response: {response.status_code}")
            return response.status_code == 200
        except requests.RequestException:
            return False
        
    def stop(self):
        if self.process:
            self.process.terminate()  # Send SIGTERM to the process
            self.process.wait()       # Wait for the process to terminate
            print("BentoML service stopped")

def signal_handler(signal_received, frame, server_thread):
    print(f'Signal {signal_received} received, shutting down gracefully...')
    server_thread.stop()
    os._exit(0)  #

# Set up signal handling
server_thread = ServerThread()
signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, server_thread))
signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, server_thread))