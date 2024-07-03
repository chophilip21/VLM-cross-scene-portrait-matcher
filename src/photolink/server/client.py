"""Singleton pattern for client."""
import bentoml

class Localclient:
    """Singleton pattern for client."""

    def __init__(self):
        self.host = "localhost"
        self.port = 5000
        self.client = None

    def health(self):
        """Health check."""
        return self.client.get("/health")

    @property
    def client(self):
        if self.client is None:
            self.client = bentoml.SyncHTTPClient(base_url=f"http://{self.host}:{self.port}")
        return self.client
    

def get_client():
    """Get the client."""
    return Localclient().client
