class NoFaceDetectedError(Exception):
    """Custom exception raised when no faces are detected in an image."""

    def __init__(self, message="No faces detected in the image."):
        super().__init__(message)
