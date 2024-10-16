"""Enums for the application."""

from enum import Enum


class Task(Enum):
    """Enum for task type."""

    FACE_SEARCH = "Search your target subject from heaps of unlabeled photos. Source refers to an image of a person you are looking for. You can add multiple subjects to the source folder, but ensure that you use exactly one source image per subject, and that each source photo has precisely one person."
    CLUSTERING = "When you do not know how many subjects you have, place all images into a single folder and run clustering. Each cluster represents identified subject. Uncertain ones will be transferred to uncertain folder. Subject must have at least two images to be considered as a cluster."
    DP2_MATCH = "Automatically detect main subject (e.g the graduating student) in the photo to ensure photos of the subjects are in order. Apply culling algorithm to remove duplicates/bad photos and get exactly the same number of photos on both source and reference folders."


class ErrorMessage(Enum):
    PATH_NOT_SELECTED = "Error, Please select all required valid paths."
    SOURCE_FOLDER_EMPTY = (
        "Error, Please make sure that there are image files in the source folder."
    )
    REFERENCE_FOLDER_EMPTY = (
        "Error, Please make sure that there are image files in the reference folder."
    )
    REFRESH_REQUIRED = "Please hit refresh first and try again."


class StatusMessage(Enum):
    DEFAULT = "Welcome to PhotoMatcher! When there are more than one faces in a photo, top 3 largest faces will be used. If you want to run a new task after completion, or if you find something buggy, please hit refresh. For any other issues, you can contact the developer."
    COMPLETE = "Task completed successfully!"
    ERROR = "An error occurred. Please check the logs for more information."
    STOPPED = "Task manager shutting down."


class OperatingSystem(Enum):
    """Enum for operating system."""

    WINDOWS = "win32"
    MACOS = "darwin"
    LINUX = "linux"


class ClusteringAlgorithm(Enum):
    """Enum for clustering algorithm."""

    DBSCAN = "DBSCAN"
    OPTICS = "OPTICS"
    HDBSCAN = "HDBSCAN"


# TODO: CR2 removed. Add support for CR2 images if needed.
IMAGE_EXTENSION = [
    "ase",
    "art",
    "bmp",
    "blp",
    "cd5",
    "cit",
    "cpt",
    "dib",
    "djvu",
    "egt",
    "exif",
    "gif",
    "gpl",
    "grf",
    "icns",
    "ico",
    "iff",
    "jng",
    "jpeg",
    "jpg",
    "jfif",
    "jp2",
    "jps",
    "lbm",
    "max",
    "miff",
    "mng",
    "msp",
    "nef",
    "nitf",
    "ota",
    "pbm",
    "pc1",
    "pc2",
    "pc3",
    "pcf",
    "pcx",
    "pdn",
    "pgm",
    "PI1",
    "PI2",
    "PI3",
    "pict",
    "pct",
    "pnm",
    "pns",
    "ppm",
    "psb",
    "psd",
    "pdd",
    "psp",
    "px",
    "pxm",
    "pxr",
    "qfx",
    "raw",
    "rle",
    "sct",
    "sgi",
    "rgb",
    "int",
    "bw",
    "tga",
    "tiff",
    "tif",
    "vtf",
    "xbm",
    "xcf",
    "xpm",
    "3dv",
    "amf",
    "ai",
    "awg",
    "cgm",
    "cdr",
    "cmx",
    "dxf",
    "e2d",
    "egt",
    "eps",
    "fs",
    "gbr",
    "odg",
    "svg",
    "stl",
    "vrml",
    "x3d",
    "sxd",
    "v2d",
    "vnd",
    "wmf",
    "emf",
    "art",
    "xar",
    "png",
    "webp",
    "jxr",
    "hdp",
    "wdp",
    "cur",
    "ecw",
    "iff",
    "lbm",
    "liff",
    "nrrd",
    "pam",
    "pcx",
    "pgf",
    "sgi",
    "rgb",
    "rgba",
    "bw",
    "int",
    "inta",
    "sid",
    "ras",
    "sun",
    "tga",
    "heic",
    "heif",
]
