"""Enums for the application."""
from enum import Enum

class Task(Enum):
    """Enum for task type."""
    SAMPLE_MATCHING = "Sample match is for matching one source photo against multiple unlabeled photos of the subject. You can have many as many subjects as you want, but ensure that you only have one source image per subject. Results will be saved to output path."
    CLUSTERING = "When you do not know how many subjects you have, place all images into a single folder and run clustering. Each cluster represents identified subject. Uncertain ones will be transferred to uncertain folder. Subject must have at least two images to be considered as a cluster."

class ErrorMessage(Enum):
    PATH_NOT_SELECTED = "Error, Please select all required valid paths."
    SOURCE_FOLDER_EMPTY = "Error, Please make sure that there are image files in the source folder."
    REFERENCE_FOLDER_EMPTY = "Error, Please make sure that there are image files in the reference folder."
    REFRESH_REQUIRED = "Please hit refresh first and try again."

class StatusMessage(Enum):
    DEFAULT = "Welcome to PhotoMatcher! When there are more than one faces in a photo, top 3 largest faces will be used. If you want to run a new task after completion, or if you find something buggy, please hit refresh. For any other issues, you can contact the developer."
    COMPLETE = "Task completed successfully!"

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

IMAGE_EXTENSION = [
    "ase",
    "art",
    "bmp",
    "blp",
    "cd5",
    "cit",
    "cpt",
    "cr2",
    "cut",
    "dds",
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