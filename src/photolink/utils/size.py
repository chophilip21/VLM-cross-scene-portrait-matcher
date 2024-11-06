"""Quick script to get the size of a Python library"""
from pathlib import Path
import argparse


def get_library_size(package_name):
    # Find the package location
    try:
        package = __import__(package_name)
        package_path = Path(package.__file__).parent
    except ImportError:
        print(f"ERROR: Could not find the package {package_name}")
        return None

    # Calculate the size
    total_size = sum(f.stat().st_size for f in package_path.rglob('*'))
    return total_size / (1024 ** 2)  # Convert bytes to MB


def main():
    parser = argparse.ArgumentParser(description="Get the size of a Python library")
    parser.add_argument("library", type=str, help="The name of the library to check")
    args = parser.parse_args()

    size = get_library_size(args.library)
    if size is not None:
        print(f"The size of {args.library} is {size} MB")


if __name__ == "__main__":
    main()
