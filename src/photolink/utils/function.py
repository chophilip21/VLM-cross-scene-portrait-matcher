import photolink.utils.enums as enums
import glob


def search_all_images(path):
    """Recursively search all images in a directory."""
    images = []
    path_ = path + '/**/*.*'
    files = glob.glob(path_,recursive = True) 
    for file in files:
        
        if file.split('.')[-1].lower() in enums.IMAGE_EXTENSION:
            images.append(file)

    return images