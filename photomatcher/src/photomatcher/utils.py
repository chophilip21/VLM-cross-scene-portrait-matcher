import photomatcher.enums as enums
import glob


def search_all_images(path):
    """Recursively search all images in a directory."""
    images = []
    path_ = path + '/**/*.*'
    files = glob.glob(path_,recursive = True) 
    for file in files:
        if file.split('.')[-1] in enums.IMAGE_EXTENSION:
            images.append(file)

    return files

if __name__ == "__main__":
    test = search_all_images("/home/chophilip21/PL_ML/dataset/selfie_dataset/processed/labeled")