import os
from dotenv import load_dotenv

env_file = os.path.join(os.path.dirname(__file__), "resources/config.env")
load_dotenv(env_file)


if __name__ == "__main__":
    # gaurd behind the main block
    from photomatcher.app import PhotoMatcher
    import multiprocessing as mp
    mp.freeze_support() 
    
    # start photomatcher app
    pm = PhotoMatcher()
    pm.main_loop()