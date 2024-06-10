import os
from dotenv import load_dotenv

env_file = os.path.join(os.path.dirname(__file__), "resources/config.env")
load_dotenv(env_file)

if __name__ == "__main__":
    from photomatcher.app import main
    main().main_loop()
