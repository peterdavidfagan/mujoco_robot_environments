import os
import glob
import tarfile

import tensorflow_datasets as tfds
from huggingface_hub import HfApi

LOCAL_FILEPATH="/home/peter/Code/temp/mujoco_robot_environments/mujoco_robot_environments/data"

if __name__=="__main__":

    for folder_name in os.listdir(LOCAL_FILEPATH):
        if os.path.isdir(os.path.join(LOCAL_FILEPATH, folder_name)):
            OUTPUT_FILENAME = folder_name + '.tar.gz'
            with tarfile.open(OUTPUT_FILENAME, "w:xz") as tar:
                tar.add(os.path.join(LOCAL_FILEPATH, folder_name), arcname=".")

            # upload to huggingface
            api = HfApi()
            api.upload_file(
                repo_id="peterdavidfagan/transporter_networks_mujoco",
                repo_type="dataset",
                path_or_fileobj=f"./{OUTPUT_FILENAME}",
                path_in_repo=f"/{OUTPUT_FILENAME}",
            )
