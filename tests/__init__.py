import os
from mongomock import MongoClient
import random

from cgnal.core.utils.fs import create_dir_if_not_exists

test_path = os.path.dirname(os.path.abspath(__file__))

DATA_FOLDER = os.path.join(test_path, "resources", "data")
TMP_FOLDER = str(
    create_dir_if_not_exists(os.path.join("/tmp", "%032x" % random.getrandbits(128)))
)

os.environ["TMP_LOG_FOLDER"] = str(
    create_dir_if_not_exists(os.path.join(TMP_FOLDER, "logs"))
)

DB_NAME = "db"

client = MongoClient()

db = client[DB_NAME]


def clean_tmp_folder():
    os.system(f"rm -rf {TMP_FOLDER}/*")


def unset_TMP_FOLDER():
    del os.environ["TMP_LOG_FOLDER"]
