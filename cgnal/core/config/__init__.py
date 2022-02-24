import os
import re
import sys
import cfg_load

from yaml import (
    Loader,
    add_implicit_resolver,
    add_constructor,
    Node,
    FullLoader,
    UnsafeLoader,
)
from typing import Optional, Any, Hashable, Union, Sequence, List
from cfg_load import Configuration
from functools import reduce
from cgnal.core.typing import PathLike
from datetime import datetime
import pytz
from cgnal.core.utils.dict import union


__this_dir__, __this_filename__ = os.path.split(__file__)

path_matcher = re.compile(r"\$\{([^}^{]+)\}")


def path_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: Node
) -> PathLike:
    """
    Extract the matched value, expand env variable, and replace the match
    """
    value = node.value
    match = path_matcher.match(value)

    if match is None:
        raise SyntaxError("Can't match pattern")

    env_var = match.group()[2:-1]
    return os.environ.get(env_var) + value[match.end() :]


# define custom tag handler
def joinPath(loader: Union[Loader, FullLoader, UnsafeLoader], node: Node) -> PathLike:
    seq = loader.construct_sequence(node)
    return os.path.join(*seq)


# register tag handlers
add_implicit_resolver("!path", path_matcher)
add_constructor("!path", path_constructor)
add_constructor("!joinPath", joinPath)


def load(filename: PathLike) -> Configuration:
    """
    Load configuration reading given filename

    :param filename: file to read
    :return: loaded configuration
    """
    try:
        return cfg_load.load(filename, safe_load=False, Loader=Loader)
    except TypeError:
        return cfg_load.load(filename)


def get_all_configuration_file(
    application_file: PathLike = "application.yml", name_env: str = "CONFIG_FILE"
) -> Sequence[str]:
    """
    Retrieve all configuration files from system path, including the one in environment variable

    :param application_file: name of the configuration file to retrieve
    :param name_env: environment variable specifying the path to a specific configuration file
    :return: list of retrieved paths
    """
    confs = [
        os.path.join(path, application_file)
        for path in sys.path
        if os.path.exists(os.path.join(path, application_file))
    ]
    env = [] if name_env not in os.environ.keys() else os.environ[name_env].split(":")
    print(f"Using Configuration files: {', '.join(confs + env)}")
    return confs + env


def merge_confs(
    filenames: Sequence[PathLike], default: Optional[str] = "defaults.yml"
) -> Configuration:
    """
    Merge configurations in given files

    :param filenames: files to merge
    :param default: default configurations
    :return: merged configuration
    """
    lst = [default, *filenames] if default is not None else filenames
    print(f"Using Default Configuration file: {lst[0]}")
    return reduce(lambda config, fil: config.update(load(fil)), lst[1:], load(lst[0]))


class BaseConfig(object):
    def __init__(self, config: Configuration):
        self.config = config

    def sublevel(self, name: Hashable) -> Configuration:
        return Configuration(
            self.config[name], self.config.meta, self.config.meta["load_remote"]
        )

    def getValue(self, name: Hashable) -> Any:
        return self.config[name]

    def safeGetValue(self, name: Hashable) -> Any:
        return self.config.get(name, None)

    def update(self, my_dict: dict) -> "BaseConfig":

        meta = union(
            self.config.meta,
            {
                "updated_params": my_dict,
                "modification_datetime": datetime.now().astimezone(
                    tz=pytz.timezone("Europe/Rome")
                ),
            },
        )
        return type(self)(Configuration(union(dict(self.config), my_dict), meta))


class FileSystemConfig(BaseConfig):
    @property
    def root(self) -> PathLike:
        return self.getValue("root")

    def getFolder(self, path: Hashable) -> PathLike:
        return self.config["folders"][path]

    def getFile(self, file: Hashable) -> PathLike:
        return self.config["files"][file]


class AuthConfig(BaseConfig):
    @property
    def method(self) -> str:
        return self.getValue("method")

    @property
    def filename(self) -> PathLike:
        return self.getValue("filename")

    @property
    def user(self) -> str:
        return self.getValue("user")

    @property
    def password(self) -> str:
        return self.getValue("password")


class AuthService(BaseConfig):
    @property
    def url(self) -> str:
        return self.getValue("url")

    @property
    def check(self) -> str:
        return self.getValue("check")

    @property
    def decode(self) -> str:
        return self.getValue("decode")


class CheckService(BaseConfig):
    @property
    def url(self) -> str:
        return self.getValue("url")

    @property
    def login(self) -> str:
        return self.getValue("login")

    @property
    def logout(self) -> str:
        return self.getValue("logout")


class AuthenticationServiceConfig(BaseConfig):
    @property
    def secured(self) -> bool:
        return self.getValue("secured")

    @property
    def ap_name(self) -> str:
        return self.getValue("ap_name")

    @property
    def jwt_free_endpoints(self) -> List[str]:
        return self.getValue("jwt_free_endpoints")

    @property
    def auth_service(self) -> AuthService:
        return AuthService(self.sublevel("auth_service"))

    @property
    def check_service(self) -> CheckService:
        return CheckService(self.sublevel("check_service"))

    @property
    def cors(self) -> str:
        return self.getValue("cors")
