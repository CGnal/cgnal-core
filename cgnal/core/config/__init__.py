"""Implementation of classes that parse configuration files."""
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
    Extract the matched value, expand env variable, and replace the match.

    :param loader: not used
    :param node: YAML node
    :return: path
    :raises SyntaxError: if the node value does not match the regex expression for a path-like string
    :raises KeyError: raises an exception if the environment variable is missing
    """
    value = node.value
    match = path_matcher.match(value)

    if match is None:
        raise SyntaxError("Can't match pattern")

    env_var = match.group()[2:-1]
    try:
        return os.environ[env_var] + value[match.end() :]
    except KeyError:
        raise KeyError(
            f"Missing definition of environment variable {env_var} "
            f"needed when parsing configuration file"
        )


def joinPath(loader: Union[Loader, FullLoader, UnsafeLoader], node: Node) -> PathLike:
    """
    Join pieces of a file system path. Can be used as a custom tag handler.

    :param loader: YAML file loader
    :param node: YAML node
    :return: path
    """
    seq = loader.construct_sequence(node)
    return os.path.join(*seq)


# register tag handlers
add_implicit_resolver("!path", path_matcher)
add_constructor("!path", path_constructor)
add_constructor("!joinPath", joinPath)


def load(filename: PathLike) -> Configuration:
    """
    Load configuration reading given filename.

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
    Retrieve all configuration files from system path, including the one in environment variable.

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
    Merge configurations in given files.

    :param filenames: files to merge
    :param default: default configurations
    :return: merged configuration
    """
    lst = [default, *filenames] if default is not None else filenames
    print(f"Using Default Configuration file: {lst[0]}")
    return reduce(lambda config, fil: config.update(load(fil)), lst[1:], load(lst[0]))


class BaseConfig(object):
    """Base configuration class."""

    def __init__(self, config: Configuration):
        """
        Class instance initializer.

        :param config: configuration
        """
        self.config = config

    def sublevel(self, name: Hashable) -> Configuration:
        """
        Return a sublevel of the main configuration.

        :param name: name of the sublevel
        :return: configuration of the sublevel
        """
        return Configuration(
            self.config[name], self.config.meta, self.config.meta["load_remote"]
        )

    def getValue(self, name: Hashable) -> Any:
        """
        Get the value of a configuration node.

        :param name: name of the configuration node
        :return: value of the configuratio node
        """
        return self.config[name]

    def safeGetValue(self, name: Hashable) -> Any:
        """
        Get the value of a configuration node, gracefully returning None if the node does not exist.

        :param name: name of the node
        :return: value of the node, or None if the node does not exist
        """
        return self.config.get(name, None)

    def update(self, my_dict: dict) -> "BaseConfig":
        """
        Update the current configuration.

        :param my_dict: dictionary containing the nodes of the configuration to be updated
        :return: new configuration with the updated nodes
        """
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
    """Configuration for file system paths."""

    @property
    def root(self) -> PathLike:
        """
        Return the root node value.

        :return: root node value
        """
        return self.getValue("root")

    def getFolder(self, path: Hashable) -> PathLike:
        """
        Return the folder name.

        :param path: name of the configuration node
        :return: folder name
        """
        return self.config["folders"][path]

    def getFile(self, file: Hashable) -> PathLike:
        """
        Return the file name.

        :param file: name of the configuration node
        :return: file name
        """
        return self.config["files"][file]


class AuthConfig(BaseConfig):
    """Authetication configuration."""

    @property
    def method(self) -> str:
        """
        Return the authentication method.

        :return: authentication method
        """
        return self.getValue("method")

    @property
    def filename(self) -> PathLike:
        """
        Return the name of the file containing the authentication details.

        :return: name of the file containing the authentication details
        """
        return self.getValue("filename")

    @property
    def user(self) -> str:
        """
        Return the user name.

        :return: user name
        """
        return self.getValue("user")

    @property
    def password(self) -> str:
        """
        Return the password.

        :return: password
        """
        return self.getValue("password")


class AuthService(BaseConfig):
    """Configuration for the authentication data."""

    @property
    def url(self) -> str:
        """
        Return the url of the authentication service.

        :return: url of the authentication service
        """
        return self.getValue("url")

    @property
    def check(self) -> str:
        """
        Return check.

        :return: check
        """
        return self.getValue("check")

    @property
    def decode(self) -> str:
        """
        Return decode.

        :return: decode
        """
        return self.getValue("decode")


class CheckService(BaseConfig):
    """Configuration for the check service."""

    @property
    def url(self) -> str:
        """
        Return the url of the check service.

        :return: url of the check service.
        """
        return self.getValue("url")

    @property
    def login(self) -> str:
        """
        Return the login url.

        :return: login url
        """
        return self.getValue("login")

    @property
    def logout(self) -> str:
        """
        Return the logout url.

        :return: logout url
        """
        return self.getValue("logout")


class AuthenticationServiceConfig(BaseConfig):
    """Configuration of the authentication service."""

    @property
    def secured(self) -> bool:
        """
        Return the secured flag.

        :return: secured flag
        """
        return self.getValue("secured")

    @property
    def ap_name(self) -> str:
        """
        Return the ap name.

        :return: ap name
        """
        return self.getValue("ap_name")

    @property
    def jwt_free_endpoints(self) -> List[str]:
        """
        Return the jwt free endpoints.

        :return: jwt free endpoints
        """
        return self.getValue("jwt_free_endpoints")

    @property
    def auth_service(self) -> AuthService:
        """
        Return the authentication data.

        :return: authentication data
        """
        return AuthService(self.sublevel("auth_service"))

    @property
    def check_service(self) -> CheckService:
        """
        Return the check service configuration.

        :return: check service configuration
        """
        return CheckService(self.sublevel("check_service"))

    @property
    def cors(self) -> str:
        """
        Return the cors.

        :return: cors
        """
        return self.getValue("cors")
