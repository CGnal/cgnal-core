"""Module for Mongo persistance layer."""
from typing import Any
from cgnal.core.config import BaseConfig, AuthConfig


# TODO: Are we sure this is the best place for this class? Wouldn't it be better to place it in the config module?
class MongoConfig(BaseConfig):
    """Configuration for a Mongo DB."""

    @property
    def host(self) -> str:
        """
        Return ost name.

        :return: host name
        """
        return self.getValue("host")

    @property
    def port(self) -> int:
        """
        Return port.

        :return: port
        """
        return self.getValue("port")

    @property
    def db_name(self) -> str:
        """
        Return database name.

        :return: database name
        """
        return self.getValue("db_name")

    def getCollection(self, name) -> str:
        """
        Return collection name at a given configuration node.

        :param name: configuration node name
        :return: collection name
        """
        return self.config["collections"][name]

    @property
    def auth(self) -> AuthConfig:
        """
        Return authetication config.

        :return: authetication config
        """
        return AuthConfig(self.sublevel("auth"))

    @property
    def admin(self) -> AuthConfig:
        """
        Return administrator authentication config.

        :return: administrator authentication config
        """
        return AuthConfig(self.sublevel("admin"))

    @property
    def authSource(self) -> Any:
        """
        Return the authentication source.

        :return: authentication source
        """
        return self.safeGetValue("authSource")
