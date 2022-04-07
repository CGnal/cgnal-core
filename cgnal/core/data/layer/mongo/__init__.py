"""Module for Mongo persistance layer."""
from typing import Any
from cgnal.core.config import BaseConfig, AuthConfig


# TODO: Are we sure this is the best place for this class? Wouldn't it be better to place it in the config module?
class MongoConfig(BaseConfig):
    """Configuration for a Mongo DB."""

    @property
    def host(self) -> str:
        """Host name."""
        return self.getValue("host")

    @property
    def port(self) -> int:
        """Port."""
        return self.getValue("port")

    @property
    def db_name(self) -> str:
        """Database name."""
        return self.getValue("db_name")

    def getCollection(self, name) -> str:
        """
        Return collection name at a given configuration node.

        :param name: configuration node name
        :return str: collection name
        """
        return self.config["collections"][name]

    @property
    def auth(self) -> AuthConfig:
        """Authetication config."""
        return AuthConfig(self.sublevel("auth"))

    @property
    def admin(self) -> AuthConfig:
        """Administrator authentication config."""
        return AuthConfig(self.sublevel("admin"))

    @property
    def authSource(self) -> Any:
        """Return the authentication source."""
        return self.safeGetValue("authSource")
