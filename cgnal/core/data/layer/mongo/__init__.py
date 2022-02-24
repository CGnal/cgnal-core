from typing import Any
from cgnal.core.config import BaseConfig, AuthConfig


# TODO: Are we sure this is the best place for this class? Wouldn't it be better to place it in the config module?
class MongoConfig(BaseConfig):
    @property
    def host(self) -> str:
        return self.getValue("host")

    @property
    def port(self) -> int:
        return self.getValue("port")

    @property
    def db_name(self) -> str:
        return self.getValue("db_name")

    def getCollection(self, name) -> str:
        return self.config["collections"][name]

    @property
    def auth(self) -> AuthConfig:
        return AuthConfig(self.sublevel("auth"))

    @property
    def admin(self) -> AuthConfig:
        return AuthConfig(self.sublevel("admin"))

    @property
    def authSource(self) -> Any:
        return self.safeGetValue("authSource")
