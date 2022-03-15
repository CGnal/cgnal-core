import unittest
import os

from cgnal.core.logging import LoggingConfig
from cgnal.core.logging.defaults import getDefaultLogger, configFromFiles
from tests import DATA_FOLDER
from cgnal.core.tests.core import TestCase, logTest
from cgnal.core.config import (
    get_all_configuration_file,
    __this_dir__ as config_dir,
    merge_confs,
    BaseConfig,
    FileSystemConfig,
    AuthConfig,
    AuthenticationServiceConfig,
)
from cgnal.core.data.layer.mongo import MongoConfig

os.environ["USER"] = os.environ.get("USER", "cgnal")
TEST_DATA_PATH = DATA_FOLDER
logger = getDefaultLogger()


class TempConfig(BaseConfig):
    @property
    def logging(self):
        return LoggingConfig(self.sublevel("logging"))

    @property
    def fs(self):
        return FileSystemConfig(self.sublevel("fs"))

    @property
    def auth(self):
        return AuthConfig(self.sublevel("auth"))

    @property
    def authentication(self):
        return AuthenticationServiceConfig(self.sublevel("authentication"))

    @property
    def mongo(self):
        return MongoConfig(self.sublevel("mongo"))


test_file = "defaults.yml"
os.environ["CONFIG_FILE"] = os.path.join(TEST_DATA_PATH, test_file)

root = os.path.join("this", "is", "a", "folder")
credentials = os.path.join(root, "myfolder", "credentials.p")

config = TempConfig(
    BaseConfig(
        merge_confs(
            get_all_configuration_file(), os.path.join(config_dir, "defaults.yml")
        )
    ).sublevel("test")
)


class TestLoggingConfig(TestCase):
    @logTest
    def test_level(self):
        logger.info(f"Logging: {config.logging.level}")
        self.assertEqual(config.logging.level, "DEBUG")

    @logTest
    def test_filename(self):
        self.assertEqual(config.logging.filename, os.path.join("logs", "tests.log"))

    @logTest
    def test_default_config_file(self):
        self.assertEqual(
            config.logging.default_config_file,
            os.path.join("confs", "logConfDefaults.yaml"),
        )

    @logTest
    def test_capture_warnings(self):
        self.assertTrue(config.logging.capture_warnings)


class TestBaseConfig(TestCase):

    # todo: differenza tra config.getValue("fs")["root"] e config.fs.root ??
    @logTest
    def test_sublevel(self):
        self.assertEqual(
            config.sublevel("fs").to_dict(),
            {
                "root": root,
                "folders": {"python": "myfolder"},
                "files": {"credentials": credentials},
            },
        )

    @logTest
    def test_getValue(self):
        self.assertEqual(config.getValue("fs")["root"], root)
        self.assertRaises(KeyError, config.getValue, "folders")

    @logTest
    def test_safeGetValue(self):
        self.assertEqual(config.safeGetValue("fs")["root"], root)
        self.assertIsNone(config.safeGetValue("folders"))

    @logTest
    def test_update(self):

        new_config = config.update({"test": {"fs": {"root": "new_folder"}}})

        self.assertEqual(new_config.getValue("test")["fs"]["root"], "new_folder")
        self.assertEqual(
            new_config.config.meta["updated_params"],
            {"test": {"fs": {"root": "new_folder"}}},
        )


class TestFileSystemConfig(TestCase):
    @logTest
    def test_root(self):
        self.assertEqual(config.fs.root, root)

    @logTest
    def test_getFolder(self):
        self.assertEqual(config.fs.getFolder("python"), "myfolder")

    @logTest
    def test_getFile(self):
        logger.info(f"Get File: {config.fs.getFile('credentials')}")
        self.assertEqual(config.fs.getFile("credentials"), credentials)


class TestAuthConfig(TestCase):
    @logTest
    def test_method(self):
        self.assertEqual(config.auth.method, "file")

    @logTest
    def test_filename(self):
        self.assertEqual(config.auth.filename, credentials)

    @logTest
    def test_user(self):
        self.assertEqual(config.auth.user, "userID")

    @logTest
    def test_password(self):
        self.assertEqual(config.auth.password, "passwordID")


class TestAuthenticationServiceConfig(TestCase):
    @logTest
    def test_secured(self):
        self.assertTrue(config.authentication.secured, "passwordID")

    @logTest
    def test_ap_name(self):
        self.assertEqual(config.authentication.ap_name, "cb")

    @logTest
    def test_cors(self):
        self.assertEqual(config.authentication.cors, "http://0.0.0.0:10001")

    @logTest
    def test_jwt_free_endpoints(self):
        self.assertEqual(
            config.authentication.jwt_free_endpoints,
            [
                "/api/v1/health/",
                "/api/v1/auth/login",
                "/api/v1/apidocs",
                "/api/v1/swagger.json",
                "/api/v1/salesforce/",
                "/api/v1/openBanking/",
            ],
        )

    @logTest
    def test_auth_service(self):
        self.assertEqual(config.authentication.auth_service.url, "http://0.0.0.0:10005")
        self.assertEqual(
            config.authentication.auth_service.check, "/tokens/{tok}/check"
        )
        self.assertEqual(
            config.authentication.auth_service.decode, "/tokens/{tok}/decode"
        )

    @logTest
    def test_check_service(self):
        self.assertEqual(
            config.authentication.check_service.url, "http://0.0.0.0:10001"
        )
        self.assertEqual(
            config.authentication.check_service.login, "/authentication/login"
        )
        self.assertEqual(
            config.authentication.check_service.logout, "/authentication/logout"
        )


class TestMongoConfig(TestCase):
    @logTest
    def test_host(self):
        self.assertEqual(config.mongo.host, "0.0.0.0")

    @logTest
    def test_port(self):
        self.assertEqual(config.mongo.port, 202020)

    @logTest
    def test_db_name(self):
        self.assertEqual(config.mongo.db_name, "database")

    @logTest
    def test_getCollection(self):
        self.assertEqual(config.mongo.getCollection("coll_name"), "coll_name")

    @logTest
    def test_auth(self):
        self.assertEqual(config.mongo.auth.method, "file")
        self.assertEqual(
            config.mongo.auth.filename,
            os.path.join(root, "myfolder", "credentials.auth.p"),
        )
        self.assertEqual(config.mongo.auth.user, "mongo.auth.db_user")
        self.assertEqual(config.mongo.auth.password, "mongo.auth.db_psswd")

    @logTest
    def test_admin(self):
        self.assertEqual(config.mongo.admin.method, "file")
        self.assertEqual(
            config.mongo.admin.filename,
            os.path.join(root, "myfolder", "credentials.admin.p"),
        )
        self.assertEqual(config.mongo.admin.user, "mongo.admin.db_user")
        self.assertEqual(config.mongo.admin.password, "mongo.admin.db_psswd")

    @logTest
    def test_authSource(self):
        self.assertEqual(config.mongo.authSource, "source")


class TestDocumentArchivers(TestCase):
    @logTest
    def test_read_logging_config(self):
        config_file = "logging.yml"

        configFromFiles([os.path.join(TEST_DATA_PATH, config_file)])

        logger.info("Example of logging!")

    @logTest
    def test_environ_variable(self):
        test_file = "defaults.yml"

        os.environ["CONFIG_FILE"] = os.path.join(TEST_DATA_PATH, test_file)

        config = TempConfig(
            BaseConfig(
                merge_confs(
                    get_all_configuration_file(),
                    os.path.join(config_dir, "defaults.yml"),
                )
            ).sublevel("test")
        )

        user = config.getValue("user")
        self.assertEqual(user, os.environ["USER"])


if __name__ == "__main__":
    unittest.main()
