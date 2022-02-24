import os
import unittest
from logging import StreamHandler, FileHandler
from cgnal.core.logging.defaults import configFromFiles, logger
from cgnal.core.tests.core import TestCase, logTest
from tests import DATA_FOLDER, TMP_FOLDER, clean_tmp_folder, unset_TMP_FOLDER

configFromFiles(
    config_files=[os.path.join(DATA_FOLDER, "logging.yml")],
    capture_warnings=True,
    catch_exceptions="except",
)


class TestSetupLogger(TestCase):
    root_logger = logger()
    cgnal_logger = logger(name="cgnal")

    @logTest
    def test_console_logger(self):
        self.root_logger.info("Example of logging with root logger!")
        self.assertEqual(self.root_logger.name, "root")
        self.assertEqual(self.root_logger.level, 20)
        self.assertTrue(
            all([isinstance(h, StreamHandler) for h in self.root_logger.handlers])
        )

    @logTest
    def test_file_logger_name(self):
        self.assertEqual(self.cgnal_logger.name, "cgnal")

    @logTest
    def test_file_logger_handlers(self):
        self.assertTrue(
            all([isinstance(h, FileHandler) for h in self.cgnal_logger.handlers])
        )

    @logTest
    def test_file_logger_path_creation(self):
        self.assertTrue(os.path.exists(TMP_FOLDER))
        self.assertTrue(
            all([os.path.exists(h.baseFilename) for h in self.cgnal_logger.handlers])
        )

    @logTest
    def test_file_logger_overwrite_level(self):
        self.assertEqual(self.cgnal_logger.level, 20)

    @logTest
    def test_file_logger_dest_file(self):
        res = {"regular.log": 10, "errors.log": 40}
        self.assertTrue(
            all(
                [
                    h.level == res[os.path.basename(h.baseFilename)]
                    for h in self.cgnal_logger.handlers
                ]
            )
        )

    @logTest
    def test_file_logger_info_message(self):
        msg = "Example of logging with cgnal logger!"
        self.cgnal_logger.info(msg)
        self.cgnal_logger.handlers[0].flush()
        with open(self.cgnal_logger.handlers[0].baseFilename, "r") as fil:
            lines = fil.readlines()
        lin = lines[-1]
        self.assertEqual(lin.split(" - ")[-1], f"{msg}\n")
        self.assertEqual(lin.split(" - ")[-2], "INFO")

    @logTest
    def test_file_logger_warning_message(self):
        warning_msg = "Example of logging a warning with cgnal logger!"
        self.cgnal_logger.warning(warning_msg)
        self.cgnal_logger.handlers[0].flush()
        with open(self.cgnal_logger.handlers[0].baseFilename, "r") as fil:
            lines = fil.readlines()
        lin = lines[-1]
        self.assertEqual(lin.split(" - ")[-1], f"{warning_msg}\n")
        self.assertEqual(lin.split(" - ")[-2], "WARNING")

    @logTest
    def test_file_logger_error_message(self):
        error_msg = "Example of logging an error with cgnal logger!"
        self.cgnal_logger.error(error_msg)
        self.cgnal_logger.handlers[1].flush()
        with open(self.cgnal_logger.handlers[1].baseFilename, "r") as fil:
            lines = fil.readlines()
        lin = lines[-1]
        self.assertEqual(lin.split(" - ")[-1], f"{error_msg}\n")
        self.assertEqual(lin.split(" - ")[-2], "ERROR")

    # TODO: [ND] Cercare un modo di testare except_logger: so che funziona ma non riesco a fare emettere eccezioni senza
    #  interrompere l'esecuzione (e mandare in errore il test)
    # @logTest
    # def test_file_logger_catch_exceptions(self):
    #     except_logger = logger(name="except")
    #
    #     raise TypeError('Tipo Sbagliato')
    #
    #     with open(except_logger.handlers[1].baseFilename, 'r') as fil:
    #         lines = fil.readlines()
    #     lin = lines[-4:]
    #     self.assertEqual(lin[0].split(" - ")[-2], 'ERROR')
    #     self.assertEqual(lin[0].split(" - ")[-1], 'TypeError: Tipo Sbagliato\n')
    #     self.assertEqual(lin[-1], 'TypeError: Tipo Sbagliato\n')


if __name__ == "__main__":
    unittest.main()
    unset_TMP_FOLDER()
    clean_tmp_folder()
