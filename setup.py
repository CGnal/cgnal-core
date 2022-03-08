import os
import setuptools
from versioneer import get_version, get_cmdclass

with open(os.path.join("requirements", "requirements.in"), "r") as fid:
    requirements = [
        line.replace("\n", "")
        for line in fid.readlines()
        if not line.strip(" ").startswith("#")
    ]

setuptools.setup(
    version=get_version(),
    cmdclass=get_cmdclass(),
    install_requires=requirements,
)
