import os

from setuptools.command.install import install

from tensorlink.utils.rsa import generate_rsa_key_pair


class CustomInstallCommand(install):
    def run(self):
        install.run(self)

        # Only create files if in a virtual environment
        if os.getenv("VIRTUAL_ENV"):
            generate_rsa_key_pair(5025)
        else:
            print("Not in a virtual environment. Skipping custom installation steps.")
