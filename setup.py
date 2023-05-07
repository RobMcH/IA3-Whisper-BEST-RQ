import site
import sys

from setuptools import setup

site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

setup(
    name="ia3_whisper",
    version="0.0.1",
    description="A python package to train Whisper-like models using the BEST-RQ objective and"
    " IA3 parameter efficient fine-tuning.",
    author="Robert McHardy",
    packages=["ia3_whisper"],
    entry_points={"console_scripts": ["train=ia3_whisper.train:main"]},
)
