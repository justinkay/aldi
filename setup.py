from setuptools import setup, find_packages

setup(
    name='aldi',
    version='0.1.1',
    author='justinkay',
    url="https://github.com/justinkay/aldi",
    description="Align and distill (ALDI): A unified framework for domain adaptive object detection",
    python_requires=">=3.7",
    packages=find_packages(include=['aldi', 'aldi.*']),
)