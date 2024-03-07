import pkg_resources
from setuptools import setup, find_packages

# dependencies that must be installed ahead of time
required = ['torch', 'torchvision', 'detectron2'] 

def check_dependencies():
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = set(required) - installed

    if missing:
        raise ValueError(f"Missing required dependencies: {', '.join(missing)}. You must install these yourself; see README.md.")

# Call the function to check dependencies
check_dependencies()

setup(
    name='aldi',
    version='0.1.1',
    author='anonymous',
    url="https://github.com/justinkay/aldi",
    description="Align and distill (ALDI): A unified framework for domain adaptive object detection",
    packages=find_packages(include=['aldi', 'aldi.*']),
    install_requires=[
        'pillow==9.5.0', # see: https://github.com/facebookresearch/detectron2/issues/5010
        'opencv-python',
        'scipy',
    ]
)
