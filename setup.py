from setuptools import find_packages, setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name="2pm-fhe",
    version="0.0.1",
    author="2PM.Network",
    author_email="zzh@2pm.network",
    description="FHE library for 2PM.Network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/2PM-Network/2pm-fhe-sdk",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)