"""
Setup script for Amir Sign Language School
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="amir-sign-language-school",
    version="1.0.0",
    author="Mohamed Hassan Jibril",
    author_email="m.h.jibreel@gmail.com",
    description="A comprehensive Python Flask web application for learning ASL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mjibreel/sign-language-school",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.11",
    install_requires=[
        "Flask>=2.0.1",
        "opencv-python>=4.12.0",
        "mediapipe>=0.10.14",
        "pandas>=2.3.1",
        "numpy>=2.2.6",
        "scipy>=1.16.1",
        "matplotlib>=3.10.5",
        "protobuf>=4.25.8"
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
        "docs": ["sphinx", "sphinx-rtd-theme"]
    },
    entry_points={
        "console_scripts": [
            "amir-sign-language=app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
