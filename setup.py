"""Setup configuration for violence detection package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="violence-detection",
    version="1.0.0",
    author="Your Team",
    author_email="your.email@example.com",
    description="Real-time violence detection in video using VGG19 + LSTM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/violence-detection",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
    ],
    entry_points={
        "console_scripts": [
            "detect-violence=src.inference.detect:main",
        ],
    },
)
