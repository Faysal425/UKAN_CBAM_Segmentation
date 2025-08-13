"""Setup script for UKAN-CBAM."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ukan-cbam",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="U-shaped Kolmogorov-Arnold Network with CBAM for binary segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ukan-cbam",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ukan-train=ukan_cbam.train:main",
            "ukan-eval=ukan_cbam.evaluate:main",
            "ukan-predict=ukan_cbam.predict:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
