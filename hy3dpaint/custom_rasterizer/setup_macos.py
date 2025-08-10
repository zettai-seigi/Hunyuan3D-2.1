# macOS setup for custom_rasterizer (CPU fallback only)
from setuptools import setup, find_packages

setup(
    packages=find_packages(),
    version="0.1",
    name="custom_rasterizer",
    include_package_data=True,
    package_dir={"": "."},
    description="CPU fallback rasterizer for macOS/MPS systems",
    install_requires=[
        "torch",
        "numpy",
    ],
)