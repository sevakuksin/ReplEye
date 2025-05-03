from setuptools import setup, find_packages

setup(
    name="ReplEye",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch",
        "opencv-python",
        "numpy",
        # Don't include picamera2 unless targeting Raspberry Pi
    ]
)
