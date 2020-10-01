from setuptools import find_packages, setup


setup(
    name="fastocr",
    version="0.0.1",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
    ],
    python_requires=">=3.6.0",
)