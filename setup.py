from setuptools import setup, find_packages

setup(
    name="text-to-3d-evaluation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "Pillow",
        "pandas",
        "tqdm",
        "jaxtyping",
        "einops",
        "pytest",
    ],
    python_requires=">=3.7",
    author="Phuc Lai",
    description="Evaluation code for Text-to-3D models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 