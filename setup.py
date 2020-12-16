import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="colocationship",
    version="0.0.1",
    author="Zexun Chen",
    author_email="sxtpy2010@gmail.com",
    description="A light package for build and analyse co-locationship",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Magica-Chen/co-locationship",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)