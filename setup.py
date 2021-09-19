import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='sudio',
    version='1.0.8',
    packages=['sudio',],
    author="hussein zahaki",
    author_email="hussein.zahaki.mansoor@gmail.com",
    description="Real-Time Audio Processing Platform",
    long_description=long_description,
    install_requires=[
            'scipy',
            'tqdm',
            'numpy',
            'pyaudio',
            'pandas',
        ],
    long_description_content_type="text/markdown",
    url="https://github.com/MrZahaki/sudio",
    license='Apache License 2.0',
    classifiers=[
     "Programming Language :: Python :: 3",
     'License :: OSI Approved :: Apache Software License',
    ],
    license_files=['LICENSE'],
 )
