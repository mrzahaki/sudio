import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='sudio',
    version='1.0.9.3',
    packages=['sudio', 'sudio.extras'],
    author="mrzahaki",
    author_email="mrzahaki@gmail.com",
    description="Audio Processing Platform",
    long_description=long_description,
    install_requires=[
            'scipy>=1.9.1',
            'numpy>=1.23.3',
            'pyaudio>=0.2.12',
            'pandas>=1.5.0',
            'miniaudio>=1.52',
            'samplerate>=0.1.0',
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
