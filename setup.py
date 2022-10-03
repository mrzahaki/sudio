import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='sudio',
    version='1.0.9',
    packages=['sudio',],
    author="hussein zahaki",
    author_email="hussein.zahaki.mansoor@gmail.com",
    description="Real-Time Audio Processing Platform",
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
