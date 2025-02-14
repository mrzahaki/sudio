# SUDIO - Audio Processing Platform
# Copyright (C) 2024 Hossein Zahaki

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
#  any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# - GitHub: https://github.com/MrZahaki/sudio


import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy
import shutil


CMAKE_PLATFORMS = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}

class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())

class CythonExtension(Extension):
    pass

class CustomBuildExt(build_ext):
    
    def build_extension(self, ext):
        if isinstance(ext, CMakeExtension):
            return self.build_cmake_extension(ext)
        elif isinstance(ext, CythonExtension) or isinstance(ext, Extension):
            return super().build_extension(ext)
        else:
            raise ValueError(f"Unknown extension type: {type(ext)}")
    
    def build_cmake_extension(self, ext):
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DPACKAGE_VERSION_INFO={self.distribution.get_version()}",
        ]
        build_args = []

        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        if self.compiler.compiler_type != "msvc":
            self._setup_unix_build(cmake_generator, cmake_args)
        else:
            self._setup_windows_build(cmake_generator, cmake_args, build_args, cfg, extdir)

        self._setup_cross_platform_args(build_args, cmake_args)

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        self._run_cmake_build(ext, cmake_args, build_args, build_temp)

    def _setup_unix_build(self, cmake_generator, cmake_args):
        if not cmake_generator or cmake_generator == "Ninja":
            try:
                import ninja
                ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                cmake_args.extend([
                    "-GNinja",
                    f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                ])
            except ImportError:
                pass
        cmake_args.extend(["-DCMAKE_POSITION_INDEPENDENT_CODE=ON", "-DCMAKE_BUILD_TYPE=Release"])

    def _setup_windows_build(self, cmake_generator, cmake_args, build_args, cfg, extdir):
        single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})
        contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

        if not single_config and not contains_arch:
            cmake_args.append(f"-A{CMAKE_PLATFORMS[self.plat_name]}")

        if not single_config:
            cmake_args.append(f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}")
            build_args.extend(["--config", cfg])

    def _setup_cross_platform_args(self, build_args, cmake_args):
        if "universal2" in self.plat_name:
            cmake_args.append("-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64")

        if self.plat_name.startswith("macosx-") and "MACOSX_DEPLOYMENT_TARGET" not in os.environ:
            os.environ["MACOSX_DEPLOYMENT_TARGET"] = self.plat_name.split("-")[1]

        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            if hasattr(self, "parallel") and self.parallel:
                build_args.append(f"-j{self.parallel}")

        output_dir = Path(self.build_lib) / 'sudio'
        cmake_args.extend([
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output_dir}",
            f"-DCMAKE_INSTALL_LIBDIR={output_dir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={output_dir}"
        ])

    def _run_cmake_build(self, ext, cmake_args, build_args, build_temp):
        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args, "-Wno-dev", "--log-level", "NOTICE" ], cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )

        output_dir = Path(self.build_lib) / 'sudio'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for so_file in build_temp.glob('**/*.so'):
            dest = output_dir / so_file.name
            shutil.copy2(so_file, dest)
            print(f"Copied {so_file} to {dest}")


if sys.platform.startswith('win'):
    extra_link_args = []
else:
    extra_link_args = ['-lm']


numpy_include = numpy.get_include()


cython_extensions = [
    Extension(
    "sudio.process.fx._tempo",
    ["sudio/process/fx/_tempo.pyx"],
    include_dirs=[numpy_include],
    extra_link_args=extra_link_args,
    extra_compile_args=["-O3"], 
    ),
    Extension(
    "sudio.process.fx._fade_envelope",
    ["sudio/process/fx/_fade_envelope.pyx"],
    extra_link_args=extra_link_args,
    include_dirs=[numpy_include],
    extra_compile_args=["-O3"], 
    ),
    Extension(
    "sudio.process.fx._channel_mixer",
    ["sudio/process/fx/_channel_mixer.pyx"],
    extra_link_args=extra_link_args,
    include_dirs=[numpy_include],
    extra_compile_args=["-O3"], 
    ),
    Extension(
    "sudio.process.fx._pitch_shifter",
    ["sudio/process/fx/_pitch_shifter.pyx"],
    extra_link_args=extra_link_args,
    include_dirs=[numpy_include],
    extra_compile_args=["-O3"], 
    ),
    Extension(
    "sudio.utils.math",
    ["sudio/utils/math.pyx"],
    extra_link_args=extra_link_args,
    include_dirs=[numpy_include], 
    extra_compile_args=['-O3'],
    language='c'
    ),
    Extension(
    "sudio.generator.tone.sinewave",
    ["sudio/generator/tone/sinewave.pyx"],
    extra_link_args=extra_link_args,
    include_dirs=[
        numpy_include,
        "sudio/generator/tone"
        ], 
    extra_compile_args=['-O3'],
    language='c++'
    )
]

cmake_extensions = [
    CMakeExtension('sudio._rateshift', sourcedir='sudio/rateshift'), 
    CMakeExtension('sudio._suio', sourcedir='sudio/io'),
]

cythonized_extensions = cythonize(
    cython_extensions,
    compiler_directives={
        'language_level': '3',
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True,
        'nonecheck': False,
    }
)

setup(
    packages=find_packages(),
    package_dir={'': '.'},
    ext_modules=[
        *cmake_extensions,  
        *cythonized_extensions, 
    ],
    cmdclass={'build_ext': CustomBuildExt},
    zip_safe=False,
    package_data={
        "": ["*.pxd", "*.pyx"],
    },
)

