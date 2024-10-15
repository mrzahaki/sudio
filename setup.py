import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

# CMake platform mapping
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

class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
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

    def _run_cmake_build(self, ext, cmake_args, build_args, build_temp):
        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )

setup(
    packages=find_packages(),
    package_dir={'': '.'},
    ext_modules=[CMakeExtension('sudio.rateshift'), CMakeExtension('sudio.suio')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
    package_data={
        "": ["*.pyi"],
    },
)
