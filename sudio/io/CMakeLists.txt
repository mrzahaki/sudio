# SUDIO - Audio Processing Platform
# Copyright (C) 2024 Hossein Zahaki

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
#  any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# - GitHub: https://github.com/MrZahaki/sudio


cmake_minimum_required(VERSION 3.20)
project(sudio_io LANGUAGES CXX C)
include(FetchContent)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_SUPPRESS_REGENERATION ON)
set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)
set(CMAKE_RULE_MESSAGES OFF)
set(CMAKE_TEST_OUTPUT OFF)
set(CMAKE_INSTALL_MESSAGE NEVER)


enable_language(C)
enable_language(CXX)


if (POLICY CMP0094)
    cmake_policy(SET CMP0094 NEW)
endif()

if (NOT DEFINED Python_FIND_REGISTRY)
    set(Python_FIND_REGISTRY "LAST")
endif()
if (NOT DEFINED Python_FIND_FRAMEWORK)
    set(Python_FIND_FRAMEWORK "LAST")
endif()

function(apply_target_compile_options target)
    if(MSVC)
        target_compile_options(${target} PRIVATE 
            /W0                 
            /EHsc               
            /MP                 
            /bigobj             
        )
    else()
        target_compile_options(${target} PRIVATE 
            -O3                 
            -fPIC               
        )
    endif()
endfunction()


find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

message(STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")
message(STATUS "Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
message(STATUS "Python_LIBRARIES: ${Python_LIBRARIES}")

list(PREPEND CMAKE_PREFIX_PATH ${Python_PREFIX})


if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.cache/pybind11")
    message(STATUS "Fetching pybind11...")
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
else()
    message(STATUS "Using existing pybind11 in ${CMAKE_CURRENT_SOURCE_DIR}/.cache/pybind11")
    add_subdirectory(".cache/pybind11")
endif()


if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/miniaudio")
    message(STATUS "Fetching miniaudio...")
    FetchContent_Declare(
        miniaudio
        GIT_REPOSITORY https://github.com/mackron/miniaudio.git
        GIT_TAG 0.11.21
    )
    FetchContent_MakeAvailable(miniaudio)
    list(APPEND CMAKE_MODULE_PATH ${miniaudio_SOURCE_DIR})
    set(MINIAUDIO_SOURCE_DIR "${miniaudio_SOURCE_DIR}/extras/miniaudio_split")
else()
    message(STATUS "Using existing MiniAudio in /miniaudio")
    set(MINIAUDIO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/miniaudio/extras/miniaudio_split")
endif()


set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build PortAudio as a static library")
set(PA_DISABLE_INSTALL ON CACHE BOOL "Disable installation of PortAudio")
set(PA_BUILD_TESTS  OFF CACHE BOOL  "Include test projects")
set(PA_BUILD_EXAMPLES  OFF CACHE BOOL  "Include test projects")

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/portaudio")
    message(STATUS "Fetching portaudio...")
    FetchContent_Declare(
        portaudio
        GIT_REPOSITORY https://github.com/PortAudio/portaudio.git
        GIT_TAG cf218ed8e3085ac3731106d3636c3c6396ec2d82
    )
    FetchContent_MakeAvailable(portaudio)
else()
    message(STATUS "Using existing portaudio in ${CMAKE_CURRENT_SOURCE_DIR}/portaudio")
    add_subdirectory(portaudio)
endif()


if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/libmp3lame-CMAKE")
    message(STATUS "Fetching LAME...")
    FetchContent_Declare(
        libmp3lame-CMAKE
        GIT_REPOSITORY https://github.com/Iunusov/libmp3lame-CMAKE.git
        GIT_TAG bb770fb6e4b4dfc963860380a5e7765c370aaef1
    )
    FetchContent_MakeAvailable(libmp3lame-CMAKE)
else()
    message(STATUS "Using existing LAME in ${CMAKE_CURRENT_SOURCE_DIR}/lame")
    add_subdirectory(libmp3lame-CMAKE)
endif()



set(BUILD_TESTING OFF CACHE BOOL "Disable Ogg tests" FORCE)
set(INSTALL_DOCS OFF CACHE BOOL "Disable Ogg documentation" FORCE)
set(INSTALL_PKG_CONFIG_MODULE OFF CACHE BOOL "Disable Ogg Installation of ogg.pc file" FORCE)
set(INSTALL_CMAKE_PACKAGE_MODULE OFF CACHE BOOL "Disable Ogg Installation of CMake package configuration module" FORCE)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ogg")
    message(STATUS "Fetching Ogg...")
    FetchContent_Declare(
        ogg
        GIT_REPOSITORY https://github.com/xiph/ogg.git
        GIT_TAG v1.3.5
    )
    FetchContent_MakeAvailable(ogg)
else()
    message(STATUS "Using existing Ogg in ${CMAKE_CURRENT_SOURCE_DIR}/ogg")
    add_subdirectory(ogg)
endif()


set(BUILD_TESTING OFF CACHE BOOL "Disable FLAC tests" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "Disable FLAC examples" FORCE)
set(BUILD_DOCS OFF CACHE BOOL "Disable FLAC documentation" FORCE)
set(BUILD_CXXLIBS OFF CACHE BOOL "Disable FLAC C++ libraries" FORCE)
set(BUILD_PROGRAMS OFF CACHE BOOL "Disable FLAC programs" FORCE)
set(INSTALL_MANPAGES OFF CACHE BOOL "Disable FLAC MAN pages Installation" FORCE)
set(WITH_FORTIFY_SOURCE OFF CACHE BOOL "Disable protection against buffer overflows" FORCE)
set(INSTALL_CMAKE_CONFIG_MODULE OFF CACHE BOOL "Disable Install CMake package-config module" FORCE)
set(INSTALL_PKGCONFIG_MODULES OFF CACHE BOOL "Disable Install PkgConfig modules" FORCE)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/flac")
    message(STATUS "Fetching FLAC...")
    FetchContent_Declare(
        flac
        GIT_REPOSITORY https://github.com/xiph/flac.git
        GIT_TAG 1.4.3
    )
    FetchContent_MakeAvailable(flac)
else()
    message(STATUS "Using existing FLAC in ${CMAKE_CURRENT_SOURCE_DIR}/flac")
    add_subdirectory(flac)
endif()


set(BUILD_TESTING OFF CACHE BOOL "Disable Vorbis tests" FORCE)
set(BUILD_FRAMEWORK OFF CACHE BOOL "Disable Vorbis Build Framework bundle for OSX" FORCE)
set(OGG_LIBRARY ogg)
set(OGG_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ogg/include")

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vorbis")
    message(STATUS "Fetching Vorbis...")
    FetchContent_Declare(
        vorbis
        GIT_REPOSITORY https://github.com/xiph/vorbis.git
        GIT_TAG v1.3.7
    )
    FetchContent_MakeAvailable(vorbis)
else()
    message(STATUS "Using existing Vorbis in ${CMAKE_CURRENT_SOURCE_DIR}/vorbis")
    add_subdirectory(vorbis)
endif()



add_library(miniaudio STATIC ${MINIAUDIO_SOURCE_DIR}/miniaudio.c)
target_include_directories(miniaudio PUBLIC ${MINIAUDIO_SOURCE_DIR})
target_compile_definitions(
    miniaudio 
    PRIVATE 
    MINIAUDIO_IMPLEMENTATION
    )



pybind11_add_module(sudio_io
    src/codec.cpp
    src/stdstream.cpp
    src/suiobind.cpp
    src/alsa_suppressor.cpp
)

target_include_directories(sudio_io PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/inc"
    "${CMAKE_CURRENT_SOURCE_DIR}/portaudio/include"
    "${MINIAUDIO_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/libmp3lame-CMAKE/include"
    "${OGG_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/flac/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/vorbis/include"
)


if(DEFINED PACKAGE_VERSION_INFO)
    add_definitions(-DPACKAGE_VERSION_INFO="${PACKAGE_VERSION_INFO}")
endif()


if(MSVC)
    add_compile_options(
        /W0                     
        /experimental:external  
        /external:anglebrackets 
        /external:W0            
        /EHsc                   
        /MP                     
        /bigobj                 
    )
    
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /MANIFEST:NO")

else()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_options(
            -w              
            -Wno-everything 
            -O3             
            -fPIC           
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        add_compile_options(
            -w              
            -Wno-all        
            -O3             
            -fPIC           
        )
    endif()
endif()

set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "Suppress deprecation warnings" FORCE)
set(CMAKE_SUPPRESS_DEPRECATION_MESSAGES ON CACHE BOOL "Suppress deprecated declarations" FORCE)

apply_target_compile_options(sudio_io)


set_target_properties(sudio_io
    PROPERTIES
        PREFIX ""
        OUTPUT_NAME "_suio"
        LINKER_LANGUAGE CXX
)


set(
    SUDIO_IO_TARGETS

    miniaudio
    PortAudio
    mp3lame
    ${OGG_LIBRARY}
    FLAC
    vorbis
    vorbisenc
    vorbisfile
    )

set_target_properties(
    ${SUDIO_IO_TARGETS} 
    PROPERTIES POSITION_INDEPENDENT_CODE ON
)


target_link_libraries(sudio_io PRIVATE
    ${SUDIO_IO_TARGETS} 
    ${Python_LIBRARIES}
)

include(GNUInstallDirs)
install(TARGETS sudio_io
    COMPONENT python
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)

