cmake_minimum_required(VERSION 3.15)
project(sudio_io LANGUAGES CXX C)



include(FetchContent)


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

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/portaudio")
    message(STATUS "Fetching portaudio...")
    FetchContent_Declare(
        portaudio
        GIT_REPOSITORY https://github.com/PortAudio/portaudio.git
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
)

# Include directories
target_include_directories(sudio_io PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/inc"
    "${CMAKE_CURRENT_SOURCE_DIR}/portaudio/include"
    "${MINIAUDIO_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/libmp3lame-CMAKE/include"
    "${OGG_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/flac/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/vorbis/include"
)

if(MSVC)
    target_compile_options(sudio_io PRIVATE /EHsc /MP /bigobj)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /MANIFEST:NO")
else()
    target_compile_options(sudio_io PRIVATE -O3 -Wall -Wextra -fPIC)
endif()



set_target_properties(sudio_io PROPERTIES
    PREFIX ""
    OUTPUT_NAME "suio"
    LINKER_LANGUAGE CXX
)


set_target_properties(
    miniaudio 
    PortAudio 
    mp3lame 
    ${OGG_LIBRARY} 
    FLAC 
    vorbis 
    vorbisenc 
    vorbisfile
    PROPERTIES POSITION_INDEPENDENT_CODE ON)


target_link_libraries(sudio_io PRIVATE
    miniaudio
    PortAudio
    mp3lame
    ${OGG_LIBRARY}
    FLAC
    vorbis
    vorbisenc
    vorbisfile
    ${Python_LIBRARIES}
)

include(GNUInstallDirs)
install(TARGETS sudio_io
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
