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
project(python_rateshift LANGUAGES CXX)

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
    include(FetchContent)
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


set(LIBSAMPLERATE_VERSION 0.2.2)
set(BUILD_TESTING OFF CACHE BOOL "Disable libsamplerate test build")


if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/libsamplerate")
    include(FetchContent)
    message(STATUS "Fetching libsamplerate...")
    FetchContent_Declare(
        libsamplerate
        GIT_REPOSITORY https://github.com/libsndfile/libsamplerate
        GIT_TAG c96f5e3de9c4488f4e6c97f59f5245f22fda22f7 # 0.2.2
    )
    FetchContent_MakeAvailable(libsamplerate)
else()
    message(STATUS "Using existing libsamplerate in ${CMAKE_CURRENT_SOURCE_DIR}/libsamplerate")
    add_subdirectory(libsamplerate)
endif()

set_target_properties(samplerate PROPERTIES POSITION_INDEPENDENT_CODE ON)

pybind11_add_module(python_rateshift
    src/python_bindings.cpp
    src/rateshift.cpp
)

target_include_directories(python_rateshift
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/external/libsamplerate/include
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
        ${Python_INCLUDE_DIRS}
)


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

apply_target_compile_options(python_rateshift)

if(DEFINED PACKAGE_VERSION_INFO)
    add_definitions(-DPACKAGE_VERSION_INFO="${PACKAGE_VERSION_INFO}")
endif()

target_compile_definitions(python_rateshift
    PUBLIC
        LIBSAMPLERATE_VERSION="${LIBSAMPLERATE_VERSION}"
    PRIVATE
        $<$<BOOL:${PACKAGE_VERSION_INFO}>:VERSION_INFO="${PACKAGE_VERSION_INFO}">
)

set_target_properties(python_rateshift
    PROPERTIES
        PREFIX ""
        OUTPUT_NAME "_rateshift"
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
        LINKER_LANGUAGE CXX
)

target_link_libraries(python_rateshift
    PRIVATE
        samplerate
        ${Python_LIBRARIES}
)

include(GNUInstallDirs)
install(TARGETS python_rateshift
  COMPONENT python
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  )

  