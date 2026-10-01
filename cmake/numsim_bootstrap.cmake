# Copy this file into a NumSim project as cmake/numsim_bootstrap.cmake and
# include it after project(). It makes the NumSim* CMake modules available:
# from the sibling checkout ../numsim-cmake when present, otherwise from the
# pinned tag below. Then: include(NumSimDependency), include(NumSimTesting), ...
include(FetchContent)
set(NUMSIM_CMAKE_TAG "v0.1.1" CACHE STRING "numsim-cmake version to fetch")
if(NOT DEFINED FETCHCONTENT_SOURCE_DIR_NUMSIM-CMAKE AND EXISTS "${CMAKE_SOURCE_DIR}/../numsim-cmake/cmake/NumSimDependency.cmake")
    set(FETCHCONTENT_SOURCE_DIR_NUMSIM-CMAKE "${CMAKE_SOURCE_DIR}/../numsim-cmake")
endif()
FetchContent_Declare(numsim-cmake
    GIT_REPOSITORY https://github.com/NumSim-Stack/numsim-cmake.git
    GIT_TAG        ${NUMSIM_CMAKE_TAG}
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR  does-not-exist-so-nothing-is-configured)
FetchContent_MakeAvailable(numsim-cmake)
list(APPEND CMAKE_MODULE_PATH "${numsim-cmake_SOURCE_DIR}/cmake")
