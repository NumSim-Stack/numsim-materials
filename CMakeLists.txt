cmake_minimum_required(VERSION 3.22)

project(numsim-materials
  VERSION 0.1.0
  LANGUAGES CXX
  DESCRIPTION "Property-graph material framework for FEM simulations"
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Dependencies ---
include(FetchContent)

set(BUILD_TESTS OFF CACHE BOOL "" FORCE)     # disable numsim-core's own tests
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)  # disable numsim-core's own examples
FetchContent_Declare(
  numsim-core
  GIT_REPOSITORY https://github.com/NumSim-Stack/numsim-core
  GIT_TAG main
)
# To use a local numsim-core checkout, configure with:
#   -DFETCHCONTENT_SOURCE_DIR_NUMSIM-CORE=/path/to/numsim-core
FetchContent_MakeAvailable(numsim-core)

# --- Dependencies (tmech tensor library) ---
# Fetched by default rather than found. find_package(tmech) succeeds against
# ANY installed version, including one predating features used here (the Mandel
# adaptor tag in solvers/unknown_layout.h), and tmech's config exposes no
# version constraint that could express the difference — so a stale system
# install would silently win and fail at compile time.
option(NUMSIM_USE_SYSTEM_TMECH "Use an installed tmech instead of fetching it" OFF)

if(NUMSIM_USE_SYSTEM_TMECH)
  find_package(tmech REQUIRED)
else()
  # tmech defaults TMECH_BUILD_TESTS and TMECH_BUILD_EXAMPLES to ON; as a
  # subproject we want neither. (TMECH_INSTALL already defaults off when tmech
  # is not the top-level project.)
  set(TMECH_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(TMECH_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(TMECH_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    tmech
    GIT_REPOSITORY https://github.com/petlenz/tmech
    GIT_TAG master
  )
  # To use a local tmech checkout, configure with:
  #   -DFETCHCONTENT_SOURCE_DIR_TMECH=/path/to/tmech
  FetchContent_MakeAvailable(tmech)
endif()

# --- Dependencies (nlohmann/json for the configuration layer) ---
# Fetched rather than left optional. Model configuration is meant to be JSON
# driven, so a build without it is missing the primary way to define a material,
# not an extra. It was previously reached only via __has_include against whatever
# happened to be installed system-wide, which meant the JSON tests silently
# vanished on a machine without it.
find_package(nlohmann_json 3.11 QUIET)
if(NOT nlohmann_json_FOUND)
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
  )
  set(JSON_BuildTests OFF CACHE INTERNAL "")
  FetchContent_MakeAvailable(nlohmann_json)
endif()

# --- Dependencies (Eigen for linear algebra) ---
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
  # SOURCE_SUBDIR names a directory with no CMakeLists.txt, so MakeAvailable
  # downloads Eigen and stops: its CMake never runs. Eigen is header-only, so
  # nothing is lost, and the alternative was fighting its build system -- it
  # gates its ~780 tests on BUILD_TESTING (EIGEN_BUILD_TESTING only exists
  # after 3.4.0), and suppressing them meant forcing a GLOBAL variable OFF that
  # our own tests would depend on the moment anyone switched to include(CTest).
  # Same treatment tmech and nlohmann already get here: take the headers, leave
  # the build system alone.
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR  headers-only-do-not-configure
  )
  FetchContent_MakeAvailable(eigen)
endif()

# --- Header-only library ---
add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_23)
target_link_libraries(${PROJECT_NAME} INTERFACE numsim-core)
# Every imported target linked INTERFACE ends up named in the exported target
# set, and a consumer of the installed package must be able to resolve it. The
# generated Config re-finds each one; without that, find_package() fails with
# "the following imported targets are referenced, but are missing".
set(NUMSIM_MATERIALS_EXPORTED_DEPS numsim-core)

# nlohmann/json is header-only, so it gets tmech's treatment: when it is FETCHED
# rather than found installed, linking the target pulls it into the install
# export set and install(EXPORT) rejects it for not being exported itself.
if(nlohmann_json_FOUND)
  target_link_libraries(${PROJECT_NAME} INTERFACE nlohmann_json::nlohmann_json)
  # Linked INTERFACE, so it is named in the exported target set and the
  # generated Config has to re-find it. Missing here, a consumer of the
  # installed package fails with "the link interface contains
  # nlohmann_json::nlohmann_json but the target was not found".
  list(APPEND NUMSIM_MATERIALS_EXPORTED_DEPS nlohmann_json)
else()
  get_target_property(_njson_inc nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
  if(_njson_inc)
    target_include_directories(${PROJECT_NAME} INTERFACE ${_njson_inc})
  endif()
endif()

# tmech is header-only — add its include path without linking a target
# (linking would pull it into the install export set)
if(TARGET tmech)
  get_target_property(_tmech_inc tmech INTERFACE_INCLUDE_DIRECTORIES)
  if(_tmech_inc)
    target_include_directories(${PROJECT_NAME} INTERFACE ${_tmech_inc})
  endif()
elseif(tmech_FOUND)
  target_link_libraries(${PROJECT_NAME} INTERFACE tmech::tmech)
  list(APPEND NUMSIM_MATERIALS_EXPORTED_DEPS tmech)
endif()

# Eigen — header-only, same pattern. Fetched, there is no target to link, so
# the populated source dir IS the include path.
if(TARGET Eigen3::Eigen)
  target_link_libraries(${PROJECT_NAME} INTERFACE Eigen3::Eigen)
  list(APPEND NUMSIM_MATERIALS_EXPORTED_DEPS Eigen3)
elseif(eigen_SOURCE_DIR)
  # BUILD_INTERFACE: the fetched path lives under the build tree, and a raw one
  # in an exported target's INTERFACE_INCLUDE_DIRECTORIES is rejected at
  # generate time ("prefixed in the build directory"). A consumer of the
  # INSTALLED package supplies its own Eigen, exactly as it does for tmech.
  target_include_directories(${PROJECT_NAME}
    INTERFACE $<BUILD_INTERFACE:${eigen_SOURCE_DIR}>)
endif()

# Force C++23 globally — numsim-core headers use <print>/std::println
set(CMAKE_CXX_STANDARD 23 CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "" FORCE)

# --- Optional Qt plotting ---
option(ENABLE_PLOTTING "Enable Qt6-based live plotting" OFF)
if(ENABLE_PLOTTING)
  find_package(Qt6 COMPONENTS Widgets PrintSupport QUIET)
  if(Qt6_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/qcustomplot/qcustomplot.cpp")
    set(CMAKE_AUTOMOC ON)
    add_library(qcustomplot STATIC third_party/qcustomplot/qcustomplot.cpp)
    target_include_directories(qcustomplot PUBLIC third_party/qcustomplot)
    target_link_libraries(qcustomplot PUBLIC Qt6::Widgets Qt6::PrintSupport)
    target_compile_definitions(${PROJECT_NAME} INTERFACE NUMSIM_HAS_PLOTTING)
    target_link_libraries(${PROJECT_NAME} INTERFACE qcustomplot)
    message(STATUS "Plotting enabled (Qt6 + QCustomPlot)")
  else()
    message(STATUS "Plotting disabled (Qt6 or QCustomPlot not found)")
  endif()
endif()

# --- Installation ---
include(GNUInstallDirs)
install(
  DIRECTORY include/numsim-materials
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${CMAKE_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
)

install(
  FILES "${CMAKE_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        "${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
)

install(
  EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
)

# --- Testing ---
# include(CTest) is the standard entry point: it calls enable_testing() and
# declares BUILD_TESTING, the switch consumers expect to reach. Only at top
# level -- embedded in a superproject, that project owns the dashboard targets
# and its BUILD_TESTING choice should reach us rather than be re-declared.
if(PROJECT_IS_TOP_LEVEL)
  include(CTest)
elseif(NOT DEFINED BUILD_TESTING)
  set(BUILD_TESTING ON)
endif()

# Two switches, deliberately: BUILD_TESTING is how a superproject turns off
# every subproject's tests at once, NUMSIM_BUILD_TESTS is how it turns off only
# ours. Either being off is enough.
option(NUMSIM_BUILD_TESTS "Build unit tests" ON)
if(NUMSIM_BUILD_TESTS AND BUILD_TESTING)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest
    GIT_TAG v1.15.2
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  add_subdirectory(tests)
endif()

# --- Examples ---
option(NUMSIM_BUILD_EXAMPLES "Build examples" ON)
if(NUMSIM_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
