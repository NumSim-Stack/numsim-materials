cmake_minimum_required(VERSION 3.25)

project(numsim-materials
  VERSION 0.1.0
  LANGUAGES CXX
  DESCRIPTION "Property-graph material framework for FEM simulations"
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(NUMSIM_MATERIALS_IS_TOP_LEVEL ON)
else()
  set(NUMSIM_MATERIALS_IS_TOP_LEVEL OFF)
endif()
function(_numsim_materials_legacy_option old new)
  if(DEFINED ${old} AND NOT DEFINED ${new})
    message(DEPRECATION "numsim-materials: ${old} is deprecated, use ${new}")
    set(${new} "${${old}}" CACHE BOOL "" FORCE)
  endif()
endfunction()
_numsim_materials_legacy_option(NUMSIM_BUILD_TESTS NUMSIM_MATERIALS_BUILD_TESTS)
_numsim_materials_legacy_option(NUMSIM_BUILD_EXAMPLES NUMSIM_MATERIALS_BUILD_EXAMPLES)
_numsim_materials_legacy_option(ENABLE_PLOTTING NUMSIM_MATERIALS_ENABLE_PLOTTING)
if(DEFINED NUMSIM_USE_SYSTEM_TMECH)
  message(DEPRECATION "numsim-materials: NUMSIM_USE_SYSTEM_TMECH is obsolete; an installed tmech >= 1.2 is used automatically (see numsim_dependency)")
endif()
option(NUMSIM_MATERIALS_BUILD_TESTS     "Build the numsim-materials test suite" ${NUMSIM_MATERIALS_IS_TOP_LEVEL})
option(NUMSIM_MATERIALS_BUILD_EXAMPLES  "Build the numsim-materials examples"   ${NUMSIM_MATERIALS_IS_TOP_LEVEL})
option(NUMSIM_MATERIALS_ENABLE_PLOTTING "Qt6 + QCustomPlot live plotting"       OFF)
option(NUMSIM_MATERIALS_INSTALL         "Install the library and its CMake package" ON)

include(cmake/numsim_bootstrap.cmake)
include(NumSimDependency)
include(NumSimWarnings)

set(NUMSIM_MATERIALS_NUMSIM_CORE_TAG "v1.0.0" CACHE STRING
    "numsim-core version (tag or commit) to fetch")
set(NUMSIM_MATERIALS_TMECH_TAG "v1.2.0" CACHE STRING
    "tmech version (tag or commit) to fetch")

numsim_dependency(numsim-core
  TARGET numsim-core::numsim-core
  GIT_REPOSITORY https://github.com/NumSim-Stack/numsim-core.git
  GIT_TAG        ${NUMSIM_MATERIALS_NUMSIM_CORE_TAG}
  SIBLING        numsim-core
  FIND_PACKAGE_ARGS CONFIG
  OPTIONS NUMSIM_CORE_BUILD_TESTS=OFF NUMSIM_CORE_BUILD_EXAMPLES=OFF
          NUMSIM_CORE_INSTALL=${NUMSIM_MATERIALS_INSTALL})

numsim_dependency(tmech
  TARGET tmech::tmech
  GIT_REPOSITORY https://github.com/petlenz/tmech.git
  GIT_TAG        ${NUMSIM_MATERIALS_TMECH_TAG}
  SIBLING        tmech
  FIND_PACKAGE_ARGS 1.2 CONFIG
  OPTIONS TMECH_BUILD_TESTS=OFF TMECH_BUILD_EXAMPLES=OFF TMECH_BUILD_BENCHMARK=OFF
          TMECH_INSTALL_LIBRARY=${NUMSIM_MATERIALS_INSTALL})

numsim_dependency(nlohmann_json
  TARGET nlohmann_json::nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3
  FIND_PACKAGE_ARGS 3.11 CONFIG
  OPTIONS JSON_BuildTests=OFF JSON_Install=ON)

find_package(Eigen3 3.4 QUIET)
if(NOT TARGET Eigen3::Eigen)
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR  headers-only-do-not-configure
  )
  FetchContent_MakeAvailable(eigen)
  message(STATUS "${PROJECT_NAME}: Eigen3 (fetched 3.4.0 headers)")
else()
  message(STATUS "${PROJECT_NAME}: Eigen3 (installed)")
endif()

add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_23)
target_link_libraries(${PROJECT_NAME} INTERFACE
  numsim-core::numsim-core tmech::tmech nlohmann_json::nlohmann_json)
set(NUMSIM_MATERIALS_EXPORTED_DEPS numsim-core tmech nlohmann_json)
if(TARGET Eigen3::Eigen)
  target_link_libraries(${PROJECT_NAME} INTERFACE Eigen3::Eigen)
  list(APPEND NUMSIM_MATERIALS_EXPORTED_DEPS Eigen3)
elseif(eigen_SOURCE_DIR)
  target_include_directories(${PROJECT_NAME}
    SYSTEM INTERFACE $<BUILD_INTERFACE:${eigen_SOURCE_DIR}>)
endif()


# --- Optional Qt plotting ---
if(NUMSIM_MATERIALS_ENABLE_PLOTTING)
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
if(NUMSIM_MATERIALS_INSTALL)
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
endif()

if(NUMSIM_MATERIALS_BUILD_TESTS)
  enable_testing()
  include(NumSimGoogleTest)
  numsim_googletest()
  include(NumSimTesting)
  add_subdirectory(tests)
endif()

if(NUMSIM_MATERIALS_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
