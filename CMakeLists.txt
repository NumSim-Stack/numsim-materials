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
  GIT_TAG origin/feature/property-graph-and-factory
)
FetchContent_MakeAvailable(numsim-core)

# --- Dependencies (tmech tensor library) ---
find_package(tmech REQUIRED)

# --- Header-only library ---
add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_23)
target_link_libraries(${PROJECT_NAME} INTERFACE numsim-core tmech::tmech)

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
option(NUMSIM_BUILD_TESTS "Build unit tests" ON)
if(NUMSIM_BUILD_TESTS)
  enable_testing()
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
