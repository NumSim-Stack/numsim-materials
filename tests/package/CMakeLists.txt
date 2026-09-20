# A consumer of the INSTALLED package, exactly as a downstream project sees it.
#
# Deliberately NOT part of the main build: it must resolve numsim-materials
# through find_package() against an install prefix, not through the source
# tree. The CI job configures it with CMAKE_PREFIX_PATH pointing at a scratch
# prefix that `cmake --install` just populated.
#
# What it guards: the installed package used to configure successfully and then
# fail to compile, because the fetched tmech headers reached the target only
# through $<BUILD_INTERFACE:> and were never installed. Nothing in the main
# build could catch that -- the source tree always has the build-tree paths.
cmake_minimum_required(VERSION 3.20)
project(numsim_materials_package_test CXX)

find_package(numsim-materials REQUIRED)

add_executable(consumer consumer.cpp)
target_link_libraries(consumer PRIVATE numsim-materials::numsim-materials)
