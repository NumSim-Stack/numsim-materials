find_package(benchmark REQUIRED)

# The debug checks rerun constants and check reads on every update; timings with them on are not the library's.
get_property(_numsim_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NUMSIM_MATERIALS_DEBUG_CHECKS STREQUAL "AUTO")
  if(NOT _numsim_multi_config AND NOT CMAKE_BUILD_TYPE MATCHES "^(Release|RelWithDebInfo|MinSizeRel)$")
    message(WARNING "numsim-materials benchmarks: build type '${CMAKE_BUILD_TYPE}' turns the debug checks on; "
                    "use -DCMAKE_BUILD_TYPE=Release")
  endif()
elseif(NUMSIM_MATERIALS_DEBUG_CHECKS)
  message(WARNING "numsim-materials benchmarks: NUMSIM_MATERIALS_DEBUG_CHECKS is on; the timings include the checks")
endif()

add_executable(numsim_materials_benchmark material_point_benchmark.cpp kernel_benchmark.cpp)
target_link_libraries(numsim_materials_benchmark PRIVATE numsim-materials::numsim-materials benchmark::benchmark_main)
target_compile_options(numsim_materials_benchmark PRIVATE -O3 -march=native)
