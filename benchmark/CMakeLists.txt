find_package(benchmark REQUIRED)

if(NOT CMAKE_BUILD_TYPE MATCHES "^(Release|RelWithDebInfo)$")
  message(WARNING "numsim-materials benchmarks: build type '${CMAKE_BUILD_TYPE}' keeps the debug checks on; "
                  "use -DCMAKE_BUILD_TYPE=Release")
endif()

add_executable(numsim_materials_benchmark material_point_benchmark.cpp kernel_benchmark.cpp)
target_link_libraries(numsim_materials_benchmark PRIVATE numsim-materials::numsim-materials benchmark::benchmark_main)
target_compile_options(numsim_materials_benchmark PRIVATE -O3 -march=native)
