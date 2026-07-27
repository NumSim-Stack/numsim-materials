include(GoogleTest)

macro(add_numsim_test TARGET_NAME)
  add_executable(${TARGET_NAME} ${ARGN})
  target_link_libraries(${TARGET_NAME}
    PRIVATE numsim-materials::numsim-materials GTest::gtest_main)
  gtest_discover_tests(${TARGET_NAME})
endmacro()

add_numsim_test(test_property_graph test_property_graph.cpp)
add_numsim_test(test_materials test_materials.cpp)
add_numsim_test(test_damage test_damage.cpp)
add_numsim_test(test_j2_plasticity test_j2_plasticity.cpp)
add_numsim_test(test_rk_integrator test_rk_integrator.cpp)
add_numsim_test(test_drucker_prager test_drucker_prager.cpp)
add_numsim_test(test_unknown_layout test_unknown_layout.cpp)

# Data dumper for plotting (not a test — standalone executable)
add_executable(plot_data plot_data.cpp)
target_link_libraries(plot_data PRIVATE numsim-materials::numsim-materials)

add_executable(debug_apex debug_apex.cpp)
target_link_libraries(debug_apex PRIVATE numsim-materials::numsim-materials)
