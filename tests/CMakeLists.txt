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
