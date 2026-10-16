add_executable(material_report material_report.cpp)
target_link_libraries(material_report PRIVATE numsim-materials::numsim-materials)

# Every docs/materials/<material>/report.json as a regression test (checks only, no figures).
if(NUMSIM_MATERIALS_BUILD_TESTS)
  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_FOUND)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy" RESULT_VARIABLE _numpy_missing OUTPUT_QUIET ERROR_QUIET)
    if(NOT _numpy_missing)
      add_test(NAME report_script COMMAND ${Python3_EXECUTABLE} -m unittest discover -q -s tests -t .
               WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
      add_test(NAME convert_uvwmaterial_card COMMAND ${Python3_EXECUTABLE} -m unittest discover -q -s tests -t .
               WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tools)
      file(GLOB _reports CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/docs/materials/*/report.json)
      foreach(_report IN LISTS _reports)
        get_filename_component(_dir ${_report} DIRECTORY)
        get_filename_component(_name ${_dir} NAME)
        add_test(NAME report_${_name}
                 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/make_report.py
                         --tool $<TARGET_FILE:material_report> --check ${_dir})
      endforeach()
    endif()
  endif()
endif()
