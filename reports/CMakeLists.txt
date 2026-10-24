add_executable(material_report material_report.cpp)
target_link_libraries(material_report PRIVATE numsim-materials::numsim-materials)

# Every docs/materials/<material>/report.json as a regression test (checks only, no figures).
if(NUMSIM_MATERIALS_BUILD_TESTS AND NOT NUMSIM_MATERIALS_REPORT_TESTS STREQUAL "OFF")
  if(NOT NUMSIM_MATERIALS_REPORT_TESTS MATCHES "^(AUTO|ON)$")
    message(FATAL_ERROR "numsim-materials: NUMSIM_MATERIALS_REPORT_TESTS must be AUTO, ON or OFF")
  endif()
  # ON: a missing module is a configure error, so CI cannot pass without the tests
  set(_missing_level WARNING)
  if(NUMSIM_MATERIALS_REPORT_TESTS STREQUAL "ON")
    set(_missing_level FATAL_ERROR)
  endif()

  find_package(Python3 COMPONENTS Interpreter)
  set(_numpy_missing 1)
  if(Python3_FOUND)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy" RESULT_VARIABLE _numpy_missing OUTPUT_QUIET ERROR_QUIET)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import matplotlib" RESULT_VARIABLE _matplotlib_missing
                    OUTPUT_QUIET ERROR_QUIET)
  endif()

  if(NOT Python3_FOUND)
    message(${_missing_level} "numsim-materials: no Python 3, the report tests are skipped")
  elseif(_numpy_missing)
    message(${_missing_level} "numsim-materials: numpy not found, the report tests are skipped")
  else()
    if(_matplotlib_missing)
      message(${_missing_level} "numsim-materials: matplotlib not found, the report figure tests are skipped")
    endif()
    add_test(NAME report_script COMMAND ${Python3_EXECUTABLE} -m unittest discover -q -s tests -t .
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set(_report_tests report_script)
    file(GLOB _reports CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/docs/materials/*/report.json)
    foreach(_report IN LISTS _reports)
      get_filename_component(_dir ${_report} DIRECTORY)
      get_filename_component(_name ${_dir} NAME)
      add_test(NAME report_${_name}
               COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/make_report.py
                       --tool $<TARGET_FILE:material_report> --check ${_dir})
      list(APPEND _report_tests report_${_name})
    endforeach()
    # no __pycache__ in the source tree
    set_tests_properties(${_report_tests} PROPERTIES LABELS report TIMEOUT 300
                         ENVIRONMENT PYTHONDONTWRITEBYTECODE=1)
  endif()
endif()
