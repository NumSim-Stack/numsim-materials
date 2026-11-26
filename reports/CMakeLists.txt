add_executable(material_report material_report.cpp)
target_link_libraries(material_report PRIVATE numsim-materials::numsim-materials)

# material_report's exit codes (0 converged, 1 a step did not converge, 2 an input error) on tests/cli/*.json
if(NUMSIM_MATERIALS_BUILD_TESTS)
  set(_cli ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli)
  set(_cli_out ${CMAKE_CURRENT_BINARY_DIR}/cli_tests)
  function(material_report_exit_test name code args)
    add_test(NAME material_report_${name}
             COMMAND ${CMAKE_COMMAND} -DTOOL=$<TARGET_FILE:material_report> -DEXPECTED=${code} -DARGS=${args} ${ARGN}
                     -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/expect_exit.cmake)
    set_tests_properties(material_report_${name} PROPERTIES LABELS report TIMEOUT 120)
  endfunction()
  material_report_exit_test(usage 2 "" -DMATCH=usage)
  material_report_exit_test(converged 0 "${_cli}/elastic.json|${_cli_out}/converged")
  material_report_exit_test(not_converged 1 "${_cli}/no_iterations.json|${_cli_out}/not_converged" "-DMATCH=did not converge")
  material_report_exit_test(unwritable_directory 2 "${_cli}/elastic.json|${_cli}/elastic.json/out")
  material_report_exit_test(bad_name 2 "${_cli}/bad_name.json|${_cli_out}/bad_name" "-DMATCH=no file name")
  material_report_exit_test(wrong_type 2 "${_cli}/string_scale.json|${_cli_out}/wrong_type" "-DMATCH=case 'u'.*scale")
  material_report_exit_test(broken_json 2 "${_cli}/broken.json|${_cli_out}/broken_json" "-DMATCH=broken.json")
  material_report_exit_test(missing_file 2 "${_cli}/missing.json|${_cli_out}/missing_file" "-DMATCH=cannot open")
endif()

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
    add_test(NAME convert_uvwmaterial_card COMMAND ${Python3_EXECUTABLE} -m unittest discover -q -s tests -t .
             WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tools)
    set(_report_tests report_script convert_uvwmaterial_card)
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
