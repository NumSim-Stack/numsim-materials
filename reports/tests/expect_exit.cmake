# cmake -DTOOL=<material_report> -DEXPECTED=<exit code> [-DARGS=a|b] [-DMATCH=<regex>] [-DEXISTS=<file>] [-DABSENT=<file>]
#       [-DFILE=<file> -DFILE_MATCH=<regex>] -P expect_exit.cmake
# Runs TOOL with ARGS; fails unless it exits with EXPECTED, its stderr matches MATCH, EXISTS exists, ABSENT does not
# and FILE matches FILE_MATCH.
string(REPLACE "|" ";" _args "${ARGS}")
foreach(_file IN ITEMS ${EXISTS} ${ABSENT} ${FILE}) # left over from an earlier run
  file(REMOVE "${_file}")
endforeach()
execute_process(COMMAND ${TOOL} ${_args} RESULT_VARIABLE _code OUTPUT_QUIET ERROR_VARIABLE _err)
if(NOT _code STREQUAL EXPECTED)
  message(FATAL_ERROR "material_report ${_args}: exit code ${_code}, expected ${EXPECTED}\n${_err}")
endif()
if(DEFINED MATCH AND NOT _err MATCHES "${MATCH}")
  message(FATAL_ERROR "material_report ${_args}: stderr does not match '${MATCH}':\n${_err}")
endif()
if(DEFINED EXISTS AND NOT EXISTS "${EXISTS}")
  message(FATAL_ERROR "material_report ${_args}: did not write ${EXISTS}\n${_err}")
endif()
if(DEFINED ABSENT AND EXISTS "${ABSENT}")
  message(FATAL_ERROR "material_report ${_args}: wrote ${ABSENT} before the input error\n${_err}")
endif()
if(DEFINED FILE)
  file(READ "${FILE}" _content)
  if(NOT _content MATCHES "${FILE_MATCH}")
    message(FATAL_ERROR "material_report ${_args}: ${FILE} does not match '${FILE_MATCH}':\n${_content}")
  endif()
endif()
