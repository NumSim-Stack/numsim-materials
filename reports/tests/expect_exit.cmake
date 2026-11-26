# cmake -DTOOL=<material_report> -DEXPECTED=<exit code> [-DARGS=a|b] [-DMATCH=<regex>] -P expect_exit.cmake
# Runs TOOL with ARGS; fails unless it exits with EXPECTED and, given MATCH, its stderr matches.
string(REPLACE "|" ";" _args "${ARGS}")
execute_process(COMMAND ${TOOL} ${_args} RESULT_VARIABLE _code OUTPUT_QUIET ERROR_VARIABLE _err)
if(NOT _code STREQUAL EXPECTED)
  message(FATAL_ERROR "material_report ${_args}: exit code ${_code}, expected ${EXPECTED}\n${_err}")
endif()
if(DEFINED MATCH AND NOT _err MATCHES "${MATCH}")
  message(FATAL_ERROR "material_report ${_args}: stderr does not match '${MATCH}':\n${_err}")
endif()
