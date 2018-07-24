macro (create_test target)
  add_test (NAME ${target} COMMAND $<TARGET_FILE:${target}>)
endmacro (create_test)

set(testname Kalman-Folding-1-Preludes)
  add_executable (${testname} ${testname}.cpp)
  target_link_libraries (${testname} CATCH ${EIGEN3_LIBRARIES})
  if (WITH_PLOT)
  target_link_libraries (${testname} "-lboost_iostreams" "-lboost_system"
    "-lboost_filesystem")
  endif()
  create_test(${testname})

set(testname Kalman-Folding-2-falling_object)
  add_executable (${testname} ${testname}.cpp)
  target_link_libraries (${testname} CATCH ${EIGEN3_LIBRARIES}
    ${RANGE-V3_LIBRARIES})
  if (WITH_PLOT)
  target_link_libraries (${testname} "-lboost_iostreams" "-lboost_system"
    "-lboost_filesystem")
  endif()
  create_test(${testname})

set(testname Kalman-Folding-4-frp)
  add_executable (${testname} ${testname}.cpp)
  target_link_libraries (${testname} CATCH ${EIGEN3_LIBRARIES}
    ${RANGE-V3_LIBRARIES} "-lsodium" )
  if (WITH_PLOT)
  target_link_libraries (${testname} "-lboost_iostreams" "-lboost_system"
    "-lboost_filesystem")
  endif()
  create_test(${testname})
