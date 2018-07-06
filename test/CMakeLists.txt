macro (create_test target)
  add_test (NAME ${target} COMMAND $<TARGET_FILE:${target}>)
endmacro (create_test)

add_library (CATCH catch.cpp)

set(testname utest-linear_least_squares)
    add_executable (${testname} ${testname}.cpp)
    target_link_libraries (${testname} CATCH ${EIGEN3_LIBRARIES})
    if (WITH_PLOT)
    target_link_libraries (${testname} "-lboost_iostreams" "-lboost_system"
      "-lboost_filesystem")
    endif()
    create_test(${testname})

set(testname utest-falling_object)
    add_executable (${testname} ${testname}.cpp)
    target_link_libraries (${testname} CATCH ${EIGEN3_LIBRARIES}
      ${RANGE-V3_LIBRARIES})
    if (WITH_PLOT)
    target_link_libraries (${testname} "-lboost_iostreams" "-lboost_system"
      "-lboost_filesystem")
    endif()
    create_test(${testname})

  set(testname utest-kalman_folding_4-frp)
    add_executable (${testname} ${testname}.cpp)
    target_link_libraries (${testname} CATCH ${EIGEN3_LIBRARIES}
      ${RANGE-V3_LIBRARIES} "-lsodium" )
    if (WITH_PLOT)
    target_link_libraries (${testname} "-lboost_iostreams" "-lboost_system"
      "-lboost_filesystem")
    endif()
    create_test(${testname})
