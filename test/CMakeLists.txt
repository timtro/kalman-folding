macro (create_test target)
  add_test (NAME ${target} COMMAND $<TARGET_FILE:${target}>)
endmacro (create_test)

add_library (CATCH catch.cpp)

set(testname utest-linear_least_squares)
    add_executable (${testname} ${testname}.cpp)
    target_link_libraries (${testname} CATCH ${EIGEN3_LIBRARIES})
    create_test(${testname})


set(testname utest-falling_object)
    add_executable (${testname} ${testname}.cpp)
    target_link_libraries (${testname} CATCH ${EIGEN3_LIBRARIES}
      ${RANGE-V3_LIBRARIES} "-lboost_iostreams -lboost_system -lboost_filesystem")
    create_test(${testname})
