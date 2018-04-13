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
    target_link_libraries (${testname} CATCH ${EIGEN3_LIBRARIES} "-lpython3.6m")
    target_include_directories(${testname} PUBLIC "/usr/include/python3.6m/")
    create_test(${testname})
