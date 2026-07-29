add_executable(mc_tests
    test_european.cpp
    test_barrier.cpp
    test_asian.cpp
    test_heston.cpp
)

target_link_libraries(mc_tests
    PRIVATE
        mc_headers
        Catch2::Catch2WithMain
)

include(Catch)
catch_discover_tests(mc_tests)
