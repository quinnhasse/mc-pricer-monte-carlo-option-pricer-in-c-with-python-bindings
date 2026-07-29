add_executable(mc_benchmarks bench_pricer.cpp)

target_link_libraries(mc_benchmarks
    PRIVATE
        mc_headers
        benchmark::benchmark
        benchmark::benchmark_main
)
