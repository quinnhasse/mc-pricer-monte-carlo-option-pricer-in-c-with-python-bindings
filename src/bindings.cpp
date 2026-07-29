// pybind11 bindings — filled in by later commits
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(mc_pricer, m) {
    m.doc() = "Monte Carlo option pricer";
}
