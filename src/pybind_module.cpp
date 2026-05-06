#include "lsmash_api.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

symbol_list_ to_symbol_list(const std::vector<unsigned int>& x) {
    symbol_list_ out;
    out.reserve(x.size());
    for (auto v : x) {
        out.push_back(symbol(v));
    }
    return out;
}

std::vector<symbol_list_> to_symbol_lists(const std::vector<std::vector<unsigned int>>& xs) {
    std::vector<symbol_list_> out;
    out.reserve(xs.size());
    for (const auto& row : xs) {
        out.push_back(to_symbol_list(row));
    }
    return out;
}

std::size_t infer_square_size(const matrix_dbl& M) {
    std::size_t n = 0;
    for (const auto& row : M) {
        n = std::max<std::size_t>(n, row.first + 1);
        for (const auto& col : row.second) {
            n = std::max<std::size_t>(n, col.first + 1);
        }
    }
    return n;
}

py::array_t<double> matrix_dbl_to_numpy(const matrix_dbl& M) {
    const std::size_t n = infer_square_size(M);

    py::array_t<double> arr({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(n)});
    auto buf = arr.mutable_unchecked<2>();

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            buf(i, j) = 0.0;
        }
    }

    for (const auto& row : M) {
        for (const auto& col : row.second) {
            buf(row.first, col.first) = col.second;
        }
    }

    return arr;
}

py::dict matrix_dbl_to_pydict(const matrix_dbl& M) {
    py::dict out;
    for (const auto& row : M) {
        py::dict row_dict;
        for (const auto& col : row.second) {
            row_dict[py::int_(col.first)] = py::float_(col.second);
        }
        out[py::int_(row.first)] = row_dict;
    }
    return out;
}

} // namespace

PYBIND11_MODULE(_lsmash, m) {
    m.doc() = "pybind11 bindings for lsmash";

    py::class_<LsmashOptions>(m, "LsmashOptions")
        .def(py::init<>())
        .def_readwrite("data_dir", &LsmashOptions::data_dir)
        .def_readwrite("data_type", &LsmashOptions::data_type)
        .def_readwrite("data_len", &LsmashOptions::data_len)
        .def_readwrite("partition", &LsmashOptions::partition)
        .def_readwrite("use_derivative", &LsmashOptions::use_derivative)
        .def_readwrite("sae", &LsmashOptions::sae)
        .def_readwrite("num_repeat", &LsmashOptions::num_repeat)
        .def_readwrite("depth", &LsmashOptions::depth);

    m.def(
        "from_file",
        [](const std::string& path, const LsmashOptions& opt) {
            py::gil_scoped_release release;
            matrix_dbl D = lsmash_from_file(path, opt);
            py::gil_scoped_acquire acquire;
            return matrix_dbl_to_numpy(D);
        },
        py::arg("path"),
        py::arg("options") = LsmashOptions{},
        "Run lsmash on a file and return a dense NumPy array."
    );

    m.def(
        "from_file_sparse",
        [](const std::string& path, const LsmashOptions& opt) {
            py::gil_scoped_release release;
            matrix_dbl D = lsmash_from_file(path, opt);
            py::gil_scoped_acquire acquire;
            return matrix_dbl_to_pydict(D);
        },
        py::arg("path"),
        py::arg("options") = LsmashOptions{},
        "Run lsmash on a file and return a sparse dict-of-dicts."
    );

    m.def(
        "from_sequences",
        [](const std::vector<std::vector<unsigned int>>& seqs, const LsmashOptions& opt) {
            std::vector<symbol_list_> native = to_symbol_lists(seqs);
            py::gil_scoped_release release;
            matrix_dbl D = lsmash_from_sequences(native, opt);
            py::gil_scoped_acquire acquire;
            return matrix_dbl_to_numpy(D);
        },
        py::arg("seqs"),
        py::arg("options") = LsmashOptions{},
        "Run lsmash on in-memory symbolic sequences and return a dense NumPy array."
    );

    m.def(
        "from_sequences_sparse",
        [](const std::vector<std::vector<unsigned int>>& seqs, const LsmashOptions& opt) {
            std::vector<symbol_list_> native = to_symbol_lists(seqs);
            py::gil_scoped_release release;
            matrix_dbl D = lsmash_from_sequences(native, opt);
            py::gil_scoped_acquire acquire;
            return matrix_dbl_to_pydict(D);
        },
        py::arg("seqs"),
        py::arg("options") = LsmashOptions{},
        "Run lsmash on in-memory symbolic sequences and return a sparse dict-of-dicts."
    );
}
