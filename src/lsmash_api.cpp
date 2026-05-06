#include "lsmash_api.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string normalize_data_dir(std::string s) {
    if (s == "row") return "across";
    if (s == "column") return "up";
    if (s == "across" || s == "up") return s;
    throw std::invalid_argument("data_dir must be one of: row, column, across, up");
}

std::string normalize_data_type(std::string s) {
    if (s == "symbolic" || s == "continuous") return s;
    throw std::invalid_argument("data_type must be either symbolic or continuous");
}

void apply_sae_diagonal(matrix_dbl& D,
                        std::vector<symbol_list_>& S,
                        unsigned int repeat,
                        unsigned int depth) {
    unsigned int alphabet = 0;
    for (unsigned int i = 0; i < S.size(); ++i) {
        Symbolic_string_ s(S[i]);
        if (s.get_alphabet() > alphabet) {
            alphabet = s.get_alphabet();
        }
    }

    const unsigned int num_elements = static_cast<unsigned int>(S.size());

    #pragma omp parallel for
    for (unsigned int i = 0; i < num_elements; i++) {
        double sum = 0.0;
        for (unsigned int r = 0; r < repeat; r++) {
            Symbolic_string_ a(S[i], alphabet);
            Symbolic_string_ tmp(!a + a);
            tmp.get_norm_new(depth);
            sum += tmp.norm;
        }
        D[i][i] = sum / (repeat + 0.0);
    }
}

} // namespace

matrix_dbl lsmash_from_sequences(std::vector<symbol_list_>& seqs, const LsmashOptions& opt) {
    normalize_data_type(opt.data_type);
    normalize_data_dir(opt.data_dir);

    if (seqs.empty()) {
        throw std::invalid_argument("seqs must not be empty");
    }

    matrix_dbl D = llk_distance(seqs);

    if (opt.sae) {
        apply_sae_diagonal(D, seqs, opt.num_repeat, opt.depth);
    }

    return D;
}

matrix_dbl lsmash_from_file(const std::string& seqfile, const LsmashOptions& opt) {
    const std::string data_dir = normalize_data_dir(opt.data_dir);
    const std::string data_type = normalize_data_type(opt.data_type);

    if (seqfile.empty()) {
        throw std::invalid_argument("seqfile must not be empty");
    }

    data_reader* R = nullptr;

    if (data_type == "continuous") {
        R = new data_reader(seqfile, data_dir, opt.partition, opt.data_len, false, opt.use_derivative);
    } else {
        R = new data_reader(seqfile, data_dir, opt.data_len);
    }

    std::vector<symbol_list_> S = R->getlist_vector();
    delete R;

    if (S.empty()) {
        throw std::runtime_error("no sequences were read from file");
    }

    matrix_dbl D = llk_distance(S);

    if (opt.sae) {
        apply_sae_diagonal(D, S, opt.num_repeat, opt.depth);
    }

    return D;
}
