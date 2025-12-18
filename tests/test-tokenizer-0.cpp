#include "llama.h"
#include "common.h"
#include "console.h"

#include <cstdio>
#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <thread>

//static const std::map<std::string, std::vector<llama_token>> & k_tests() {
//    static std::map<std::string, std::vector<llama_token>> _k_tests = {
//        { ""                      , {  }, },
//        { " "                     , {     220, }, },
//        { "  "                    , {     256, }, },
//        { "   "                   , {     262, }, },
//        { "\t"                    , {     197, }, },
//        { "\n"                    , {     198, }, },
//        { "\n\n"                  , {     271, }, },
//        { "\n\n\n"                , {    1432, }, },
//        { "\t\n"                  , {    1602, }, },
//        { "Hello world"           , {    9906,   1917, }, },
//        { " Hello world"          , {   22691,   1917, }, },
//        { "Hello World"           , {    9906,   4435, }, },
//        { " Hello World"          , {   22691,   4435, }, },
//        { " Hello World!"         , {   22691,   4435,      0, }, },
//        { "Hello, world!"         , {    9906,     11,   1917,      0, }, },
//        { " Hello, world!"        , {   22691,     11,   1917,      0, }, },
//        { " this is 🦙.cpp"        , {     420,    374,  11410,     99,    247,     13,  11055, }, },
//        { "w048 7tuijk dsdfhu"    , {      86,  23904,    220,     22,     83,   2005,  42908,  11729,   3013,  17156, }, },
//        { "нещо на Български"     , {   79862, 102118,  13373,  64571,  34694,   3114, 112203,  80112, }, },
//        { "កាន់តែពិសេសអាចខលចេញ"   , {   21549,    222,  98629,    241,  45358,    233,  21549,    237,  45358,    224,  21549,    244,  21549,    115,  21549,    253,  45358,    223,  21549,    253,  21549,     95,  98629,    227,  21549,    223,  21549,    249,  21549,    227,  45358,    223,  21549,    231, }, },
//        { "🚀 (normal) 😶‍🌫️ (multiple emojis concatenated) ✅ (only emoji that has its own token)", {    9468,    248,    222,    320,   8416,      8,  27623,    114, 102470,   9468,    234,    104,  31643,    320,  36773, 100166,  98634,      8,  26602,    227,    320,   3323,  43465,    430,    706,   1202,   1866,   4037,      8, }, },
//        { "Hello"                 , {    9906, }, },
//        { " Hello"                , {   22691, }, },
//        { "  Hello"               , {     220,  22691, }, },
//        { "   Hello"              , {     256,  22691, }, },
//        { "    Hello"             , {     262,  22691, }, },
//        { "    Hello\n    Hello"  , {     262,  22691,    198,    262,  22691, }, },
//        { " ("                    , {     320, }, },
//        { "\n ="                  , {     198,    284, }, },
//        { "' era"                 , {       6,  11639, }, },
//        { "Hello, y'all! How are you 😁 ?我想在apple工作1314151天～", {    9906,     11,    379,  65948,      0,   2650,    527,    499,  27623,    223,    949,  37046, 101067,  19000,  23182, 102301,   9263,  18136,     16,  36827,  21909, }, },
//        { "3"                     , {      18, }, },
//        { "33"                    , {    1644, }, },
//        { "333"                   , {    8765, }, },
//        { "3333"                  , {    8765,     18, }, },
//        { "33333"                 , {    8765,   1644, }, },
//        { "333333"                , {    8765,   8765, }, },
//        { "3333333"               , {    8765,   8765,     18, }, },
//        { "33333333"              , {    8765,   8765,   1644, }, },
//        { "333333333"             , {    8765,   8765,   8765, }, },
//    };
//
//    return _k_tests;
//}

using llama_tests = std::map<std::string, std::vector<llama_token>>;

static llama_tests read_tests(const std::string & fname_inp, const std::string & fname_out) {
    llama_tests tests;

    std::ifstream ifs_inp(fname_inp);
    if (!ifs_inp) {
        fprintf(stderr, "%s : error: could not open file '%s'\n", __func__, fname_inp.c_str());
        return tests;
    }

    std::string sraw((std::istreambuf_iterator<char>(ifs_inp)), std::istreambuf_iterator<char>());

    std::ifstream ifs_out(fname_out);
    if (!ifs_out) {
        fprintf(stderr, "%s : error: could not open file '%s'\n", __func__, fname_out.c_str());
        return tests;
    }

    std::vector<std::string> sout;
    for (std::string line; std::getline(ifs_out, line);) {
        sout.push_back(line);
    }

    const std::string sep = "\n__ggml_vocab_test__\n";

    std::vector<std::string> sinp;

    size_t pos = 0;
    while (pos < sraw.size()) {
        const size_t next = sraw.find(sep, pos);
        if (next == std::string::npos) {
            sinp.push_back(sraw.substr(pos));
            break;
        }
        sinp.push_back(sraw.substr(pos, next - pos));
        pos = next + sep.size();
    }

    if (sinp.size() != sout.size()) {
        fprintf(stderr, "%s : error: input and output files have different number of tests\n", __func__);
        return tests;
    }

    for (size_t i = 0; i < sinp.size(); ++i) {
        const std::string & s = sinp[i];
        const std::string & o = string_strip(sout[i]);

        std::vector<llama_token> toks;

        size_t pos = 0;
        while (pos < o.size()) {
            size_t next = o.find(' ', pos);
            if (next == std::string::npos) {
                next = o.size();
            }
            const std::string stok = o.substr(pos, next - pos);
            toks.push_back(std::stoi(stok));
            pos = next + 1;
        }

        tests[s] = toks;
    }

    return tests;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s vocab-file [text-file]\n", argv[0]);
        return 1;
    }

    const std::string fname = argv[1];

    const std::string fname_inp = fname + ".inp";
    const std::string fname_out = fname + ".out";

    std::string fname_text;
    if (argc > 2) {
        fname_text = argv[2];
    }

    fprintf(stderr, "%s : reading vocab from: '%s'\n", __func__, fname.c_str());

    llama_model * model;
    llama_context * ctx;

    llama_backend_init();

    // load the vocab
    {
        auto mparams = llama_model_default_params();

        mparams.vocab_only = true;

        model = llama_model_load_from_file(fname.c_str(), mparams);

        if (model == NULL) {
            fprintf(stderr, "%s: error: failed to load vocab '%s'\n", __func__, fname.c_str());
            return 1;
        }

        auto cparams = llama_context_default_params();

        ctx = llama_init_from_model(model, cparams);

        if (ctx == NULL) {
            fprintf(stderr, "%s: error: failed to load vocab '%s'\n", __func__, fname.c_str());
            llama_model_free(model);
            return 1;
        }
    }

#ifdef _WIN32
    // We need this for unicode console support
    console::init(false, false);
    atexit([]() { console::cleanup(); });
#endif

    bool success = true;

    const auto k_tests = [&]() -> llama_tests {
        if (!fname_text.empty()) {
            return {};
        }

        const auto res = read_tests(fname_inp, fname_out);

        if (res.empty()) {
            fprintf(stderr, "%s : error: no tests found\n", __func__);
            exit(1);
        }

        return res;
    }();

    const bool add_special = false;

    // multi-threaded tokenization
    const int nthread = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(nthread);

    for (int i = 0; i < nthread; i++) {
        threads[i] = std::thread([&, i]() {
            for (const auto & test_kv : k_tests) {
                const std::vector<llama_token> res = common_tokenize(ctx, test_kv.first, add_special, false);

                // here only print the result of the first thread
                // because the other threads are running the same tests
                if (i != 0) {
                    continue;
                }

                printf("\n");
                printf("src: '%s'\n", test_kv.first.c_str());
                printf("res: '%s'\n", common_detokenize(ctx, res).c_str());
                printf("tok: ");
                for (const auto & tok : res) {
                    printf("%d ", tok);
                }
                printf("\n");

                bool correct = res.size() == test_kv.second.size();
                for (int i = 0; i < (int) res.size() && correct; ++i) {
                    if (test_kv.second[i] != res[i]) {
                        correct = false;
                    }
                }

                if (!correct) {
                    fprintf(stderr, "%s : failed test:    '%s'\n", __func__, test_kv.first.c_str());
                    fprintf(stderr, "%s : detokenized to: '%s' instead of '%s'\n", __func__,
                        common_detokenize(ctx, res).c_str(),
                        common_detokenize(ctx, test_kv.second).c_str());
                    fprintf(stderr, "%s : expected tokens: ", __func__);
                    for (const auto & t : test_kv.second) {
                        fprintf(stderr, "%6d '%s', ", t, common_token_to_piece(ctx, t).c_str());
                    }
                    fprintf(stderr, "\n");
                    fprintf(stderr, "%s : got tokens:      ", __func__);
                    for (const auto & t : res) {
                        fprintf(stderr, "%6d '%s', ", t, common_token_to_piece(ctx, t).c_str());
                    }
                    fprintf(stderr, "\n");

                    success = false;
                }
            }
        });
    }

    for (int i = 0; i < nthread; i++) {
        threads[i].join();
    }

    // single threaded tokenization
    if (!fname_text.empty()) {
        fprintf(stderr, "%s : tokenizing: '%s'\n", __func__, fname_text.c_str());

        std::string text;
        {
            std::ifstream ifs(fname_text);
            if (!ifs) {
                fprintf(stderr, "%s : error: could not open file '%s'\n", __func__, fname_text.c_str());
                return 1;
            }
            text = std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
        }

        fprintf(stderr, "%s : text size: %zu\n", __func__, text.size());

        std::vector<llama_token> res;

        {
            const auto t_start = ggml_time_us();

            res = common_tokenize(ctx, text, add_special, false);

            const auto t_end = ggml_time_us();

            fprintf(stderr, "%s : tokenized in %.3f ms (cpp)\n", __func__, (t_end - t_start) / 1000.0);
        }

        fprintf(stderr, "%s : tokens: %zu\n", __func__, res.size());

        {
            const std::string fname_out = fname_text + ".tokcpp";

            std::ofstream ofs(fname_out);
            if (!ofs) {
                fprintf(stderr, "%s : error: could not open file '%s'\n", __func__, fname_out.c_str());
                return 1;
            }

            for (const auto & tok : res) {
                //ofs << tok << " '" << string_strip(llama_detokenize(ctx, std::vector<int>{tok})) << "'" << std::endl;
                ofs << tok << "\n";
            }
        }

        fprintf(stderr, "%s : tokens written to '%s'\n", __func__, (fname_text + ".tokcpp").c_str());
    }

    llama_model_free(model);
    llama_free(ctx);

    llama_backend_free();

    printf("\n");
    printf("Tests %s\n", success ? "passed" : "failed");

    return success ? 0 : 3;
}
