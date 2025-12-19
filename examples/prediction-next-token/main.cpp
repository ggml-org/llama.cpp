#include <llama.h>
#include <windows.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <string>

struct TokenInfo {
    int         id;
    float       p;
    std::string piece;
};

#include <windows.h>

#include <cstdlib>  // для malloc/free
#include <cstring>  // для strlen

const char * Utf8FromUtf16(const wchar_t * wstr) {
    if (!wstr) {
        return nullptr;
    }

    int size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, nullptr, 0, nullptr, nullptr);

    char * buffer = (char *) malloc(size_needed);
    if (!buffer) {
        return nullptr;
    }

    WideCharToMultiByte(CP_UTF8, 0, wstr, -1, buffer, size_needed, nullptr, nullptr);

    return buffer;  // caller должен вызвать free()
}

static int wmain(int argc, wchar_t * argv[]) {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    // Установка значений по умолчанию
    const char * model_path = nullptr;
    const char * prompt     = nullptr;
    const char * word       = nullptr;

    // Разбор аргументов
    for (int i = 1; i < argc; i++) {
        if ((wcscmp(argv[i], L"-m") == 0 || wcscmp(argv[i], L"--model") == 0) && i + 1 < argc) {
            model_path = Utf8FromUtf16(argv[++i]);
        } else if ((wcscmp(argv[i], L"-p") == 0 || wcscmp(argv[i], L"--prompt") == 0) && i + 1 < argc) {
            prompt = Utf8FromUtf16(argv[++i]);
        } else if ((wcscmp(argv[i], L"-h") == 0 || wcscmp(argv[i], L"--hypothesis") == 0) && i + 1 < argc) {
            word = Utf8FromUtf16(argv[++i]);
        } else if (i == 1 && argv[i][0] != L'-') {
            model_path = Utf8FromUtf16(argv[i]);
            if (i + 1 < argc) {
                prompt = Utf8FromUtf16(argv[++i]);
            }
        }
    }

    // Проверка обязательных аргументов
    if (model_path == nullptr || prompt == nullptr) {
        fprintf(stderr,
                "Usage: %s -m or --model <model_path> -p|--prompt <prompt> [-h|--hypothesis <first_word>]\n",
                Utf8FromUtf16(argv[0]));
        return 1;
    }

    // 0) backend
    llama_backend_init();

    // 1) load model
    llama_model_params model_params = llama_model_default_params();
    llama_model *      model        = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        fprintf(stderr, "failed to load model: %s\n", model_path);
        llama_backend_free();
        return 1;
    }

    // 2) context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx                = 512;
    llama_context * ctx             = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // 3) vocab
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // 4) tokenize full prompt
    int                      max_tokens = 256;
    std::vector<llama_token> tok(max_tokens);

    int n_tok = llama_tokenize(vocab,
                               prompt,
                               (int) strlen(prompt),
                               tok.data(),
                               (int) tok.size(),
                               /*add_bos=*/true,
                               /*special=*/true);
    if (n_tok < 0) {
        max_tokens = -n_tok;
        tok.resize(max_tokens);
        n_tok = llama_tokenize(vocab, prompt, (int) strlen(prompt), tok.data(), (int) tok.size(), true, true);
    }
    if (n_tok <= 0) {
        fprintf(stderr, "tokenization failed\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    tok.resize(n_tok);

    // 5) build batch correctly (НЕ аллоцируем seq_id вручную!)
    llama_batch batch = llama_batch_get_one(tok.data(), (int) tok.size());
    // batch.pos / batch.seq_id / batch.n_seq_id / batch.logits = nullptr
    // рантайм сам подставит корректные значения и вернёт логиты для последнего токена

    // 6) decode
    int ret = llama_decode(ctx, batch);
    if (ret != 0) {
        fprintf(stderr, "llama_decode failed, ret = %d\n", ret);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // 7) logits of the last token in the batch
    // (так безопаснее: это "последние" логиты, соответствующие отмеченному последнему токену)
    const float * logits  = llama_get_logits(ctx);
    const int     n_vocab = llama_vocab_n_tokens(vocab);

    // 8) softmax + top-10
    // найдём максимум для стабильного softmax
    float max_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    // вычислим exp и сумму
    std::vector<float> probs(n_vocab);
    double             sum = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        float e  = std::exp(logits[i] - max_logit);
        probs[i] = e;
        sum += e;
    }
    for (int i = 0; i < n_vocab; ++i) {
        probs[i] = (float) (probs[i] / sum);
    }

    // соберём индексы и отсортируем по вероятности
    std::vector<int> ids(n_vocab);
    for (int i = 0; i < n_vocab; ++i) {
        ids[i] = i;
    }
    std::partial_sort(ids.begin(), ids.begin() + 10, ids.end(), [&](int a, int b) { return probs[a] > probs[b]; });

   // 9) распечатаем top-10
    char piece[256];
    for (int r = 0; r < 10; ++r) {
        int id = ids[r];
        int n  = llama_token_to_piece(vocab,
                                     id,
                                     piece,
                                     sizeof(piece),
                                     /*special=*/true,
                                     /*clean=*/true);
        if (n < 0) {
            snprintf(piece, sizeof(piece), "<tok %d>", id);
        } else {
            piece[n] = '\0';
        }
        printf("%2d) id=%6d  p=%.6f  \"%s\"\n", r + 1, id, probs[id], piece);
    }

    if (word != nullptr) {
        // 10) распечатаем ещё интересующие токены
        std::vector<TokenInfo> tokens_info;

        // Получаем все префиксы строки
        std::vector<std::string> prefixes;
        size_t                   text_len = strlen(word);
        for (size_t len = 1; len <= text_len; len++) {
            char buf[256];
            memcpy(buf, word, len);
            buf[len] = '\0';
            prefixes.push_back(buf);
        }

        // Проходим по словарю и ищем все токены, которые совпадают с префиксами
        for (int id = 0; id < llama_vocab_n_tokens(vocab); ++id) {
            char piece[256];
            int  n = llama_token_to_piece(vocab, id, piece, sizeof(piece), true, true);
            if (n <= 0) {
                continue;
            }
            piece[n] = '\0';

            // проверка на совпадение с префиксом
            for (const auto & pref : prefixes) {
                if (strcmp(piece, pref.c_str()) == 0) {
                    tokens_info.push_back({ id, probs[id], piece });
                }
            }
        }

        // Сортируем по убыванию вероятности
        std::sort(
            tokens_info.begin(), tokens_info.end(), [](const TokenInfo & a, const TokenInfo & b) { return a.p > b.p; });

        // Вывод
        for (const auto & t : tokens_info) {
            if (t.p > 0.00000049f) {
                printf("id=%6d  p=%.6f  \"%s\"\n", t.id, t.p, t.piece.c_str());
            }
        }
    }
    
    // 11) cleanup
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
