//#include "llama.h"
//#include "get-model.h"
//
//#include <cstdlib>
//
//int main(int argc, char *argv[] ) {
//    auto * model_path = "C:\\NHKI\\mymodell\\kilocal\\Teuken-7.5B-F16d.gguf";
//    auto * file = fopen(model_path, "r");
//    if (file == nullptr) {
//        fprintf(stderr, "no model at '%s' found\n", model_path);
//        return EXIT_FAILURE;
//    }
//
//    fprintf(stderr, "using '%s'\n", model_path);
//    fclose(file);
//
//    llama_backend_init();
//    auto params = llama_model_params{};
//    //params.use_mmap = false;
//    params.vocab_only=true;
//    params.main_gpu=-1;
//    params.n_gpu_layers=0;
//    params.progress_callback = [](float progress, void * ctx){
//        (void) ctx;
//        return progress > 0.50;
//    };
//    auto * model = llama_model_load_from_file(model_path, params);
//
//    model.
//
//    llama_backend_free();
//    return model == nullptr ? EXIT_SUCCESS : EXIT_FAILURE;
//}
//

#include "llama.h"
#include "common.h"
#include "console.h"
#include "../src/llama-vocab-sentencepiece.h"
#include "../src/unicode.h"

#include <cassert>
#include <codecvt>
#include <cstdio>
#include <cstring>
#include <locale>
#include <string>
#include <thread>
#include <vector>
#include <atomic>

int main(int argc, char ** argv) {

    const std::string fname = "C:\\NHKI\\mymodell\\kilocal\\Teuken-7.5B-BF16-CM.gguf";

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

    const llama_vocab * vocab = llama_model_get_vocab(model);

#ifdef _WIN32
    // We need this for unicode console support
    console::init(false, false);
    atexit([]() { console::cleanup(); });
#endif

    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::string str="…";
    str="n ⋅Ball";
   // str="Hallo wer bist Du?";

    std::vector<llama_token> tokens = common_tokenize(ctx, str, false, true);

    std::vector<llama_token> tokensbos = common_tokenize(ctx, str,true, true);

    {
   std::vector<llama_chat_message> conversation {
     {"user", "Hello, how are you?"},
     {"assistant", "I'm doing great. How can I help you today?"},
     {"user", "I'd like to show off how chat templating works!"},
   };

   const char* cl=llama_model_chat_template(model,NULL);

   std::string formatted_chat;

  formatted_chat.resize(1024);
  bool add_generation_prompt=false;
  int res = llama_chat_apply_template(
            cl,
            conversation.data(),
            conversation.size(),
            add_generation_prompt,
            formatted_chat.data(),
            formatted_chat.size()
        );
        formatted_chat.resize(res);
        std::string output(formatted_chat.data(), formatted_chat.size());

       std::string check= "System: A chat between a human and an artificial intelligence assistant.The assistant gives helpful and polite answers to the human's questions.\nUser: Hello, how are you?\nAssistant: I'm doing great. How can I help you today?</s>\nUser: I'd like to show off how chat templating works!\n";
       if (check != output)
       {
           size_t z=check.size();
           size_t y=output.size();
           for (size_t i = 0; i < z && i < y; i++)
           {
               if (check[i] != output[i])
               {
                   char a=check[i];
                   char b=output[i];
               }
           }
       }
    }

     {

     std::vector<llama_chat_message> conversation {
        {"system", "DE"},
        {"user", "Wie geht es dir?"},
        {"assistant", "Mir geht es gut. Kann ich Dir helfen?"},
        {"user", "Ich möchte gerne wissen wie chat templates funktionieren!"},
     };

   const char* cl=llama_model_chat_template(model,NULL);

   std::string formatted_chat;

  formatted_chat.resize(1024);
  bool add_generation_prompt=false;
  int res = llama_chat_apply_template(
            cl,
            conversation.data(),
            conversation.size(),
            add_generation_prompt,
            formatted_chat.data(),
            formatted_chat.size()
        );
        formatted_chat.resize(res);
        std::string output(formatted_chat.data(), formatted_chat.size());

       std::string check= "System: Ein Gespräch zwischen einem Menschen und einem Assistenten mit künstlicher Intelligenz. Der Assistent gibt hilfreiche und höfliche Antworten auf die Fragen des Menschen.\nUser: Wie geht es dir?\nAssistant: Mir geht es gut. Kann ich Dir helfen?</s>\nUser: Ich möchte gerne wissen wie chat templates funktionieren!\n";
       if (check != output)
       {
           size_t z=check.size();
           size_t y=output.size();
           for (size_t i = 0; i < z && i < y; i++)
           {
               if (check[i] != output[i])
               {
                   char a=check[i];
                   char b=output[i];
               }
           }

       }
    }


     
     {

     std::vector<llama_chat_message> conversation {
        {"system", "DE"},
        {"user", "Wie geht es dir?"},
        {"assistant", "Mir geht es gut. Kann ich Dir helfen?"},
        {"user", "Ich möchte gerne wissen wie chat templates funktionieren!"},
     };

   const char* cl=llama_model_chat_template(model,NULL);

   std::string formatted_chat;

  formatted_chat.resize(1024);
  bool add_generation_prompt=true;
  int res = llama_chat_apply_template(
            cl,
            conversation.data(),
            conversation.size(),
            add_generation_prompt,
            formatted_chat.data(),
            formatted_chat.size()
        );
        formatted_chat.resize(res);
        std::string output(formatted_chat.data(), formatted_chat.size());

       std::string check="System: Ein Gespräch zwischen einem Menschen und einem Assistenten mit künstlicher Intelligenz. Der Assistent gibt hilfreiche und höfliche Antworten auf die Fragen des Menschen.\nUser: Wie geht es dir?\nAssistant: Mir geht es gut. Kann ich Dir helfen?</s>\nUser: Ich möchte gerne wissen wie chat templates funktionieren!\nAssistant: ";

       if (check != output)
       {
           size_t z=check.size();
           size_t y=output.size();
           for (size_t i = 0; i < z && i < y; i++)
           {
               if (check[i] != output[i])
               {
                   char a=check[i];
                   char b=output[i];
               }
           }

       }
    }

  //  std::vector<int> token_ids;
  //  sp_encode(str,token_ids);

    //{
    //    sentencepiece::SentencePieceProcessor processor;
    //    const auto status = processor.Load("C:/NHKI/mymodell/kilocal/teuken/tokenizer.model");
    //    if (!status.ok()) {
    //        //std::cerr << status.ToString() << std::endl;
    //        // error
    //    }
    //    std::vector<int> token_ids;
    //    processor.Encode(str, &token_ids);
    //    for (const int id : token_ids) {
    //        // std::cout << token << std::endl;
    //    }
    //}
    //{
    //    std::string filename="C:/NHKI/mymodell/kilocal/teuken/tokenizer.model";
    //    auto input = sentencepiece::filesystem::NewReadableFile(filename, true);
    //    std::string serialized;
    //    if (!input->ReadAll(&serialized)) {
    //        // hh
    //    }
    //    sentencepiece::SentencePieceProcessor processor;
    //    const auto status = processor.LoadFromSerializedProto(serialized);
    //    std::vector<int> token_ids;
    //    processor.Encode(str, &token_ids);
    //    for (const int id : token_ids) {
    //        // std::cout << token << std::endl;
    //    }
    //}


    //auto model_proto = std::make_unique<sentencepiece::ModelProto>();

    //model_proto->ParseFromArray(serialized.data(), serialized.size())

    llama_model_free(model);
    llama_free(ctx);

    llama_backend_free();

    return 0;
}
