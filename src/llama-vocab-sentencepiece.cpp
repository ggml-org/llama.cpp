#include "sentencepiece_processor.h"
//#include "filesystem.h"
#include "llama-vocab-sentencepiece.h"
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>



std::unique_ptr<sentencepiece::SentencePieceProcessor> processor;

int sp_init(const std::string& sp_binary)
 {
        if (processor.get()==NULL)
        {
                processor.reset(new sentencepiece::SentencePieceProcessor);
                const auto status = processor->LoadFromSerializedProto(sp_binary);
                if (!status.ok()) {
                    //std::cerr << status.ToString() << std::endl;
                    // error
                     throw  std::invalid_argument("sentencepiece not initialized");
                    return 0;
               }
        }
        return 1;
  }

 int sp_encode(const std::string& str,std::vector<int32_t>& token_ids)
 {
        if (processor.get()==NULL)
        {
               throw  std::invalid_argument("sentencepiece not initialized");
        }
       
        processor->Encode(str, &token_ids);
        return 0;
  }
