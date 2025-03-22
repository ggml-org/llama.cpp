# llama.cpp/example/simple-tts

The purpose of this example is to demonstrate a minimal usage of llama.cpp to generate speech from text using the outetts series of models.

## Usage

To use this example you will need the Text to Codes model [`OuteTTS-0.2-500M-q8_0.gguf`](https://huggingface.co/OuteAI/OuteTTS-0.2-500M-GGUF/blob/main/OuteTTS-0.2-500M-Q8_0.gguf), the Wav Tokenizer model [`WavTokenizer-Large-75-F16.gguf`](https://huggingface.co/ggml-org/WavTokenizer/blob/main/WavTokenizer-Large-75-F16.gguf), and a [outetts voice file](https://github.com/edwko/OuteTTS/tree/main/outetts/version/v1/default_speakers).

Once you have the files you can run the following command to generate speech from text:

```bash
./llama-simple-tts -m  ./OuteTTS-0.2-500M-q8_0.gguf -mv ./WavTokenizer-Large-75-F16.gguf -v ./en_male_1.json -p "Hello, world!"
```
