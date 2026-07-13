# model-toolkit — офлайн-подготовка моделей (build-time, НЕ рантайм)

Тонкие обёртки над инструментами llama.cpp для подготовки GGUF-весов в контуре:
квантование, разбиение/слияние шардов, importance-matrix, слияние LoRA. Эти утилиты
собираются профилем (`LLAMA_BUILD_TOOLS=ON`), но **в рантайм-образ не входят** (Dockerfile
копирует только `infcore_gateway`, `infcore-cli`, `llama-server`) — работа с весами
делается отдельно от боевого шлюза, на подготовительном хосте.

Всё офлайн: на вход — локальный `.gguf` (сконвертированный из HF конвертером апстрима
`convert_hf_to_gguf.py`), на выход — локальный `.gguf`. Никаких сетевых загрузок.

## Использование
```sh
# каталог сборки с бинарями (по умолчанию ./build/bin от корня форка)
export INFCORE_BUILD=/path/to/build

infcore/model-toolkit/model-toolkit.sh quantize  model-f16.gguf  model-Q4_K_M.gguf  Q4_K_M
infcore/model-toolkit/model-toolkit.sh split      model.gguf      model-shard        --split-max-size 20G
infcore/model-toolkit/model-toolkit.sh merge      model-shard-00001-of-00003.gguf   model-merged.gguf
infcore/model-toolkit/model-toolkit.sh imatrix    -m model-f16.gguf -f calib.txt -o model.imatrix
infcore/model-toolkit/model-toolkit.sh export-lora -m base.gguf --lora adapter.gguf -o merged.gguf
```

`quantize` с imatrix (лучшее качество на низких битах):
```sh
infcore/model-toolkit/model-toolkit.sh quantize --imatrix model.imatrix model-f16.gguf out-Q4_K_M.gguf Q4_K_M
```

Типы квантования (частые): `Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`, `Q4_0`, `Q3_K_M`, `Q2_K`.
Полный список — `model-toolkit.sh quantize --help`.

## Соответствие стратегии
Обёртки не редактируют апстрим; вызывают штатные `llama-quantize` / `llama-gguf-split`
/ `llama-imatrix` / `llama-export-lora` из нашей же сборки. При drop-in обновлении
движка обёртки продолжают работать (имена бинарей стабильны).
