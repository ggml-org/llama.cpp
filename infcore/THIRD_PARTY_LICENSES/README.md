# Тексты лицензий третьих сторон

Полные тексты лицензий включённых open-source компонентов. Тексты извлечены
непосредственно из исходных файлов компонентов в дереве форка (`./LICENSE`,
`./vendor/`). Машиночитаемая опись — в `../sbom.cdx.json` (CycloneDX).

| Компонент | SPDX-идентификатор | Файл | Версия | Источник текста |
|-----------|--------------------|------|--------|-----------------|
| llama.cpp / ggml (ggml поставляется вместе с llama.cpp) | MIT | [ggml-llama.cpp.txt](ggml-llama.cpp.txt) | e8ecce5 | `./LICENSE` (корень форка) |
| nlohmann/json | MIT | [nlohmann-json.txt](nlohmann-json.txt) | 3.12.0 | `vendor/nlohmann/json.hpp` (SPDX-заголовок) |
| cpp-httplib | MIT | [cpp-httplib.txt](cpp-httplib.txt) | 0.48.0 | `vendor/cpp-httplib/LICENSE` |
| stb (stb_image) | MIT OR Unlicense | [stb.txt](stb.txt) | 2.30 | конец `vendor/stb/stb_image.h` |
| miniaudio | Unlicense OR MIT-0 | [miniaudio.txt](miniaudio.txt) | 0.11.25 | конец `vendor/miniaudio/miniaudio.h` |
| sheredom / subprocess.h | Unlicense | [sheredom.txt](sheredom.txt) | — | начало `vendor/sheredom/subprocess.h` |

Примечания:
- stb и miniaudio — двойное лицензирование; допускается выбор любой из указанных
  лицензий (для дистрибуции сохранён полный текст обеих альтернатив).
- sheredom/subprocess.h распространяется в public domain (Unlicense); версия в
  заголовке не проставлена, идентификация — по upstream GitHub.
