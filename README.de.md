<div align="center">

# Summer.cpp

### Ein llama.cpp-Fork für GGUF-Modelle, die größer als der VRAM sind und über NVIDIA-GPU, DRAM und SSD ausgeführt werden

**Hierarchischer Speicher · geteilte GGUF-Dateien · lokale CUDA-Inferenz**

[日本語](README.md) · [English](README.en.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Français](README.fr.md) · **Deutsch**

[![Platform](https://img.shields.io/badge/platform-Linux-111827?logo=linux&logoColor=white)](#voraussetzungen)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76B900?logo=nvidia&logoColor=white)](#voraussetzungen)
[![Language](https://img.shields.io/badge/language-C%2B%2B17-00599C?logo=cplusplus&logoColor=white)](#manueller-build)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#lizenz)

</div>

> [!IMPORTANT]
> Die derzeit stabilste Konfiguration ist **VRAM + DRAM**. Auf einer GTX 1660 SUPER wurde `--vram-mib 3800 --dram-mib 6500` mit 0 MiB SSD-Platzierung validiert. Selektives SSD-Streaming auf Turing-GPUs ist weiterhin experimentell.

## Überblick

Summer.cpp ergänzt llama.cpp um ein hierarchisches Speicher-Backend und das dedizierte Programm `llama-tiered`. Große GGUF-Tensoren werden anhand ihres Einsatzzwecks und der Speicherbudgets verteilt.

| Ebene | Speicherort | Zweck |
|---|---|---|
| VRAM | CUDA-Gerätespeicher | Häufig verwendete Dense-Gewichte, Embeddings und Hot-Tensoren |
| DRAM | CUDA-gemappter Hostspeicher | Gewichte außerhalb des VRAM über Zero-Copy oder mapped pinned copies |
| SSD | Dateibasiertes GGUF-Mapping | Ausgewählte MoE-Experten bei Bedarf laden und häufige Experten im adaptiven Cache halten |

```text
GGUF file
   │ mmap
   ▼
Placement planner
   ├── VRAM: resident CUDA allocations
   ├── DRAM: mapped host memory / pinned copy fallback
   └── SSD : selected MoE slabs -> adaptive VRAM cache -> reusable scratch
```

## Validierte Konfiguration

| Eintrag | Wert |
|---|---|
| GPU | NVIDIA GeForce GTX 1660 SUPER |
| Compute capability | 7.5 |
| Modell | Qwen3.6-35B-A3B-UD-IQ1_M.gguf |
| GGUF-Größe | Etwa 9,35 GiB |
| VRAM-Budget | 3800 MiB |
| DRAM-Budget | 6500 MiB |
| SSD-Platzierung | 0 MiB |
| Ladezeit | Etwa 5,65 s |
| Prompt-Verarbeitung | Etwa 31,7 tokens/s |
| Token-Generierung | Etwa 27,7 tokens/s |

Diese Werte stammen aus einem Test mit kurzem Prompt. Modell, Kontext, Sampler, CPU, PCIe, Treiber und Hintergrundlast beeinflussen das Ergebnis.

## Voraussetzungen

- Linux; Ubuntu 22.04 oder 24.04 empfohlen
- NVIDIA-GPU und Treiber
- CUDA Toolkit mit `nvcc`
- CMake und C++17-Compiler
- Python 3.10 oder neuer
- GGUF-Modell
- Ausreichend SSD-Speicher
- Ausreichend System-RAM für die DRAM-Ebene

Für eine GTX 1660 SUPER und ein Modell mit ungefähr 9,35 GiB werden mindestens 16 GiB System-RAM empfohlen. Der DRAM-Fallback kann zusätzlich zum Datei-mmap eine mapped pinned copy anlegen.

```bash
nvidia-smi
nvcc --version
cmake --version
python3 --version
```

## Schnellinstallation

### 1. Abhängigkeiten installieren

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  git \
  pkg-config \
  python3
```

Installieren Sie bei Bedarf ein CUDA Toolkit, das zum NVIDIA-Treiber und zur GPU passt.

### 2. Repository klonen

```bash
git clone https://github.com/vnlpscale/Summer.cpp.git "$HOME/Summer.cpp"
cd "$HOME/Summer.cpp"
```

### 3. Bauen und installieren

```bash
bash scripts/install-summer.sh
```

Das Installationsskript wendet die DRAM-Fallback-Patches für Turing/GTX 16 an, baut `llama-tiered` mit CUDA im Release-Modus, installiert es nach `~/.local/bin`, entfernt alte SummerCLI-Befehle und erstellt `~/models`.

Für eine GTX 1660 SUPER:

```bash
CUDA_ARCH=75 FORCE_MMQ=ON bash scripts/install-summer.sh
```

Für Tensor-Core-GPUs ohne erzwungenes MMQ:

```bash
FORCE_MMQ=OFF bash scripts/install-summer.sh
```

### 4. PATH konfigurieren

```bash
grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" || \
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
source "$HOME/.bashrc"
hash -r
command -v llama-tiered
```

### 5. Modell ablegen

```bash
mkdir -p "$HOME/models"
cp /path/to/model.gguf "$HOME/models/"
```

Bei einem geteilten GGUF müssen alle Teile im selben Verzeichnis liegen.

### 6. Ausführen

```bash
llama-tiered \
  -m "$HOME/models/Qwen3.6-35B-A3B-UD-IQ1_M.gguf" \
  --vram-mib 3800 \
  --dram-mib 6500 \
  -n 128 \
  "Stelle dich vor."
```

Die CLI schreibt das Blocklogo von llama.cpp und das Summer.cpp-Banner auf Standardfehler. Der generierte Text auf Standardausgabe bleibt dadurch für Pipes geeignet.

## Manueller Build

```bash
cd "$HOME/Summer.cpp"
python3 scripts/apply-tiered-dram-pinned-fallback.py
python3 scripts/apply-tiered-dram-matmul-staging.py
python3 scripts/apply-tiered-no-prompt-echo.py
rm -rf build
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FORCE_MMQ=ON \
  -DGGML_BACKEND_DL=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_TESTS=OFF
cmake --build build --target llama-tiered -j"$(nproc)"
install -Dm755 build/bin/llama-tiered "$HOME/.local/bin/llama-tiered"
```

## Adaptiver Expert-Cache

Für große MoE-Modelle mit Gewichten auf SSD kann `--cache-mib` aktiviert werden. Die Cache-Kapazität ist Teil von `--vram-mib` und wird automatisch vom Budget der residenten Gewichte abgezogen.

```bash
llama-tiered \
  -m "$HOME/models/Laguna-S-2.1-UD-IQ1_S.gguf" \
  --vram-mib 3800 \
  --dram-mib 6500 \
  --cache-mib 1024 \
  -n 128 \
  "Hallo"
```

Der Cache lernt häufige Experten aus der Router-Historie beim Single-Row-Decode. Bei Multi-Row-Prompt-Batches wird er zur Vermeidung von Cache-Verschmutzung umgangen. Das Abschlussprotokoll zeigt Hit-Rate, H2D/D2D-Verkehr, Admissions und Evictions.

## Speicherbudgets abstimmen

Startwert für eine GTX 1660 SUPER:

```text
--vram-mib 3800 --dram-mib 6500
```

- `--vram-mib` nach GPU-Allokationsfehlern reduzieren.
- `--dram-mib` erhöhen, wenn Tensoren auf SSD landen.
- Bei zu wenig RAM ein kleineres oder stärker quantisiertes GGUF verwenden.
- Mehr VRAM reservieren, wenn die GPU auch den Desktop ausgibt.

Für maximale Stabilität sollte das Startprotokoll `SSD 0.00 MiB` anzeigen.

## Fehlerbehebung

- `build/bin/llama-tiered` fehlt: `bash scripts/install-summer.sh` erneut ausführen.
- `invalid argument` auf Turing/GTX 16: DRAM-pinned-Fallback anwenden und neu bauen.
- `tensor_state layout did not match expected source`: `ggml/src/ggml-cuda/tiered.cu` wiederherstellen und Installer erneut ausführen.
- `operation not supported`: `build` löschen und mit aktuellen Patches neu bauen.
- Illegaler CUDA-Speicherzugriff: aktualisieren, neu bauen und einen kurzen Prompt mit `compute-sanitizer --tool memcheck` prüfen.
- `summer: command not found`: Der unterstützte Befehl lautet `llama-tiered`; `~/.local/bin` muss in `PATH` enthalten sein.

## Status des SSD-Streamings

Die SSD-Ebene hält gestapelte MoE-Expert-Tensoren im GGUF-mmap und überträgt während `MUL_MAT_ID` nur den ausgewählten Expert-Slab in wiederverwendbaren VRAM-Scratch. Aktuelle Einschränkungen sind möglicherweise residente page-locked Quellseiten, Scratch in Größe des größten gestapelten Expert-Tensors, keine Überlappung von Übertragung und Berechnung, adaptiver Cache nur für Single-Row-Decode, serialisierte Graph-Ausführung für Kontexte mit gemeinsamem Modell sowie notwendige Hardwaretests pro GPU-Architektur.

GTX 1660 SUPER mit Laguna-S-2.1 IQ1_S wurde mit 1, 16 und 128 generierten Tokens sowie mit `compute-sanitizer` getestet. Auf anderen GPUs und Modellen sollte mit kurzen Generierungen begonnen werden.

## Bibliotheks-API

```cpp
#include "llama-tiered.h"

llama_model_params model_params = llama_model_default_params();
llama_tiered_memory_params memory = llama_tiered_memory_default_params();
memory.vram_budget_bytes = 3800ull * 1024 * 1024;
memory.dram_budget_bytes = 6500ull * 1024 * 1024;

llama_tiered_model * owner = llama_tiered_model_load_from_file(
        "model.gguf", model_params, memory);
if (!owner) {
    fprintf(stderr, "tiered load failed: %s\n", llama_tiered_last_error());
    return 1;
}
llama_model * model = llama_tiered_model_get_model(owner);
// llama_context erstellen und verwenden, solange owner lebt.
llama_tiered_model_free(owner);
```

Der von `llama_tiered_model_get_model()` gelieferte Zeiger ist geliehen. Er darf nicht direkt an `llama_model_free()` übergeben werden.

## Upstream und Lizenz

Summer.cpp basiert auf [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) und verwendet dieselbe MIT-Lizenz. Siehe [LICENSE](LICENSE).
