<div align="center">

# Summer.cpp

### Un fork de llama.cpp para ejecutar modelos GGUF mayores que la VRAM entre GPU NVIDIA, DRAM y SSD

**Ejecución con memoria jerárquica · GGUF dividido · inferencia CUDA local**

[日本語](README.md) · [English](README.en.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [한국어](README.ko.md) · **Español** · [Français](README.fr.md) · [Deutsch](README.de.md)

[![Platform](https://img.shields.io/badge/platform-Linux-111827?logo=linux&logoColor=white)](#requisitos)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76B900?logo=nvidia&logoColor=white)](#requisitos)
[![Language](https://img.shields.io/badge/language-C%2B%2B17-00599C?logo=cplusplus&logoColor=white)](#compilación-manual)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#licencia)

</div>

> [!IMPORTANT]
> La configuración más estable actualmente es **VRAM + DRAM**. En una GTX 1660 SUPER se validó `--vram-mib 3800 --dram-mib 6500` con 0 MiB asignados al SSD. El streaming selectivo desde SSD en GPU Turing sigue siendo experimental.

## Descripción

Summer.cpp añade a llama.cpp un backend de memoria jerárquica y el ejecutable dedicado `llama-tiered`. Los tensores GGUF grandes se distribuyen según el uso y los presupuestos de memoria.

| Nivel | Ubicación | Uso |
|---|---|---|
| VRAM | Memoria de dispositivo CUDA | Pesos densos frecuentes, embeddings y tensores calientes |
| DRAM | Memoria host mapeada por CUDA | Pesos que no caben en VRAM mediante zero-copy o mapped pinned copy |
| SSD | Mapeo GGUF respaldado por archivo | Carga bajo demanda de expertos MoE seleccionados y caché adaptativa de expertos calientes |

```text
GGUF file
   │ mmap
   ▼
Placement planner
   ├── VRAM: resident CUDA allocations
   ├── DRAM: mapped host memory / pinned copy fallback
   └── SSD : selected MoE slabs -> adaptive VRAM cache -> reusable scratch
```

## Configuración validada

| Elemento | Valor |
|---|---|
| GPU | NVIDIA GeForce GTX 1660 SUPER |
| Compute capability | 7.5 |
| Modelo | Qwen3.6-35B-A3B-UD-IQ1_M.gguf |
| Tamaño GGUF | Aproximadamente 9.35 GiB |
| Presupuesto VRAM | 3800 MiB |
| Presupuesto DRAM | 6500 MiB |
| Ubicación SSD | 0 MiB |
| Tiempo de carga | Aproximadamente 5.65 s |
| Procesamiento del prompt | Aproximadamente 31.7 tokens/s |
| Generación | Aproximadamente 27.7 tokens/s |

Son resultados de una prueba con un prompt corto. El modelo, contexto, sampler, CPU, PCIe, driver y carga en segundo plano afectan el rendimiento.

## Requisitos

- Linux; se recomienda Ubuntu 22.04 o 24.04
- GPU NVIDIA y driver
- CUDA Toolkit con `nvcc`
- CMake y compilador compatible con C++17
- Python 3.10 o posterior
- Modelo GGUF
- Espacio SSD suficiente
- RAM del sistema suficiente para usar el nivel DRAM

Para una GTX 1660 SUPER y un modelo de unos 9.35 GiB se recomiendan al menos 16 GiB de RAM del sistema. El fallback de DRAM puede asignar una mapped pinned copy adicional al mmap del archivo.

```bash
nvidia-smi
nvcc --version
cmake --version
python3 --version
```

## Instalación rápida

### 1. Dependencias

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  git \
  pkg-config \
  python3
```

Instala una versión de CUDA Toolkit compatible con el driver y la GPU si todavía no está disponible.

### 2. Clonar

```bash
git clone https://github.com/vnlpscale/Summer.cpp.git "$HOME/Summer.cpp"
cd "$HOME/Summer.cpp"
```

### 3. Compilar e instalar

```bash
bash scripts/install-summer.sh
```

El instalador aplica los parches de fallback DRAM para Turing/GTX 16, compila `llama-tiered` con CUDA en modo Release, lo instala en `~/.local/bin`, elimina comandos antiguos de SummerCLI y crea `~/models`.

Para una GTX 1660 SUPER:

```bash
CUDA_ARCH=75 FORCE_MMQ=ON bash scripts/install-summer.sh
```

Para GPU Tensor Core que no necesitan MMQ forzado:

```bash
FORCE_MMQ=OFF bash scripts/install-summer.sh
```

### 4. Configurar PATH

```bash
grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" || \
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
source "$HOME/.bashrc"
hash -r
command -v llama-tiered
```

### 5. Colocar el modelo

```bash
mkdir -p "$HOME/models"
cp /path/to/model.gguf "$HOME/models/"
```

Para un GGUF dividido, coloca todas las partes en el mismo directorio.

### 6. Ejecutar

```bash
llama-tiered \
  -m "$HOME/models/Qwen3.6-35B-A3B-UD-IQ1_M.gguf" \
  --vram-mib 3800 \
  --dram-mib 6500 \
  -n 128 \
  "Preséntate."
```

La CLI escribe el logotipo de bloques de llama.cpp y el banner de Summer.cpp en la salida de error. El texto generado en la salida estándar puede seguir usándose en pipelines.

## Compilación manual

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

## Caché adaptativa de expertos

Para modelos MoE grandes con pesos en SSD, activa `--cache-mib`. Esta capacidad forma parte de `--vram-mib` y se descuenta automáticamente del presupuesto de pesos residentes.

```bash
llama-tiered \
  -m "$HOME/models/Laguna-S-2.1-UD-IQ1_S.gguf" \
  --vram-mib 3800 \
  --dram-mib 6500 \
  --cache-mib 1024 \
  -n 128 \
  "Hola"
```

La caché aprende expertos calientes del historial del router durante decode de una fila. Se omite durante lotes de prompt de varias filas para evitar contaminación. El log final muestra tasa de aciertos, tráfico H2D/D2D, admisiones y expulsiones.

## Ajuste de memoria

Punto de partida para GTX 1660 SUPER:

```text
--vram-mib 3800 --dram-mib 6500
```

- Reduce `--vram-mib` si hay errores de asignación de GPU.
- Aumenta `--dram-mib` si aparecen tensores en SSD.
- Usa un GGUF más pequeño o más cuantizado si falta RAM.
- Reserva más VRAM si la GPU también controla el escritorio.

Para priorizar estabilidad, utiliza una configuración cuyo log indique `SSD 0.00 MiB`.

## Solución de problemas

- Falta `build/bin/llama-tiered`: vuelve a ejecutar `bash scripts/install-summer.sh`.
- `invalid argument` en Turing/GTX 16: aplica el fallback DRAM pinned y recompila.
- `tensor_state layout did not match expected source`: restaura `ggml/src/ggml-cuda/tiered.cu` y repite la instalación.
- `operation not supported`: elimina `build` y recompila con los parches actuales.
- Acceso ilegal de memoria CUDA: actualiza, recompila y prueba un prompt corto con `compute-sanitizer --tool memcheck`.
- `summer: command not found`: el ejecutable compatible es `llama-tiered`; verifica `~/.local/bin` en `PATH`.

## Estado del streaming SSD

El nivel SSD conserva tensores stacked MoE en el mmap de GGUF y transfiere únicamente el expert slab seleccionado a un scratch VRAM reutilizable durante `MUL_MAT_ID`. Las limitaciones actuales incluyen páginas fuente residentes y bloqueadas, scratch del tamaño del mayor tensor stacked expert, ausencia de solapamiento entre transferencia y cálculo, caché adaptativa limitada a decode de una fila, ejecución serializada para contextos que comparten modelo y validación física por arquitectura de GPU.

GTX 1660 SUPER con Laguna-S-2.1 IQ1_S se ha probado con 1, 16 y 128 tokens generados y con `compute-sanitizer`. Empieza con generaciones cortas en otras GPU y modelos.

## API de biblioteca

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
// Crea y usa llama_context mientras owner siga vivo.
llama_tiered_model_free(owner);
```

El puntero devuelto por `llama_tiered_model_get_model()` es prestado. No lo pases directamente a `llama_model_free()`.

## Upstream y licencia

Summer.cpp se basa en [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) y utiliza la misma licencia MIT. Consulta [LICENSE](LICENSE).
