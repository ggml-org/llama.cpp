<div align="center">

# Summer.cpp

### Un fork de llama.cpp pour exécuter des modèles GGUF dépassant la VRAM sur GPU NVIDIA, DRAM et SSD

**Exécution à mémoire hiérarchisée · GGUF fractionné · inférence CUDA locale**

[日本語](README.md) · [English](README.en.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [한국어](README.ko.md) · [Español](README.es.md) · **Français** · [Deutsch](README.de.md)

[![Platform](https://img.shields.io/badge/platform-Linux-111827?logo=linux&logoColor=white)](#prérequis)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76B900?logo=nvidia&logoColor=white)](#prérequis)
[![Language](https://img.shields.io/badge/language-C%2B%2B17-00599C?logo=cplusplus&logoColor=white)](#compilation-manuelle)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#licence)

</div>

> [!IMPORTANT]
> La configuration actuellement la plus stable est **VRAM + DRAM**. Sur une GTX 1660 SUPER, `--vram-mib 3800 --dram-mib 6500` a été validé avec 0 MiB placé sur SSD. Le streaming SSD sélectif sur GPU Turing reste expérimental.

## Présentation

Summer.cpp ajoute à llama.cpp un backend de mémoire hiérarchisée et un exécutable dédié, `llama-tiered`. Les grands tenseurs GGUF sont répartis selon leur usage et les budgets mémoire.

| Niveau | Emplacement | Usage |
|---|---|---|
| VRAM | Mémoire CUDA du GPU | Poids denses fréquents, embeddings et tenseurs chauds |
| DRAM | Mémoire hôte mappée par CUDA | Poids ne tenant pas en VRAM, via zero-copy ou mapped pinned copy |
| SSD | Mapping GGUF adossé au fichier | Chargement à la demande des experts MoE sélectionnés et conservation des experts chauds en cache adaptatif |

```text
GGUF file
   │ mmap
   ▼
Placement planner
   ├── VRAM: resident CUDA allocations
   ├── DRAM: mapped host memory / pinned copy fallback
   └── SSD : selected MoE slabs -> adaptive VRAM cache -> reusable scratch
```

## Configuration validée

| Élément | Valeur |
|---|---|
| GPU | NVIDIA GeForce GTX 1660 SUPER |
| Compute capability | 7.5 |
| Modèle | Qwen3.6-35B-A3B-UD-IQ1_M.gguf |
| Taille GGUF | Environ 9,35 GiB |
| Budget VRAM | 3800 MiB |
| Budget DRAM | 6500 MiB |
| Placement SSD | 0 MiB |
| Temps de chargement | Environ 5,65 s |
| Traitement du prompt | Environ 31,7 tokens/s |
| Génération | Environ 27,7 tokens/s |

Ces chiffres correspondent à une mesure sur un prompt court. Le modèle, le contexte, le sampler, le CPU, le bus PCIe, le driver et la charge système modifient les résultats.

## Prérequis

- Linux ; Ubuntu 22.04 ou 24.04 recommandé
- GPU NVIDIA et driver
- CUDA Toolkit avec `nvcc`
- CMake et compilateur C++17
- Python 3.10 ou version ultérieure
- Modèle GGUF
- Espace SSD suffisant
- RAM système suffisante pour utiliser le niveau DRAM

Avec une GTX 1660 SUPER et un modèle d’environ 9,35 GiB, au moins 16 GiB de RAM système sont recommandés. Le fallback DRAM peut allouer une mapped pinned copy en plus du mmap du fichier.

```bash
nvidia-smi
nvcc --version
cmake --version
python3 --version
```

## Installation rapide

### 1. Dépendances

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  git \
  pkg-config \
  python3
```

Installez une version du CUDA Toolkit compatible avec le driver NVIDIA et le GPU si nécessaire.

### 2. Cloner

```bash
git clone https://github.com/vnlpscale/Summer.cpp.git "$HOME/Summer.cpp"
cd "$HOME/Summer.cpp"
```

### 3. Compiler et installer

```bash
bash scripts/install-summer.sh
```

L’installateur applique les patchs de fallback DRAM pour Turing/GTX 16, compile `llama-tiered` avec CUDA en mode Release, l’installe dans `~/.local/bin`, supprime les anciennes commandes SummerCLI et crée `~/models`.

Pour une GTX 1660 SUPER :

```bash
CUDA_ARCH=75 FORCE_MMQ=ON bash scripts/install-summer.sh
```

Pour un GPU Tensor Core ne nécessitant pas MMQ forcé :

```bash
FORCE_MMQ=OFF bash scripts/install-summer.sh
```

### 4. Configurer PATH

```bash
grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" || \
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
source "$HOME/.bashrc"
hash -r
command -v llama-tiered
```

### 5. Placer le modèle

```bash
mkdir -p "$HOME/models"
cp /path/to/model.gguf "$HOME/models/"
```

Pour un GGUF fractionné, placez toutes les parties dans le même répertoire.

### 6. Exécuter

```bash
llama-tiered \
  -m "$HOME/models/Qwen3.6-35B-A3B-UD-IQ1_M.gguf" \
  --vram-mib 3800 \
  --dram-mib 6500 \
  -n 128 \
  "Présente-toi."
```

La CLI écrit le logo en blocs de llama.cpp et la bannière Summer.cpp sur la sortie d’erreur. Le texte généré sur la sortie standard reste utilisable dans un pipeline.

## Compilation manuelle

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

## Cache adaptatif des experts

Pour les grands modèles MoE dont certains poids sont placés sur SSD, activez `--cache-mib`. Cette capacité fait partie de `--vram-mib` et est automatiquement retirée du budget des poids résidents.

```bash
llama-tiered \
  -m "$HOME/models/Laguna-S-2.1-UD-IQ1_S.gguf" \
  --vram-mib 3800 \
  --dram-mib 6500 \
  --cache-mib 1024 \
  -n 128 \
  "Bonjour"
```

Le cache apprend les experts chauds à partir de l’historique du routeur pendant le decode à une ligne. Il est contourné pour les prompt batches multi-lignes afin d’éviter la pollution. Le log final indique le taux de hit, le trafic H2D/D2D, les admissions et les évictions.

## Réglage des budgets mémoire

Point de départ pour une GTX 1660 SUPER :

```text
--vram-mib 3800 --dram-mib 6500
```

- Diminuez `--vram-mib` après une erreur d’allocation GPU.
- Augmentez `--dram-mib` si des tenseurs sont placés sur SSD.
- Utilisez un GGUF plus petit ou davantage quantifié si la RAM manque.
- Réservez plus de VRAM lorsque le GPU gère aussi l’affichage du bureau.

Pour privilégier la stabilité, utilisez une configuration dont le log affiche `SSD 0.00 MiB`.

## Dépannage

- `build/bin/llama-tiered` absent : relancez `bash scripts/install-summer.sh`.
- `invalid argument` sur Turing/GTX 16 : appliquez le fallback DRAM pinned puis recompilez.
- `tensor_state layout did not match expected source` : restaurez `ggml/src/ggml-cuda/tiered.cu`, puis relancez l’installateur.
- `operation not supported` : supprimez `build` et recompilez avec les patchs actuels.
- Accès mémoire CUDA illégal : mettez à jour, recompilez et testez un prompt court avec `compute-sanitizer --tool memcheck`.
- `summer: command not found` : la commande prise en charge est `llama-tiered`; vérifiez que `~/.local/bin` est dans `PATH`.

## État du streaming SSD

Le niveau SSD conserve les tenseurs stacked MoE expert dans le mmap GGUF et ne transfère que l’expert slab sélectionné vers un scratch VRAM réutilisable pendant `MUL_MAT_ID`. Les limites actuelles comprennent la résidence possible des pages source en RAM page-locked, un scratch dimensionné sur le plus grand tenseur stacked expert, l’absence de recouvrement transfert/calcul, un cache limité au decode à une ligne, la sérialisation des graphes pour les contextes partageant un modèle et la nécessité d’une validation physique par architecture GPU.

La GTX 1660 SUPER avec Laguna-S-2.1 IQ1_S a été testée avec 1, 16 et 128 tokens générés ainsi qu’avec `compute-sanitizer`. Commencez par de courtes générations sur les autres GPU et modèles.

## API de bibliothèque

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
// Créez et utilisez llama_context tant que owner reste vivant.
llama_tiered_model_free(owner);
```

Le pointeur renvoyé par `llama_tiered_model_get_model()` est emprunté. Ne le transmettez pas directement à `llama_model_free()`.

## Projet amont et licence

Summer.cpp repose sur [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) et utilise la même licence MIT. Consultez [LICENSE](LICENSE).
