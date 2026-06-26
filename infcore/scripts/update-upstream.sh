#!/usr/bin/env bash
# infcore — drop-in обновление движка из оригинального llama.cpp.
# Модель «обернуть, не трогая ядро»: апдейт = слияние release-тега апстрима.
# Запускать на машине сборки с доступом в интернет.
#
#   ./infcore/scripts/update-upstream.sh b1234
#
# Шаги:
#   1) фетч тегов апстрима;  2) слияние тега в ветку infcore;
#   3) разрешить конфликты (ожидаются только в cmake/корне и в удалённом
#      compliance-наборе — см. infcore/docs/COMPLIANCE.md);
#   4) пересборка профилем и прогон тестов + egress-тест;
#   5) обновить UPSTREAM_COMMIT/NOTICE/SBOM.
set -euo pipefail
TAG="${1:?укажите release-тег апстрима, напр. b1234}"
git remote get-url upstream >/dev/null 2>&1 || \
  git remote add upstream https://github.com/ggml-org/llama.cpp.git
git fetch --tags upstream
echo "Слияние upstream тега ${TAG} в $(git branch --show-current)..."
git merge --no-ff "${TAG}" || {
  echo "Есть конфликты — разрешите их (cmake/корень + compliance-набор), затем 'git commit'." >&2
  exit 1
}
echo "Слияние ок. Пересоберите: ./infcore/scripts/build.sh и прогоните тесты + egress."
