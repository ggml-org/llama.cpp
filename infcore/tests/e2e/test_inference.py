"""E2E приёмочный тест: реальный инференс через РАБОТАЮЩИЙ gateway.

Требует поднятый шлюз и реальную модель — поэтому по умолчанию SKIP. Запускать на
целевом хосте после деплоя (шаг приёмки §7 AUDIT: загрузка + инференс + мультимодальность
+ embeddings). Веса в репозиторий не кладём — тест не может быть self-contained.

Переменные окружения:
  INFCORE_URL          базовый URL шлюза (напр. http://127.0.0.1:8080)   [обязательно]
  INFCORE_KEY          API-ключ (Bearer)                                  [обязательно]
  INFCORE_E2E_MODEL    logical_name текстовой модели для chat             [обязательно]
  INFCORE_E2E_EMBED    logical_name embedding-модели (опц.)
  INFCORE_E2E_VISION   logical_name vision-модели + INFCORE_E2E_IMAGE (data URL/URL) (опц.)

Пример:
  INFCORE_URL=http://127.0.0.1:8080 INFCORE_KEY=$ADMIN INFCORE_E2E_MODEL=qwen3-moe-a3b \
    pytest infcore/tests/e2e -v
"""
import os
import sys

import pytest

# используем наш же SDK (dogfooding); путь к sdk/python относительно этого файла
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "sdk", "python"))

URL = os.environ.get("INFCORE_URL")
KEY = os.environ.get("INFCORE_KEY")
MODEL = os.environ.get("INFCORE_E2E_MODEL")

pytestmark = pytest.mark.skipif(
    not (URL and KEY and MODEL),
    reason="e2e: задайте INFCORE_URL, INFCORE_KEY, INFCORE_E2E_MODEL (нужен работающий gateway)",
)


@pytest.fixture(scope="module")
def client():
    from infcore import Client
    return Client(URL, api_key=KEY, timeout=float(os.environ.get("INFCORE_E2E_TIMEOUT", "180")))


def test_models_lists_target(client):
    ids = [m["id"] for m in client.models()]
    assert MODEL in ids, f"модель {MODEL} не видна роли ключа: {ids}"


def test_chat_nonstream(client):
    out = client.chat(MODEL, [{"role": "user", "content": "Ответь одним словом: столица России?"}])
    assert isinstance(out, str) and out.strip(), "пустой ответ chat"


def test_chat_stream(client):
    chunks = list(client.chat_stream(MODEL, [{"role": "user", "content": "Считай от 1 до 5."}]))
    assert chunks, "стрим не отдал ни одной дельты"
    assert "".join(chunks).strip(), "стрим пустой"


def test_embeddings():
    embed_model = os.environ.get("INFCORE_E2E_EMBED")
    if not embed_model:
        pytest.skip("INFCORE_E2E_EMBED не задан")
    from infcore import Client
    c = Client(URL, api_key=KEY)
    vecs = c.embeddings(embed_model, ["первый текст", "второй текст"])
    assert len(vecs) == 2 and all(len(v) > 0 for v in vecs), "неверная форма embeddings"


def test_vision_chat():
    vision_model = os.environ.get("INFCORE_E2E_VISION")
    image = os.environ.get("INFCORE_E2E_IMAGE")
    if not (vision_model and image):
        pytest.skip("INFCORE_E2E_VISION / INFCORE_E2E_IMAGE не заданы")
    from infcore import Client
    c = Client(URL, api_key=KEY)
    out = c.chat(vision_model, [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Что на изображении? Кратко."},
            {"type": "image_url", "image_url": {"url": image}},
        ],
    }])
    assert isinstance(out, str) and out.strip(), "пустой ответ vision-chat"
