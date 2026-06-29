// infcore gateway — корпоративная лицензия.
// Минимальный валидатор JSON-Schema (подмножество draft 2020-12, которое реально
// используется в gateway.schema.json): type, required, properties,
// additionalProperties:false, enum, items, minimum/maximum, minItems, pattern,
// $ref ("#/$defs/..."). Без внешних зависимостей (offline-контур).
#pragma once

#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace infcore {

// Возвращает список человекочитаемых ошибок (пустой = инстанс валиден по схеме).
std::vector<std::string> json_schema_validate(const nlohmann::json& instance,
                                              const nlohmann::json& schema);

}  // namespace infcore
