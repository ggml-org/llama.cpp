// infcore gateway — корпоративная лицензия.
#include "json_schema.hpp"

#include <regex>

using json = nlohmann::json;

namespace infcore {

namespace {

bool type_matches(const json& v, const std::string& t) {
    if (t == "object")  return v.is_object();
    if (t == "array")   return v.is_array();
    if (t == "string")  return v.is_string();
    if (t == "boolean") return v.is_boolean();
    if (t == "integer") return v.is_number_integer();
    if (t == "number")  return v.is_number();
    return true;   // неизвестный тип не валидируем
}

const json* resolve_ref(const std::string& ref, const json& root) {
    // Поддерживаем только локальные ссылки вида "#/$defs/Name".
    const std::string pfx = "#/$defs/";
    if (ref.compare(0, pfx.size(), pfx) != 0) return nullptr;
    std::string name = ref.substr(pfx.size());
    if (!root.contains("$defs") || !root.at("$defs").contains(name)) return nullptr;
    return &root.at("$defs").at(name);
}

void validate(const json& inst, const json& schema, const json& root,
              const std::string& path, std::vector<std::string>& errs) {
    const std::string P = path.empty() ? "(root)" : path;
    if (schema.contains("$ref")) {
        const json* target = resolve_ref(schema.at("$ref").get<std::string>(), root);
        if (target) validate(inst, *target, root, path, errs);
        return;
    }

    if (schema.contains("type")) {
        const std::string t = schema.at("type").get<std::string>();
        if (!type_matches(inst, t)) {
            errs.push_back(P + ": ожидался тип '" + t + "'");
            return;   // дальше проверять бессмысленно
        }
    }

    if (schema.contains("enum")) {
        bool ok = false;
        for (const auto& e : schema.at("enum")) if (e == inst) { ok = true; break; }
        if (!ok) errs.push_back(P + ": значение вне списка enum");
    }

    if (inst.is_number()) {
        double x = inst.get<double>();
        if (schema.contains("minimum") && x < schema.at("minimum").get<double>())
            errs.push_back(P + ": меньше minimum");
        if (schema.contains("maximum") && x > schema.at("maximum").get<double>())
            errs.push_back(P + ": больше maximum");
    }

    if (inst.is_string() && schema.contains("pattern")) {
        try {
            std::regex re(schema.at("pattern").get<std::string>());
            if (!std::regex_search(inst.get<std::string>(), re))
                errs.push_back(P + ": не соответствует pattern");
        } catch (const std::exception&) { /* некорректный pattern в схеме - пропускаем */ }
    }

    if (inst.is_array()) {
        if (schema.contains("minItems") && inst.size() < schema.at("minItems").get<size_t>())
            errs.push_back(P + ": элементов меньше minItems");
        if (schema.contains("items")) {
            const json& isch = schema.at("items");
            for (size_t i = 0; i < inst.size(); ++i)
                validate(inst[i], isch, root, path + "/" + std::to_string(i), errs);
        }
    }

    if (inst.is_object()) {
        if (schema.contains("required"))
            for (const auto& r : schema.at("required")) {
                const std::string key = r.get<std::string>();
                if (!inst.contains(key)) errs.push_back(P + ": отсутствует обязательное поле '" + key + "'");
            }

        const bool has_props = schema.contains("properties");
        const bool addl_false = schema.contains("additionalProperties") &&
                                schema.at("additionalProperties").is_boolean() &&
                                !schema.at("additionalProperties").get<bool>();

        for (auto it = inst.begin(); it != inst.end(); ++it) {
            const std::string key = it.key();
            if (has_props && schema.at("properties").contains(key)) {
                validate(it.value(), schema.at("properties").at(key), root, path + "/" + key, errs);
            } else if (addl_false) {
                errs.push_back(P + ": неизвестное поле '" + key + "'");
            }
        }
    }
}

}  // namespace

std::vector<std::string> json_schema_validate(const json& instance, const json& schema) {
    std::vector<std::string> errs;
    validate(instance, schema, schema, "", errs);
    return errs;
}

}  // namespace infcore
