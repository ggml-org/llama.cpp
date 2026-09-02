#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <optional>

/**
 * 解析 GBNF 语法并根据 root 规则的结构修复 input 字符串。
 *
 * 逻辑：
 * 1. 从 thought 规则中提取结束常量（如 "<channel|>"）。
 * 2. 从 root 规则中分析 (thought | )? 与 response-format-schema 之间的常量或 space 规则。
 * 3. 在 input 中定位结束符，并插入缺失的内容。
 */
void fix_thought_json_transition(const std::string& gbnf_grammar, std::string& input) {
    std::string_view gbnf(gbnf_grammar);

    // --- 1. 提取 thought 规则的结束标志位 ---
    // 找到 thought ::= ... 并取最后一个引号内的字符串
    size_t thought_def = gbnf.find("thought ::= ");
    if (thought_def == std::string_view::npos) return;

    size_t thought_eol = gbnf.find('\n', thought_def);
    std::string_view thought_line = gbnf.substr(thought_def, thought_eol - thought_def);

    size_t last_quote_end = thought_line.rfind('"');
    if (last_quote_end == std::string_view::npos) return;
    size_t last_quote_start = thought_line.rfind('"', last_quote_end - 1);
    if (last_quote_start == std::string_view::npos) return;

    std::string thought_end_tag{thought_line.substr(last_quote_start + 1, last_quote_end - last_quote_start - 1)};

    // --- 2. 分析 root 规则中间件 ---
    size_t root_def = gbnf.find("root ::= ");
    if (root_def == std::string_view::npos) return;

    size_t root_eol = gbnf.find('\n', root_def);
    std::string_view root_line = gbnf.substr(root_def, root_eol - root_def);

    // 寻找 (thought | )? 之后，response-format-schema 之前的内容
    std::string_view anchor = "(thought | )?";
    size_t anchor_pos = root_line.find(anchor);
    if (anchor_pos == std::string_view::npos) return;

    size_t schema_pos = root_line.find("response-format-schema");
    if (schema_pos == std::string_view::npos) return;

    std::string_view middle_part = root_line.substr(anchor_pos + anchor.length(), schema_pos - (anchor_pos + anchor.length()));

    // 解析中间件中的常量字符串和 space 变量
    std::string to_insert;
    size_t cursor = 0;
    while (cursor < middle_part.length()) {
        if (middle_part[cursor] == '"') {
            size_t next_q = middle_part.find('"', cursor + 1);
            if (next_q != std::string_view::npos) {
                to_insert += middle_part.substr(cursor + 1, next_q - cursor - 1);
                cursor = next_q + 1;
            } else break;
        } else if (middle_part.compare(cursor, 5, "space") == 0) {
            to_insert += "\n"; // 将 GBNF 的 space 规则默认解析为一个换行符，因为 ```json 更适合它
            cursor += 5;
        } else {
            cursor++;
        }
    }

    // --- 3. 修改 input 字符串 ---
    size_t idx = input.find(thought_end_tag);
    if (idx == std::string::npos) return;

    size_t insert_pos = idx + thought_end_tag.length();

    // 检查是否已经存在该内容，避免重复插入
    if (insert_pos < input.length() && !to_insert.empty()) {
        // 如果 input 对应位置还没这段内容，则插入
        if (input.compare(insert_pos, to_insert.length(), to_insert) != 0) {
            input.insert(insert_pos, to_insert);
        }
    } else if (insert_pos == input.length()) {
        input.insert(insert_pos, to_insert);
    }
}
