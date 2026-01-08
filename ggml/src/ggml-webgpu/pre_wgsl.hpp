#ifndef PRE_WGSL_HPP
#define PRE_WGSL_HPP

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace pre_wgsl {

//==============================================================
// Options
//==============================================================
struct Options {
    std::string              include_path = ".";
    std::vector<std::string> macros;
};

//==============================================================
// Utility: trim
//==============================================================
static std::string trim(const std::string & s) {
    size_t a = 0;
    while (a < s.size() && std::isspace((unsigned char) s[a])) {
        a++;
    }
    size_t b = s.size();
    while (b > a && std::isspace((unsigned char) s[b - 1])) {
        b--;
    }
    return s.substr(a, b - a);
}

static std::string trim_value(std::istream & is) {
    std::string str;
    std::getline(is, str);
    return trim(str);
}

static bool isIdentChar(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

static std::string expandMacroValue(const std::string &                                  name,
                                    const std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    if (visiting.count(name)) {
        throw std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    if (it == macros.end()) {
        visiting.erase(name);
        return name;
    }

    const std::string & value = it->second;
    if (value.empty()) {
        visiting.erase(name);
        return "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    return expanded;
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(line.size());

    size_t i = 0;
    while (i < line.size()) {
        if (isIdentChar(line[i])) {
            size_t start = i;
            while (i < line.size() && isIdentChar(line[i])) {
                i++;
            }
            std::string token = line.substr(start, i - start);

            auto it = macros.find(token);
            if (it != macros.end()) {
                result += expandMacroValue(token, macros, visiting);
            } else {
                result += token;
            }
        } else {
            result += line[i];
            i++;
        }
    }

    return result;
}

static std::string expandMacrosRecursive(const std::string &                                  line,
                                         const std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    return expandMacrosRecursiveInternal(line, macros, visiting);
}

//==============================================================
// Tokenizer for expressions in #if/#elif
//==============================================================
class ExprLexer {
  public:
    enum Kind { END, IDENT, NUMBER, OP, LPAREN, RPAREN };

    struct Tok {
        Kind        kind;
        std::string text;
    };

    explicit ExprLexer(std::string_view sv) : src(sv), pos(0) {}

    Tok next() {
        skipWS();
        if (pos >= src.size()) {
            return { END, "" };
        }

        char c = src[pos];

        // number
        if (std::isdigit((unsigned char) c)) {
            size_t start = pos;
            while (pos < src.size() && std::isdigit((unsigned char) src[pos])) {
                pos++;
            }
            return { NUMBER, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        if (std::isalpha((unsigned char) c) || c == '_') {
            size_t start = pos;
            while (pos < src.size() && (std::isalnum((unsigned char) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            return { IDENT, std::string(src.substr(start, pos - start)) };
        }

        if (c == '(') {
            pos++;
            return { LPAREN, "(" };
        }
        if (c == ')') {
            pos++;
            return { RPAREN, ")" };
        }

        // multi-char operators
        static const char * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        for (auto op : two_ops) {
            if (src.substr(pos, 2) == op) {
                pos += 2;
                return { OP, std::string(op) };
            }
        }

        // single-char operators
        if (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            return { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        return { END, "" };
    }

  private:
    std::string_view src;
    size_t           pos;

    void skipWS() {
        while (pos < src.size() && std::isspace((unsigned char) src[pos])) {
            pos++;
        }
    }
};

//==============================================================
// Expression Parser (recursive descent)
//==============================================================
class ExprParser {
  public:
    ExprParser(std::string_view                                     expr,
               const std::unordered_map<std::string, std::string> & macros,
               std::unordered_set<std::string> &                    visiting) :
        lex(expr),
        macros(macros),
        visiting(visiting) {
        advance();
    }

    int parse() { return parseLogicalOr(); }

  private:
    ExprLexer                                            lex;
    ExprLexer::Tok                                       tok;
    const std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    void advance() { tok = lex.next(); }

    bool acceptOp(const std::string & s) {
        if (tok.kind == ExprLexer::OP && tok.text == s) {
            advance();
            return true;
        }
        return false;
    }

    bool acceptKind(ExprLexer::Kind k) {
        if (tok.kind == k) {
            advance();
            return true;
        }
        return false;
    }

    int parseLogicalOr() {
        int v = parseLogicalAnd();
        while (acceptOp("||")) {
            int rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        return v;
    }

    int parseLogicalAnd() {
        int v = parseEquality();
        while (acceptOp("&&")) {
            int rhs = parseEquality();
            v       = (v && rhs);
        }
        return v;
    }

    int parseEquality() {
        int v = parseRelational();
        for (;;) {
            if (acceptOp("==")) {
                int rhs = parseRelational();
                v       = (v == rhs);
            } else if (acceptOp("!=")) {
                int rhs = parseRelational();
                v       = (v != rhs);
            } else {
                break;
            }
        }
        return v;
    }

    int parseRelational() {
        int v = parseShift();
        for (;;) {
            if (acceptOp("<")) {
                int rhs = parseShift();
                v       = (v < rhs);
            } else if (acceptOp(">")) {
                int rhs = parseShift();
                v       = (v > rhs);
            } else if (acceptOp("<=")) {
                int rhs = parseShift();
                v       = (v <= rhs);
            } else if (acceptOp(">=")) {
                int rhs = parseShift();
                v       = (v >= rhs);
            } else {
                break;
            }
        }
        return v;
    }

    int parseShift() {
        int v = parseAdd();
        for (;;) {
            if (acceptOp("<<")) {
                int rhs = parseAdd();
                v       = (v << rhs);
            } else if (acceptOp(">>")) {
                int rhs = parseAdd();
                v       = (v >> rhs);
            } else {
                break;
            }
        }
        return v;
    }

    int parseAdd() {
        int v = parseMult();
        for (;;) {
            if (acceptOp("+")) {
                int rhs = parseMult();
                v       = (v + rhs);
            } else if (acceptOp("-")) {
                int rhs = parseMult();
                v       = (v - rhs);
            } else {
                break;
            }
        }
        return v;
    }

    int parseMult() {
        int v = parseUnary();
        for (;;) {
            if (acceptOp("*")) {
                int rhs = parseUnary();
                v       = (v * rhs);
            } else if (acceptOp("/")) {
                int rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } else if (acceptOp("%")) {
                int rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } else {
                break;
            }
        }
        return v;
    }

    int parseUnary() {
        if (acceptOp("!")) {
            return !parseUnary();
        }
        if (acceptOp("-")) {
            return -parseUnary();
        }
        if (acceptOp("+")) {
            return +parseUnary();
        }
        return parsePrimary();
    }

    int parsePrimary() {
        // '(' expr ')'
        if (acceptKind(ExprLexer::LPAREN)) {
            int v = parse();
            if (!acceptKind(ExprLexer::RPAREN)) {
                throw std::runtime_error("missing ')'");
            }
            return v;
        }

        // number
        if (tok.kind == ExprLexer::NUMBER) {
            int v = std::stoi(tok.text);
            advance();
            return v;
        }

        // defined(identifier)
        if (tok.kind == ExprLexer::IDENT && tok.text == "defined") {
            advance();
            if (acceptKind(ExprLexer::LPAREN)) {
                if (tok.kind != ExprLexer::IDENT) {
                    throw std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                if (!acceptKind(ExprLexer::RPAREN)) {
                    throw std::runtime_error("missing ) in defined()");
                }
                return macros.count(name) ? 1 : 0;
            } else {
                // defined NAME
                if (tok.kind != ExprLexer::IDENT) {
                    throw std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                return macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        if (tok.kind == ExprLexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            if (it == macros.end()) {
                return 0;
            }
            if (it->second.empty()) {
                return 1;
            }
            return evalMacroExpression(name, it->second);
        }

        // unexpected
        return 0;
    }

    int evalMacroExpression(const std::string & name, const std::string & value) {
        if (visiting.count(name)) {
            throw std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        ExprParser ep(value, macros, visiting);
        int        v = ep.parse();
        visiting.erase(name);
        return v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
class Preprocessor {
  public:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        if (opts_.include_path.empty()) {
            opts_.include_path = ".";
        }
        parseMacroDefinitions(opts_.macros);
    }

    std::string preprocess_file(const std::string & filename, const std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processFile(filename, macros, predefined, include_stack, DirectiveMode::All);
        return result;
    }

    std::string preprocess(const std::string & contents, const std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::All);
        return result;
    }

    std::string preprocess_includes_file(const std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processFile(filename, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        return result;
    }

    std::string preprocess_includes(const std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        return result;
    }

  private:
    Options                                      opts_;
    std::unordered_map<std::string, std::string> global_macros;

    enum class DirectiveMode { All, IncludesOnly };

    struct Cond {
        bool parent_active;
        bool active;
        bool taken;
    };

    //----------------------------------------------------------
    // Parse macro definitions into global_macros
    //----------------------------------------------------------
    void parseMacroDefinitions(const std::vector<std::string> & macro_defs) {
        for (const auto & def : macro_defs) {
            size_t eq_pos = def.find('=');
            if (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } else {
                // Format: NAME
                std::string name    = trim(def);
                global_macros[name] = "";
            }
        }
    }

    //----------------------------------------------------------
    // Build combined macro map and predefined set for a preprocessing operation
    //----------------------------------------------------------
    void buildMacros(const std::vector<std::string> &               additional_macros,
                     std::unordered_map<std::string, std::string> & macros,
                     std::unordered_set<std::string> &              predefined) {
        macros = global_macros;
        predefined.clear();

        for (const auto & [name, value] : global_macros) {
            predefined.insert(name);
        }

        for (const auto & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            if (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } else {
                name  = trim(def);
                value = "";
            }

            // Add to macros map (will override global if same name)
            macros[name] = value;
            predefined.insert(name);
        }
    }

    //----------------------------------------------------------
    // Helpers
    //----------------------------------------------------------
    std::string loadFile(const std::string & fname) {
        std::ifstream f(fname);
        if (!f.is_open()) {
            throw std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        return ss.str();
    }

    bool condActive(const std::vector<Cond> & cond) const {
        if (cond.empty()) {
            return true;
        }
        return cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(const std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            const std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_stack,
                            DirectiveMode                                  mode) {
        if (include_stack.count(name)) {
            throw std::runtime_error("Recursive include: " + name);
        }

        include_stack.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_stack, mode);
        include_stack.erase(name);
        return out;
    }

    std::string processIncludeFile(const std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   const std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_stack,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        return processFile(full_path, macros, predefined_macros, include_stack, mode);
    }

    //----------------------------------------------------------
    // Process text
    //----------------------------------------------------------
    std::string processString(const std::string &                            shader_code,
                              std::unordered_map<std::string, std::string> & macros,
                              const std::unordered_set<std::string> &        predefined_macros,
                              std::unordered_set<std::string> &              include_stack,
                              DirectiveMode                                  mode) {
        std::vector<Cond>  cond;  // Conditional stack for this shader
        std::stringstream  out;
        std::istringstream in(shader_code);
        std::string        line;

        while (std::getline(in, line)) {
            std::string t = trim(line);

            if (!t.empty() && t[0] == '#') {
                bool handled = handleDirective(t, out, macros, predefined_macros, cond, include_stack, mode);
                if (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << line << "\n";
                }
            } else {
                if (mode == DirectiveMode::IncludesOnly) {
                    out << line << "\n";
                } else if (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(line, macros);
                    out << expanded << "\n";
                }
            }
        }

        if (mode == DirectiveMode::All && !cond.empty()) {
            throw std::runtime_error("Unclosed #if directive");
        }

        return out.str();
    }

    //----------------------------------------------------------
    // Directive handler
    //----------------------------------------------------------
    bool handleDirective(const std::string &                            t,
                         std::stringstream &                            out,
                         std::unordered_map<std::string, std::string> & macros,
                         const std::unordered_set<std::string> &        predefined_macros,
                         std::vector<Cond> &                            cond,
                         std::unordered_set<std::string> &              include_stack,
                         DirectiveMode                                  mode) {
        // split into tokens
        std::string        body = t.substr(1);
        std::istringstream iss(body);
        std::string        cmd;
        iss >> cmd;

        if (cmd == "include") {
            if (mode == DirectiveMode::All && !condActive(cond)) {
                return true;
            }
            std::string file;
            iss >> file;
            if (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_stack, mode);
            return true;
        }

        if (mode == DirectiveMode::IncludesOnly) {
            return false;
        }

        if (cmd == "define") {
            if (!condActive(cond)) {
                return true;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            if (predefined_macros.count(name)) {
                return true;
            }
            std::string value = trim_value(iss);
            macros[name]      = value;
            return true;
        }

        if (cmd == "undef") {
            if (!condActive(cond)) {
                return true;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            if (predefined_macros.count(name)) {
                return true;
            }
            macros.erase(name);
            return true;
        }

        if (cmd == "ifdef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            return true;
        }

        if (cmd == "ifndef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            return true;
        }

        if (cmd == "if") {
            std::string expr = trim_value(iss);
            bool        p    = condActive(cond);
            bool        v    = false;
            if (p) {
                std::unordered_set<std::string> visiting;
                ExprParser                      ep(expr, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            return true;
        }

        if (cmd == "elif") {
            std::string expr = trim_value(iss);

            if (cond.empty()) {
                throw std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            if (!c.parent_active) {
                c.active = false;
                return true;
            }

            if (c.taken) {
                c.active = false;
                return true;
            }

            std::unordered_set<std::string> visiting;
            ExprParser                      ep(expr, macros, visiting);
            bool                            v = ep.parse() != 0;
            c.active                          = v;
            if (v) {
                c.taken = true;
            }
            return true;
        }

        if (cmd == "else") {
            if (cond.empty()) {
                throw std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            if (!c.parent_active) {
                c.active = false;
                return true;
            }
            if (c.taken) {
                c.active = false;
            } else {
                c.active = true;
                c.taken  = true;
            }
            return true;
        }

        if (cmd == "endif") {
            if (cond.empty()) {
                throw std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            return true;
        }

        // Unknown directive
        throw std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
