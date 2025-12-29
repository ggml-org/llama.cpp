#include <llamax.h>

#include <iostream>

const char * json_grammar = R"(
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= | " " | "\n" [ \t]{0,20}
)";

int main(int _argc, const char ** _argv) {
    if (_argc != 3) {
        std::cerr << "llamax_simple [model] \"What is up doctor?\"" << std::endl;
        return -1;
    }

    llamax::model   model = llamax::model::load_from_file(_argv[1], llamax::model_params::default_params());
    llamax::context ctx   = model.create_context(
        llamax::context_params::default_params().set_context_size(4096).set_batch_size(2048),
        llamax::sampler_builder().grammar(json_grammar, "root").min_p(0.05, 1).temp(0.8f).dist(llamax::default_seed()));
    llamax::iterator it = ctx.prompt(_argv[2]);

    while (std::optional<std::string> s = it.next()) {
        std::cout << s.value();
    }
    std::cout << std::endl;

    return 0;
}
