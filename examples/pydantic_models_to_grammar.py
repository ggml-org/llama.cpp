from __future__ import annotations

import inspect
import json
import re
from copy import copy
from enum import Enum
from inspect import getdoc, isclass
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union, get_args, get_origin, get_type_hints

from docstring_parser import parse
from pydantic import BaseModel, create_model

if TYPE_CHECKING:
    from types import GenericAlias
else:
    # python 3.8 compat
    from typing import _GenericAlias as GenericAlias

# TODO: fix this
# pyright: reportAttributeAccessIssue=information


class PydanticDataType(Enum):
    """
    Defines the data types supported by the grammar_generator.

    Attributes:
        STRING (str): Represents a string data type.
        BOOLEAN (str): Represents a boolean data type.
        INTEGER (str): Represents an integer data type.
        FLOAT (str): Represents a float data type.
        OBJECT (str): Represents an object data type.
        ARRAY (str): Represents an array data type.
        ENUM (str): Represents an enum data type.
        CUSTOM_CLASS (str): Represents a custom class data type.
    """

    STRING = "string"
    TRIPLE_QUOTED_STRING = "triple_quoted_string"
    MARKDOWN_CODE_BLOCK = "markdown_code_block"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    OBJECT = "object"
    ARRAY = "array"
    ENUM = "enum"
    ANY = "any"
    NULL = "null"
    CUSTOM_CLASS = "custom-class"
    CUSTOM_DICT = "custom-dict"
    SET = "set"


def map_pydantic_type_to_gbnf(pydantic_type: type[Any]) -> str:
    origin_type = get_origin(pydantic_type)
    origin_type = pydantic_type if origin_type is None else origin_type

    if isclass(origin_type) and issubclass(origin_type, str):
        return PydanticDataType.STRING.value
    elif isclass(origin_type) and issubclass(origin_type, bool):
        return PydanticDataType.BOOLEAN.value
    elif isclass(origin_type) and issubclass(origin_type, int):
        return PydanticDataType.INTEGER.value
    elif isclass(origin_type) and issubclass(origin_type, float):
        return PydanticDataType.FLOAT.value
    elif isclass(origin_type) and issubclass(origin_type, Enum):
        return PydanticDataType.ENUM.value

    elif isclass(origin_type) and issubclass(origin_type, BaseModel):
        return format_model_and_field_name(origin_type.__name__)
    elif origin_type is list:
        element_type = get_args(pydantic_type)[0]
        return f"{map_pydantic_type_to_gbnf(element_type)}-list"
    elif origin_type is set:
        element_type = get_args(pydantic_type)[0]
        return f"{map_pydantic_type_to_gbnf(element_type)}-set"
    elif origin_type is Union:
        union_types = get_args(pydantic_type)
        union_rules = [map_pydantic_type_to_gbnf(ut) for ut in union_types]
        return f"union-{'-or-'.join(union_rules)}"
    elif origin_type is Optional:
        element_type = get_args(pydantic_type)[0]
        return f"optional-{map_pydantic_type_to_gbnf(element_type)}"
    elif isclass(origin_type):
        return f"{PydanticDataType.CUSTOM_CLASS.value}-{format_model_and_field_name(origin_type.__name__)}"
    elif origin_type is dict:
        key_type, value_type = get_args(pydantic_type)
        return f"custom-dict-key-type-{format_model_and_field_name(map_pydantic_type_to_gbnf(key_type))}-value-type-{format_model_and_field_name(map_pydantic_type_to_gbnf(value_type))}"
    else:
        return "unknown"


def format_model_and_field_name(model_name: str) -> str:
    parts = re.findall("[A-Z][^A-Z]*", model_name)
    if not parts:  # Check if the list is empty
        return model_name.lower().replace("_", "-")
    return "-".join(part.lower().replace("_", "-") for part in parts)


def generate_list_rule(element_type):
    """
    Generate a GBNF rule for a list of a given element type.

    :param element_type: The type of the elements in the list (e.g., 'string').
    :return: A string representing the GBNF rule for a list of the given type.
    """
    rule_name = f"{map_pydantic_type_to_gbnf(element_type)}-list"
    element_rule = map_pydantic_type_to_gbnf(element_type)
    list_rule = rf'{rule_name} ::= "["  {element_rule} (","  {element_rule})* "]"'
    return list_rule


def get_members_structure(cls, rule_name):
    if issubclass(cls, Enum):
        # Handle Enum types
        members = [f'"\\"{member.value}\\""' for name, member in cls.__members__.items()]
        return f"{cls.__name__.lower()} ::= " + " | ".join(members)
    if cls.__annotations__ and cls.__annotations__ != {}:
        result = f'{rule_name} ::= "{{"'
        # Modify this comprehension
        members = [
            f'  "\\"{name}\\"" ":"  {map_pydantic_type_to_gbnf(param_type)}'
            for name, param_type in get_type_hints(cls).items()
            if name != "self"
        ]

        result += '"," '.join(members)
        result += '  "}"'
        return result
    if rule_name == "custom-class-any":
        result = f"{rule_name} ::= "
        result += "value"
        return result

    init_signature = inspect.signature(cls.__init__)
    parameters = init_signature.parameters
    result = f'{rule_name} ::=  "{{"'
    # Modify this comprehension too
    members = [
        f'  "\\"{name}\\"" ":"  {map_pydantic_type_to_gbnf(param.annotation)}'
        for name, param in parameters.items()
        if name != "self" and param.annotation != inspect.Parameter.empty
    ]

    result += '", "'.join(members)
    result += '  "}"'
    return result


def regex_to_gbnf(regex_pattern: str) -> str:
    """
    Translate a basic regex pattern to a GBNF rule.
    Note: This function handles only a subset of simple regex patterns.
    """
    gbnf_rule = regex_pattern

    # Translate common regex components to GBNF
    gbnf_rule = gbnf_rule.replace("\\d", "[0-9]")
    gbnf_rule = gbnf_rule.replace("\\s", "[ \t\n]")

    # Handle quantifiers and other regex syntax that is similar in GBNF
    # (e.g., '*', '+', '?', character classes)

    return gbnf_rule


def generate_gbnf_integer_rules(max_digit=None, min_digit=None):
    """

    Generate GBNF Integer Rules

    Generates GBNF (Generalized Backus-Naur Form) rules for integers based on the given maximum and minimum digits.

    Parameters:
        max_digit (int): The maximum number of digits for the integer. Default is None.
        min_digit (int): The minimum number of digits for the integer. Default is None.

    Returns:
        integer_rule (str): The identifier for the integer rule generated.
        additional_rules (list): A list of additional rules generated based on the given maximum and minimum digits.

    """
    additional_rules = []

    # Define the rule identifier based on max_digit and min_digit
    integer_rule = "integer-part"
    if max_digit is not None:
        integer_rule += f"-max{max_digit}"
    if min_digit is not None:
        integer_rule += f"-min{min_digit}"

    # Handling Integer Rules
    if max_digit is not None or min_digit is not None:
        # Start with an empty rule part
        integer_rule_part = ""

        # Add mandatory digits as per min_digit
        if min_digit is not None:
            integer_rule_part += "[0-9] " * min_digit

        # Add optional digits up to max_digit
        if max_digit is not None:
            optional_digits = max_digit - (min_digit if min_digit is not None else 0)
            integer_rule_part += "".join(["[0-9]? " for _ in range(optional_digits)])

        # Trim the rule part and append it to additional rules
        integer_rule_part = integer_rule_part.strip()
        if integer_rule_part:
            additional_rules.append(f"{integer_rule} ::= {integer_rule_part}")

    return integer_rule, additional_rules


def generate_gbnf_float_rules(max_digit=None, min_digit=None, max_precision=None, min_precision=None):
    """
    Generate GBNF float rules based on the given constraints.

    :param max_digit: Maximum number of digits in the integer part (default: None)
    :param min_digit: Minimum number of digits in the integer part (default: None)
    :param max_precision: Maximum number of digits in the fractional part (default: None)
    :param min_precision: Minimum number of digits in the fractional part (default: None)
    :return: A tuple containing the float rule and additional rules as a list

    Example Usage:
    max_digit = 3
    min_digit = 1
    max_precision = 2
    min_precision = 1
    generate_gbnf_float_rules(max_digit, min_digit, max_precision, min_precision)

    Output:
    ('float-3-1-2-1', ['integer-part-max3-min1 ::= [0-9] [0-9] [0-9]?', 'fractional-part-max2-min1 ::= [0-9] [0-9]?', 'float-3-1-2-1 ::= integer-part-max3-min1 "." fractional-part-max2-min
    *1'])

    Note:
    GBNF stands for Generalized Backus-Naur Form, which is a notation technique to specify the syntax of programming languages or other formal grammars.
    """
    additional_rules = []

    # Define the integer part rule
    integer_part_rule = (
        "integer-part"
        + (f"-max{max_digit}" if max_digit is not None else "")
        + (f"-min{min_digit}" if min_digit is not None else "")
    )

    # Define the fractional part rule based on precision constraints
    fractional_part_rule = "fractional-part"
    fractional_rule_part = ""
    if max_precision is not None or min_precision is not None:
        fractional_part_rule += (f"-max{max_precision}" if max_precision is not None else "") + (
            f"-min{min_precision}" if min_precision is not None else ""
        )
        # Minimum number of digits
        fractional_rule_part = "[0-9]" * (min_precision if min_precision is not None else 1)
        # Optional additional digits
        fractional_rule_part += "".join(
            [" [0-9]?"] * ((max_precision - (
                min_precision if min_precision is not None else 1)) if max_precision is not None else 0)
        )
        additional_rules.append(f"{fractional_part_rule} ::= {fractional_rule_part}")

    # Define the float rule
    float_rule = f"float-{max_digit if max_digit is not None else 'X'}-{min_digit if min_digit is not None else 'X'}-{max_precision if max_precision is not None else 'X'}-{min_precision if min_precision is not None else 'X'}"
    additional_rules.append(f'{float_rule} ::= {integer_part_rule} "." {fractional_part_rule}')

    # Generating the integer part rule definition, if necessary
    if max_digit is not None or min_digit is not None:
        integer_rule_part = "[0-9]"
        if min_digit is not None and min_digit > 1:
            integer_rule_part += " [0-9]" * (min_digit - 1)
        if max_digit is not None:
            integer_rule_part += "".join([" [0-9]?"] * (max_digit - (min_digit if min_digit is not None else 1)))
        additional_rules.append(f"{integer_part_rule} ::= {integer_rule_part.strip()}")

    return float_rule, additional_rules


def generate_gbnf_rule_for_type(
    model_name, field_name, field_type, is_optional, processed_models, created_rules, field_info=None
) -> tuple[str, list[str]]:
    """
    Generate GBNF rule for a given field type.

    :param model_name: Name of the model.

    :param field_name: Name of the field.
    :param field_type: Type of the field.
    :param is_optional: Whether the field is optional.
    :param processed_models: List of processed models.
    :param created_rules: List of created rules.
    :param field_info: Additional information about the field (optional).

    :return: Tuple containing the GBNF type and a list of additional rules.
    :rtype: tuple[str, list]
    """
    rules = []

    field_name = format_model_and_field_name(field_name)
    gbnf_type = map_pydantic_type_to_gbnf(field_type)

    origin_type = get_origin(field_type)
    origin_type = field_type if origin_type is None else origin_type

    if isclass(origin_type) and issubclass(origin_type, BaseModel):
        nested_model_name = format_model_and_field_name(field_type.__name__)
        nested_model_rules, _ = generate_gbnf_grammar(field_type, processed_models, created_rules)
        rules.extend(nested_model_rules)
        gbnf_type, rules = nested_model_name, rules
    elif isclass(origin_type) and issubclass(origin_type, Enum):
        enum_values = [f'"\\"{e.value}\\""' for e in field_type]  # Adding escaped quotes
        enum_rule = f"{model_name}-{field_name} ::= {' | '.join(enum_values)}"
        rules.append(enum_rule)
        gbnf_type, rules = model_name + "-" + field_name, rules
    elif origin_type is list:  # Array
        element_type = get_args(field_type)[0]
        element_rule_name, additional_rules = generate_gbnf_rule_for_type(
            model_name, f"{field_name}-element", element_type, is_optional, processed_models, created_rules
        )
        rules.extend(additional_rules)
        array_rule = f"""{model_name}-{field_name} ::= "[" ws {element_rule_name} ("," ws {element_rule_name})*  "]" """
        rules.append(array_rule)
        gbnf_type, rules = model_name + "-" + field_name, rules

    elif origin_type is set:  # Array
        element_type = get_args(field_type)[0]
        element_rule_name, additional_rules = generate_gbnf_rule_for_type(
            model_name, f"{field_name}-element", element_type, is_optional, processed_models, created_rules
        )
        rules.extend(additional_rules)
        array_rule = f"""{model_name}-{field_name} ::= "[" ws {element_rule_name} ("," ws {element_rule_name})*  "]" """
        rules.append(array_rule)
        gbnf_type, rules = model_name + "-" + field_name, rules

    elif gbnf_type.startswith("custom-class-"):
        rules.append(get_members_structure(field_type, gbnf_type))
    elif gbnf_type.startswith("custom-dict-"):
        key_type, value_type = get_args(field_type)

        additional_key_type, additional_key_rules = generate_gbnf_rule_for_type(
            model_name, f"{field_name}-key-type", key_type, is_optional, processed_models, created_rules
        )
        additional_value_type, additional_value_rules = generate_gbnf_rule_for_type(
            model_name, f"{field_name}-value-type", value_type, is_optional, processed_models, created_rules
        )
        gbnf_type = rf'{gbnf_type} ::= "{{"  ( {additional_key_type} ": "  {additional_value_type} ("," "\n" ws {additional_key_type} ":"  {additional_value_type})*  )? "}}" '

        rules.extend(additional_key_rules)
        rules.extend(additional_value_rules)
    elif gbnf_type.startswith("union-"):
        union_types = get_args(field_type)
        union_rules = []

        for union_type in union_types:
            if isinstance(union_type, GenericAlias):
                union_gbnf_type, union_rules_list = generate_gbnf_rule_for_type(
                    model_name, field_name, union_type, False, processed_models, created_rules
                )
                union_rules.append(union_gbnf_type)
                rules.extend(union_rules_list)

            elif not issubclass(union_type, type(None)):
                union_gbnf_type, union_rules_list = generate_gbnf_rule_for_type(
                    model_name, field_name, union_type, False, processed_models, created_rules
                )
                union_rules.append(union_gbnf_type)
                rules.extend(union_rules_list)

        # Defining the union grammar rule separately
        if len(union_rules) == 1:
            union_grammar_rule = f"{model_name}-{field_name}-optional ::= {' | '.join(union_rules)} | null"
        else:
            union_grammar_rule = f"{model_name}-{field_name}-union ::= {' | '.join(union_rules)}"
        rules.append(union_grammar_rule)
        if len(union_rules) == 1:
            gbnf_type = f"{model_name}-{field_name}-optional"
        else:
            gbnf_type = f"{model_name}-{field_name}-union"
    elif isclass(origin_type) and issubclass(origin_type, str):
        if field_info and hasattr(field_info, "json_schema_extra") and field_info.json_schema_extra is not None:
            triple_quoted_string = field_info.json_schema_extra.get("triple_quoted_string", False)
            markdown_string = field_info.json_schema_extra.get("markdown_code_block", False)

            gbnf_type = PydanticDataType.TRIPLE_QUOTED_STRING.value if triple_quoted_string else PydanticDataType.STRING.value
            gbnf_type = PydanticDataType.MARKDOWN_CODE_BLOCK.value if markdown_string else gbnf_type

        elif field_info and hasattr(field_info, "pattern"):
            # Convert regex pattern to grammar rule
            regex_pattern = field_info.regex.pattern
            gbnf_type = f"pattern-{field_name} ::= {regex_to_gbnf(regex_pattern)}"
        else:
            gbnf_type = PydanticDataType.STRING.value

    elif (
        isclass(origin_type)
        and issubclass(origin_type, float)
        and field_info
        and hasattr(field_info, "json_schema_extra")
        and field_info.json_schema_extra is not None
    ):
        # Retrieve precision attributes for floats
        max_precision = (
            field_info.json_schema_extra.get("max_precision") if field_info and hasattr(field_info,
                                                                                        "json_schema_extra") else None
        )
        min_precision = (
            field_info.json_schema_extra.get("min_precision") if field_info and hasattr(field_info,
                                                                                        "json_schema_extra") else None
        )
        max_digits = field_info.json_schema_extra.get("max_digit") if field_info and hasattr(field_info,
                                                                                             "json_schema_extra") else None
        min_digits = field_info.json_schema_extra.get("min_digit") if field_info and hasattr(field_info,
                                                                                             "json_schema_extra") else None

        # Generate GBNF rule for float with given attributes
        gbnf_type, rules = generate_gbnf_float_rules(
            max_digit=max_digits, min_digit=min_digits, max_precision=max_precision, min_precision=min_precision
        )

    elif (
        isclass(origin_type)
        and issubclass(origin_type, int)
        and field_info
        and hasattr(field_info, "json_schema_extra")
        and field_info.json_schema_extra is not None
    ):
        # Retrieve digit attributes for integers
        max_digits = field_info.json_schema_extra.get("max_digit") if field_info and hasattr(field_info,
                                                                                             "json_schema_extra") else None
        min_digits = field_info.json_schema_extra.get("min_digit") if field_info and hasattr(field_info,
                                                                                             "json_schema_extra") else None

        # Generate GBNF rule for integer with given attributes
        gbnf_type, rules = generate_gbnf_integer_rules(max_digit=max_digits, min_digit=min_digits)
    else:
        gbnf_type, rules = gbnf_type, []

    return gbnf_type, rules


def generate_gbnf_grammar(model: type[BaseModel], processed_models: set[type[BaseModel]], created_rules: dict[str, list[str]]) -> tuple[list[str], bool]:
    """

    Generate GBnF Grammar

    Generates a GBnF grammar for a given model.

    :param model: A Pydantic model class to generate the grammar for. Must be a subclass of BaseModel.
    :param processed_models: A set of already processed models to prevent infinite recursion.
    :param created_rules: A dict containing already created rules to prevent duplicates.
    :return: A list of GBnF grammar rules in string format. And two booleans indicating if an extra markdown or triple quoted string is in the grammar.
    Example Usage:
    ```
    model = MyModel
    processed_models = set()
    created_rules = dict()

    gbnf_grammar = generate_gbnf_grammar(model, processed_models, created_rules)
    ```
    """
    if model in processed_models:
        return [], False

    processed_models.add(model)
    model_name = format_model_and_field_name(model.__name__)

    if not issubclass(model, BaseModel):
        # For non-Pydantic classes, generate model_fields from __annotations__ or __init__
        if hasattr(model, "__annotations__") and model.__annotations__:
            model_fields = {name: (typ, ...) for name, typ in get_type_hints(model).items()}
        else:
            init_signature = inspect.signature(model.__init__)
            parameters = init_signature.parameters
            model_fields = {name: (param.annotation, param.default) for name, param in parameters.items() if
                            name != "self"}
    else:
        # For Pydantic models, use model_fields and check for ellipsis (required fields)
        model_fields = get_type_hints(model)

    model_rule_parts = []
    nested_rules = []
    has_markdown_code_block = False
    has_triple_quoted_string = False
    look_for_markdown_code_block = False
    look_for_triple_quoted_string = False
    for field_name, field_info in model_fields.items():
        if not issubclass(model, BaseModel):
            field_type, default_value = field_info
            # Check if the field is optional (not required)
            is_optional = (default_value is not inspect.Parameter.empty) and (default_value is not Ellipsis)
        else:
            field_type = field_info
            field_info = model.model_fields[field_name]
            is_optional = field_info.is_required is False and get_origin(field_type) is Optional
        rule_name, additional_rules = generate_gbnf_rule_for_type(
            model_name, format_model_and_field_name(field_name), field_type, is_optional, processed_models,
            created_rules, field_info
        )
        look_for_markdown_code_block = True if rule_name == "markdown_code_block" else False
        look_for_triple_quoted_string = True if rule_name == "triple_quoted_string" else False
        if not look_for_markdown_code_block and not look_for_triple_quoted_string:
            if rule_name not in created_rules:
                created_rules[rule_name] = additional_rules
            model_rule_parts.append(f' ws "\\"{field_name}\\"" ":" ws {rule_name}')  # Adding escaped quotes
            nested_rules.extend(additional_rules)
        else:
            has_triple_quoted_string = look_for_triple_quoted_string
            has_markdown_code_block = look_for_markdown_code_block

    fields_joined = r' "," "\n" '.join(model_rule_parts)
    model_rule = rf'{model_name} ::= "{{" "\n" {fields_joined} "\n" ws "}}"'

    has_special_string = False
    if has_triple_quoted_string:
        model_rule += '"\\n" ws "}"'
        model_rule += '"\\n" triple-quoted-string'
        has_special_string = True
    if has_markdown_code_block:
        model_rule += '"\\n" ws "}"'
        model_rule += '"\\n" markdown-code-block'
        has_special_string = True
    all_rules = [model_rule] + nested_rules

    return all_rules, has_special_string


def generate_gbnf_grammar_from_pydantic_models(
    models: list[type[BaseModel]], outer_object_name: str | None = None, outer_object_content: str | None = None,
    list_of_outputs: bool = False
) -> str:
    """
    Generate GBNF Grammar from Pydantic Models.

    This method takes a list of Pydantic models and uses them to generate a GBNF grammar string. The generated grammar string can be used for parsing and validating data using the generated
    * grammar.

    Args:
        models (list[type[BaseModel]]): A list of Pydantic models to generate the grammar from.
        outer_object_name (str): Outer object name for the GBNF grammar. If None, no outer object will be generated. Eg. "function" for function calling.
        outer_object_content (str): Content for the outer rule in the GBNF grammar. Eg. "function_parameters" or "params" for function calling.
        list_of_outputs (str, optional): Allows a list of output objects
    Returns:
        str: The generated GBNF grammar string.

    Examples:
        models = [UserModel, PostModel]
        grammar = generate_gbnf_grammar_from_pydantic(models)
        print(grammar)
        # Output:
        # root ::= UserModel | PostModel
        # ...
    """
    processed_models: set[type[BaseModel]] = set()
    all_rules = []
    created_rules: dict[str, list[str]] = {}
    if outer_object_name is None:
        for model in models:
            model_rules, _ = generate_gbnf_grammar(model, processed_models, created_rules)
            all_rules.extend(model_rules)

        if list_of_outputs:
            root_rule = r'root ::= (" "| "\n") "[" ws grammar-models ("," ws grammar-models)* ws "]"' + "\n"
        else:
            root_rule = r'root ::= (" "| "\n") grammar-models' + "\n"
        root_rule += "grammar-models ::= " + " | ".join(
            [format_model_and_field_name(model.__name__) for model in models])
        all_rules.insert(0, root_rule)
        return "\n".join(all_rules)
    elif outer_object_name is not None:
        if list_of_outputs:
            root_rule = (
                rf'root ::= (" "| "\n") "[" ws {format_model_and_field_name(outer_object_name)} ("," ws {format_model_and_field_name(outer_object_name)})* ws "]"'
                + "\n"
            )
        else:
            root_rule = f"root ::= {format_model_and_field_name(outer_object_name)}\n"

        model_rule = (
            rf'{format_model_and_field_name(outer_object_name)} ::= (" "| "\n") "{{" ws "\"{outer_object_name}\""  ":" ws grammar-models'
        )

        fields_joined = " | ".join(
            [rf"{format_model_and_field_name(model.__name__)}-grammar-model" for model in models])

        grammar_model_rules = f"\ngrammar-models ::= {fields_joined}"
        mod_rules = []
        for model in models:
            mod_rule = rf"{format_model_and_field_name(model.__name__)}-grammar-model ::= "
            mod_rule += (
                rf'"\"{model.__name__}\"" "," ws "\"{outer_object_content}\"" ":" ws {format_model_and_field_name(model.__name__)}' + "\n"
            )
            mod_rules.append(mod_rule)
        grammar_model_rules += "\n" + "\n".join(mod_rules)

        for model in models:
            model_rules, has_special_string = generate_gbnf_grammar(model, processed_models,
                                                                    created_rules)

            if not has_special_string:
                model_rules[0] += r'"\n" ws "}"'

            all_rules.extend(model_rules)

        all_rules.insert(0, root_rule + model_rule + grammar_model_rules)
        return "\n".join(all_rules)


def get_primitive_grammar(grammar):
    """
    Returns the needed GBNF primitive grammar for a given GBNF grammar string.

    Args:
        grammar (str): The string containing the GBNF grammar.

    Returns:
        str: GBNF primitive grammar string.
    """
    type_list: list[type[object]] = []
    if "string-list" in grammar:
        type_list.append(str)
    if "boolean-list" in grammar:
        type_list.append(bool)
    if "integer-list" in grammar:
        type_list.append(int)
    if "float-list" in grammar:
        type_list.append(float)
    additional_grammar = [generate_list_rule(t) for t in type_list]
    primitive_grammar = r"""
boolean ::= "true" | "false"
null ::= "null"
string ::= "\"" (
        [^"\\] |
        "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
      )* "\"" ws
ws ::= ([ \t\n] ws)?
float ::= ("-"? ([0] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

integer ::= [0-9]+"""

    any_block = ""
    if "custom-class-any" in grammar:
        any_block = """
value ::= object | array | string | number | boolean | null

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

number ::= integer | float"""

    markdown_code_block_grammar = ""
    if "markdown-code-block" in grammar:
        markdown_code_block_grammar = r'''
markdown-code-block ::= opening-triple-ticks markdown-code-block-content closing-triple-ticks
markdown-code-block-content ::= ( [^`] | "`" [^`] |  "`"  "`" [^`]  )*
opening-triple-ticks ::= "```" "python" "\n" | "```" "c" "\n" | "```" "cpp" "\n" | "```" "txt" "\n" | "```" "text" "\n" | "```" "json" "\n" | "```" "javascript" "\n" | "```" "css" "\n" | "```" "html" "\n" | "```" "markdown" "\n"
closing-triple-ticks ::= "```" "\n"'''

    if "triple-quoted-string" in grammar:
        markdown_code_block_grammar = r"""
triple-quoted-string ::= triple-quotes triple-quoted-string-content triple-quotes
triple-quoted-string-content ::= ( [^'] | "'" [^'] |  "'"  "'" [^']  )*
triple-quotes ::= "'''" """
    return "\n" + "\n".join(additional_grammar) + any_block + primitive_grammar + markdown_code_block_grammar


def generate_markdown_documentation(
    pydantic_models: list[type[BaseModel]], model_prefix="Model", fields_prefix="Fields",
    documentation_with_field_description=True
) -> str:
    """
    Generate markdown documentation for a list of Pydantic models.

    Args:
        pydantic_models (list[type[BaseModel]]): list of Pydantic model classes.
        model_prefix (str): Prefix for the model section.
        fields_prefix (str): Prefix for the fields section.
        documentation_with_field_description (bool): Include field descriptions in the documentation.

    Returns:
        str: Generated text documentation.
    """
    documentation = ""
    pyd_models: list[tuple[type[BaseModel], bool]] = [(model, True) for model in pydantic_models]
    for model, add_prefix in pyd_models:
        if add_prefix:
            documentation += f"{model_prefix}: {model.__name__}\n"
        else:
            documentation += f"Model: {model.__name__}\n"

        # Handling multi-line model description with proper indentation

        class_doc = getdoc(model)
        base_class_doc = getdoc(BaseModel)
        class_description = class_doc if class_doc and class_doc != base_class_doc else ""
        if class_description != "":
            documentation += "  Description: "
            documentation += format_multiline_description(class_description, 0) + "\n"

        if add_prefix:
            # Indenting the fields section
            documentation += f"  {fields_prefix}:\n"
        else:
            documentation += f"  Fields:\n"  # noqa: F541
        if isclass(model) and issubclass(model, BaseModel):
            for name, field_type in get_type_hints(model).items():
                # if name == "markdown_code_block":
                #    continue
                if get_origin(field_type) == list:
                    element_type = get_args(field_type)[0]
                    if isclass(element_type) and issubclass(element_type, BaseModel):
                        pyd_models.append((element_type, False))
                if get_origin(field_type) == Union:
                    element_types = get_args(field_type)
                    for element_type in element_types:
                        if isclass(element_type) and issubclass(element_type, BaseModel):
                            pyd_models.append((element_type, False))
                documentation += generate_field_markdown(
                    name, field_type, model, documentation_with_field_description=documentation_with_field_description
                )
            documentation += "\n"

        if hasattr(model, "Config") and hasattr(model.Config,
                                                "json_schema_extra") and "example" in model.Config.json_schema_extra:
            documentation += f"  Expected Example Output for {format_model_and_field_name(model.__name__)}:\n"
            json_example = json.dumps(model.Config.json_schema_extra["example"])
            documentation += format_multiline_description(json_example, 2) + "\n"

    return documentation


def generate_field_markdown(
    field_name: str, field_type: type[Any], model: type[BaseModel], depth=1,
    documentation_with_field_description=True
) -> str:
    """
    Generate markdown documentation for a Pydantic model field.

    Args:
        field_name (str): Name of the field.
        field_type (type[Any]): Type of the field.
        model (type[BaseModel]): Pydantic model class.
        depth (int): Indentation depth in the documentation.
        documentation_with_field_description (bool): Include field descriptions in the documentation.

    Returns:
        str: Generated text documentation for the field.
    """
    indent = "    " * depth

    field_info = model.model_fields.get(field_name)
    field_description = field_info.description if field_info and field_info.description else ""

    origin_type = get_origin(field_type)
    origin_type = field_type if origin_type is None else origin_type

    if origin_type == list:
        element_type = get_args(field_type)[0]
        field_text = f"{indent}{field_name} ({format_model_and_field_name(field_type.__name__)} of {format_model_and_field_name(element_type.__name__)})"
        if field_description != "":
            field_text += ":\n"
        else:
            field_text += "\n"
    elif origin_type == Union:
        element_types = get_args(field_type)
        types = []
        for element_type in element_types:
            types.append(format_model_and_field_name(element_type.__name__))
        field_text = f"{indent}{field_name} ({' or '.join(types)})"
        if field_description != "":
            field_text += ":\n"
        else:
            field_text += "\n"
    else:
        field_text = f"{indent}{field_name} ({format_model_and_field_name(field_type.__name__)})"
        if field_description != "":
            field_text += ":\n"
        else:
            field_text += "\n"

    if not documentation_with_field_description:
        return field_text

    if field_description != "":
        field_text += f"        Description: {field_description}\n"

    # Check for and include field-specific examples if available
    if hasattr(model, "Config") and hasattr(model.Config,
                                            "json_schema_extra") and "example" in model.Config.json_schema_extra:
        field_example = model.Config.json_schema_extra["example"].get(field_name)
        if field_example is not None:
            example_text = f"'{field_example}'" if isinstance(field_example, str) else field_example
            field_text += f"{indent}  Example: {example_text}\n"

    if isclass(origin_type) and issubclass(origin_type, BaseModel):
        field_text += f"{indent}  Details:\n"
        for name, type_ in get_type_hints(field_type).items():
            field_text += generate_field_markdown(name, type_, field_type, depth + 2)

    return field_text


def format_json_example(example: dict[str, Any], depth: int) -> str:
    """
    Format a JSON example into a readable string with indentation.

    Args:
        example (dict): JSON example to be formatted.
        depth (int): Indentation depth.

    Returns:
        str: Formatted JSON example string.
    """
    indent = "    " * depth
    formatted_example = "{\n"
    for key, value in example.items():
        value_text = f"'{value}'" if isinstance(value, str) else value
        formatted_example += f"{indent}{key}: {value_text},\n"
    formatted_example = formatted_example.rstrip(",\n") + "\n" + indent + "}"
    return formatted_example


def generate_text_documentation(
    pydantic_models: list[type[BaseModel]], model_prefix="Model", fields_prefix="Fields",
    documentation_with_field_description=True
) -> str:
    """
    Generate text documentation for a list of Pydantic models.

    Args:
        pydantic_models (list[type[BaseModel]]): List of Pydantic model classes.
        model_prefix (str): Prefix for the model section.
        fields_prefix (str): Prefix for the fields section.
        documentation_with_field_description (bool): Include field descriptions in the documentation.

    Returns:
        str: Generated text documentation.
    """
    documentation = ""
    pyd_models: list[tuple[type[BaseModel], bool]] = [(model, True) for model in pydantic_models]
    for model, add_prefix in pyd_models:
        if add_prefix:
            documentation += f"{model_prefix}: {model.__name__}\n"
        else:
            documentation += f"Model: {model.__name__}\n"

        # Handling multi-line model description with proper indentation

        class_doc = getdoc(model)
        base_class_doc = getdoc(BaseModel)
        class_description = class_doc if class_doc and class_doc != base_class_doc else ""
        if class_description != "":
            documentation += "  Description: "
            documentation += "\n" + format_multiline_description(class_description, 2) + "\n"

        if isclass(model) and issubclass(model, BaseModel):
            documentation_fields = ""
            for name, field_type in get_type_hints(model).items():
                # if name == "markdown_code_block":
                #    continue
                if get_origin(field_type) == list:
                    element_type = get_args(field_type)[0]
                    if isclass(element_type) and issubclass(element_type, BaseModel):
                        pyd_models.append((element_type, False))
                if get_origin(field_type) == Union:
                    element_types = get_args(field_type)
                    for element_type in element_types:
                        if isclass(element_type) and issubclass(element_type, BaseModel):
                            pyd_models.append((element_type, False))
                documentation_fields += generate_field_text(
                    name, field_type, model, documentation_with_field_description=documentation_with_field_description
                )
            if documentation_fields != "":
                if add_prefix:
                    documentation += f"  {fields_prefix}:\n{documentation_fields}"
                else:
                    documentation += f"  Fields:\n{documentation_fields}"
            documentation += "\n"

        if hasattr(model, "Config") and hasattr(model.Config,
                                                "json_schema_extra") and "example" in model.Config.json_schema_extra:
            documentation += f"  Expected Example Output for {format_model_and_field_name(model.__name__)}:\n"
            json_example = json.dumps(model.Config.json_schema_extra["example"])
            documentation += format_multiline_description(json_example, 2) + "\n"

    return documentation


def generate_field_text(
    field_name: str, field_type: type[Any], model: type[BaseModel], depth=1,
    documentation_with_field_description=True
) -> str:
    """
    Generate text documentation for a Pydantic model field.

    Args:
        field_name (str): Name of the field.
        field_type (type[Any]): Type of the field.
        model (type[BaseModel]): Pydantic model class.
        depth (int): Indentation depth in the documentation.
        documentation_with_field_description (bool): Include field descriptions in the documentation.

    Returns:
        str: Generated text documentation for the field.
    """
    indent = "    " * depth

    field_info = model.model_fields.get(field_name)
    field_description = field_info.description if field_info and field_info.description else ""

    if get_origin(field_type) == list:
        element_type = get_args(field_type)[0]
        field_text = f"{indent}{field_name} ({format_model_and_field_name(field_type.__name__)} of {format_model_and_field_name(element_type.__name__)})"
        if field_description != "":
            field_text += ":\n"
        else:
            field_text += "\n"
    elif get_origin(field_type) == Union:
        element_types = get_args(field_type)
        types = []
        for element_type in element_types:
            types.append(format_model_and_field_name(element_type.__name__))
        field_text = f"{indent}{field_name} ({' or '.join(types)})"
        if field_description != "":
            field_text += ":\n"
        else:
            field_text += "\n"
    else:
        field_text = f"{indent}{field_name} ({format_model_and_field_name(field_type.__name__)})"
        if field_description != "":
            field_text += ":\n"
        else:
            field_text += "\n"

    if not documentation_with_field_description:
        return field_text

    if field_description != "":
        field_text += f"{indent}  Description: " + field_description + "\n"

    # Check for and include field-specific examples if available
    if hasattr(model, "Config") and hasattr(model.Config,
                                            "json_schema_extra") and "example" in model.Config.json_schema_extra:
        field_example = model.Config.json_schema_extra["example"].get(field_name)
        if field_example is not None:
            example_text = f"'{field_example}'" if isinstance(field_example, str) else field_example
            field_text += f"{indent}  Example: {example_text}\n"

    if isclass(field_type) and issubclass(field_type, BaseModel):
        field_text += f"{indent}  Details:\n"
        for name, type_ in get_type_hints(field_type).items():
            field_text += generate_field_text(name, type_, field_type, depth + 2)

    return field_text


def format_multiline_description(description: str, indent_level: int) -> str:
    """
    Format a multiline description with proper indentation.

    Args:
        description (str): Multiline description.
        indent_level (int): Indentation level.

    Returns:
        str: Formatted multiline description.
    """
    indent = "    " * indent_level
    return indent + description.replace("\n", "\n" + indent)


def save_gbnf_grammar_and_documentation(
    grammar, documentation, grammar_file_path="./grammar.gbnf", documentation_file_path="./grammar_documentation.md"
):
    """
    Save GBNF grammar and documentation to specified files.

    Args:
        grammar (str): GBNF grammar string.
        documentation (str): Documentation string.
        grammar_file_path (str): File path to save the GBNF grammar.
        documentation_file_path (str): File path to save the documentation.

    Returns:
        None
    """
    try:
        with open(grammar_file_path, "w") as file:
            file.write(grammar + get_primitive_grammar(grammar))
        print(f"Grammar successfully saved to {grammar_file_path}")
    except IOError as e:
        print(f"An error occurred while saving the grammar file: {e}")

    try:
        with open(documentation_file_path, "w") as file:
            file.write(documentation)
        print(f"Documentation successfully saved to {documentation_file_path}")
    except IOError as e:
        print(f"An error occurred while saving the documentation file: {e}")


def remove_empty_lines(string):
    """
    Remove empty lines from a string.

    Args:
        string (str): Input string.

    Returns:
        str: String with empty lines removed.
    """
    lines = string.splitlines()
    non_empty_lines = [line for line in lines if line.strip() != ""]
    string_no_empty_lines = "\n".join(non_empty_lines)
    return string_no_empty_lines


def generate_and_save_gbnf_grammar_and_documentation(
    pydantic_model_list,
    grammar_file_path="./generated_grammar.gbnf",
    documentation_file_path="./generated_grammar_documentation.md",
    outer_object_name: str | None = None,
    outer_object_content: str | None = None,
    model_prefix: str = "Output Model",
    fields_prefix: str = "Output Fields",
    list_of_outputs: bool = False,
    documentation_with_field_description=True,
):
    """
    Generate GBNF grammar and documentation, and save them to specified files.

    Args:
        pydantic_model_list: List of Pydantic model classes.
        grammar_file_path (str): File path to save the generated GBNF grammar.
        documentation_file_path (str): File path to save the generated documentation.
        outer_object_name (str): Outer object name for the GBNF grammar. If None, no outer object will be generated. Eg. "function" for function calling.
        outer_object_content (str): Content for the outer rule in the GBNF grammar. Eg. "function_parameters" or "params" for function calling.
        model_prefix (str): Prefix for the model section in the documentation.
        fields_prefix (str): Prefix for the fields section in the documentation.
        list_of_outputs (bool): Whether the output is a list of items.
        documentation_with_field_description (bool): Include field descriptions in the documentation.

    Returns:
        None
    """
    documentation = generate_markdown_documentation(
        pydantic_model_list, model_prefix, fields_prefix,
        documentation_with_field_description=documentation_with_field_description
    )
    grammar = generate_gbnf_grammar_from_pydantic_models(pydantic_model_list, outer_object_name, outer_object_content,
                                                         list_of_outputs)
    grammar = remove_empty_lines(grammar)
    save_gbnf_grammar_and_documentation(grammar, documentation, grammar_file_path, documentation_file_path)


def generate_gbnf_grammar_and_documentation(
    pydantic_model_list,
    outer_object_name: str | None = None,
    outer_object_content: str | None = None,
    model_prefix: str = "Output Model",
    fields_prefix: str = "Output Fields",
    list_of_outputs: bool = False,
    documentation_with_field_description=True,
):
    """
    Generate GBNF grammar and documentation for a list of Pydantic models.

    Args:
        pydantic_model_list: List of Pydantic model classes.
        outer_object_name (str): Outer object name for the GBNF grammar. If None, no outer object will be generated. Eg. "function" for function calling.
        outer_object_content (str): Content for the outer rule in the GBNF grammar. Eg. "function_parameters" or "params" for function calling.
        model_prefix (str): Prefix for the model section in the documentation.
        fields_prefix (str): Prefix for the fields section in the documentation.
        list_of_outputs (bool): Whether the output is a list of items.
        documentation_with_field_description (bool): Include field descriptions in the documentation.

    Returns:
        tuple: GBNF grammar string, documentation string.
    """
    documentation = generate_markdown_documentation(
        copy(pydantic_model_list), model_prefix, fields_prefix,
        documentation_with_field_description=documentation_with_field_description
    )
    grammar = generate_gbnf_grammar_from_pydantic_models(pydantic_model_list, outer_object_name, outer_object_content,
                                                         list_of_outputs)
    grammar = remove_empty_lines(grammar + get_primitive_grammar(grammar))
    return grammar, documentation


def generate_gbnf_grammar_and_documentation_from_dictionaries(
    dictionaries: list[dict[str, Any]],
    outer_object_name: str | None = None,
    outer_object_content: str | None = None,
    model_prefix: str = "Output Model",
    fields_prefix: str = "Output Fields",
    list_of_outputs: bool = False,
    documentation_with_field_description=True,
):
    """
    Generate GBNF grammar and documentation from a list of dictionaries.

    Args:
        dictionaries (list[dict]): List of dictionaries representing Pydantic models.
        outer_object_name (str): Outer object name for the GBNF grammar. If None, no outer object will be generated. Eg. "function" for function calling.
        outer_object_content (str): Content for the outer rule in the GBNF grammar. Eg. "function_parameters" or "params" for function calling.
        model_prefix (str): Prefix for the model section in the documentation.
        fields_prefix (str): Prefix for the fields section in the documentation.
        list_of_outputs (bool): Whether the output is a list of items.
        documentation_with_field_description (bool): Include field descriptions in the documentation.

    Returns:
        tuple: GBNF grammar string, documentation string.
    """
    pydantic_model_list = create_dynamic_models_from_dictionaries(dictionaries)
    documentation = generate_markdown_documentation(
        copy(pydantic_model_list), model_prefix, fields_prefix,
        documentation_with_field_description=documentation_with_field_description
    )
    grammar = generate_gbnf_grammar_from_pydantic_models(pydantic_model_list, outer_object_name, outer_object_content,
                                                         list_of_outputs)
    grammar = remove_empty_lines(grammar + get_primitive_grammar(grammar))
    return grammar, documentation


def create_dynamic_model_from_function(func: Callable[..., Any]):
    """
    Creates a dynamic Pydantic model from a given function's type hints and adds the function as a 'run' method.

    Args:
        func (Callable): A function with type hints from which to create the model.

    Returns:
        A dynamic Pydantic model class with the provided function as a 'run' method.
    """

    # Get the signature of the function
    sig = inspect.signature(func)

    # Parse the docstring
    assert func.__doc__ is not None
    docstring = parse(func.__doc__)

    dynamic_fields = {}
    param_docs = []
    for param in sig.parameters.values():
        # Exclude 'self' parameter
        if param.name == "self":
            continue

        # Assert that the parameter has a type annotation
        if param.annotation == inspect.Parameter.empty:
            raise TypeError(f"Parameter '{param.name}' in function '{func.__name__}' lacks a type annotation")

        # Find the parameter's description in the docstring
        param_doc = next((d for d in docstring.params if d.arg_name == param.name), None)

        # Assert that the parameter has a description
        if not param_doc or not param_doc.description:
            raise ValueError(
                f"Parameter '{param.name}' in function '{func.__name__}' lacks a description in the docstring")

        # Add parameter details to the schema
        param_docs.append((param.name, param_doc))
        if param.default == inspect.Parameter.empty:
            default_value = ...
        else:
            default_value = param.default
        dynamic_fields[param.name] = (
            param.annotation if param.annotation != inspect.Parameter.empty else str, default_value)
    # Creating the dynamic model
    dynamic_model = create_model(f"{func.__name__}", **dynamic_fields)

    for name, param_doc in param_docs:
        dynamic_model.model_fields[name].description = param_doc.description

    dynamic_model.__doc__ = docstring.short_description

    def run_method_wrapper(self):
        func_args = {name: getattr(self, name) for name, _ in dynamic_fields.items()}
        return func(**func_args)

    # Adding the wrapped function as a 'run' method
    setattr(dynamic_model, "run", run_method_wrapper)
    return dynamic_model


def add_run_method_to_dynamic_model(model: type[BaseModel], func: Callable[..., Any]):
    """
    Add a 'run' method to a dynamic Pydantic model, using the provided function.

    Args:
        model (type[BaseModel]): Dynamic Pydantic model class.
        func (Callable): Function to be added as a 'run' method to the model.

    Returns:
        type[BaseModel]: Pydantic model class with the added 'run' method.
    """

    def run_method_wrapper(self):
        func_args = {name: getattr(self, name) for name in model.model_fields}
        return func(**func_args)

    # Adding the wrapped function as a 'run' method
    setattr(model, "run", run_method_wrapper)

    return model


def create_dynamic_models_from_dictionaries(dictionaries: list[dict[str, Any]]):
    """
    Create a list of dynamic Pydantic model classes from a list of dictionaries.

    Args:
        dictionaries (list[dict]): List of dictionaries representing model structures.

    Returns:
        list[type[BaseModel]]: List of generated dynamic Pydantic model classes.
    """
    dynamic_models = []
    for func in dictionaries:
        model_name = format_model_and_field_name(func.get("name", ""))
        dyn_model = convert_dictionary_to_pydantic_model(func, model_name)
        dynamic_models.append(dyn_model)
    return dynamic_models


def map_grammar_names_to_pydantic_model_class(pydantic_model_list):
    output = {}
    for model in pydantic_model_list:
        output[format_model_and_field_name(model.__name__)] = model

    return output


def json_schema_to_python_types(schema):
    type_map = {
        "any": Any,
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "array": list,
    }
    return type_map[schema]


def list_to_enum(enum_name, values):
    return Enum(enum_name, {value: value for value in values})


def convert_dictionary_to_pydantic_model(dictionary: dict[str, Any], model_name: str = "CustomModel") -> type[Any]:
    """
    Convert a dictionary to a Pydantic model class.

    Args:
        dictionary (dict): Dictionary representing the model structure.
        model_name (str): Name of the generated Pydantic model.

    Returns:
        type[BaseModel]: Generated Pydantic model class.
    """
    fields: dict[str, Any] = {}

    if "properties" in dictionary:
        for field_name, field_data in dictionary.get("properties", {}).items():
            if field_data == "object":
                submodel = convert_dictionary_to_pydantic_model(dictionary, f"{model_name}_{field_name}")
                fields[field_name] = (submodel, ...)
            else:
                field_type = field_data.get("type", "str")

                if field_data.get("enum", []):
                    fields[field_name] = (list_to_enum(field_name, field_data.get("enum", [])), ...)
                elif field_type == "array":
                    items = field_data.get("items", {})
                    if items != {}:
                        array = {"properties": items}
                        array_type = convert_dictionary_to_pydantic_model(array, f"{model_name}_{field_name}_items")
                        fields[field_name] = (List[array_type], ...)
                    else:
                        fields[field_name] = (list, ...)
                elif field_type == "object":
                    submodel = convert_dictionary_to_pydantic_model(field_data, f"{model_name}_{field_name}")
                    fields[field_name] = (submodel, ...)
                elif field_type == "required":
                    required = field_data.get("enum", [])
                    for key, field in fields.items():
                        if key not in required:
                            optional_type = fields[key][0]
                            fields[key] = (Optional[optional_type], ...)
                else:
                    field_type = json_schema_to_python_types(field_type)
                    fields[field_name] = (field_type, ...)
    if "function" in dictionary:
        for field_name, field_data in dictionary.get("function", {}).items():
            if field_name == "name":
                model_name = field_data
            elif field_name == "description":
                fields["__doc__"] = field_data
            elif field_name == "parameters":
                return convert_dictionary_to_pydantic_model(field_data, f"{model_name}")

    if "parameters" in dictionary:
        field_data = {"function": dictionary}
        return convert_dictionary_to_pydantic_model(field_data, f"{model_name}")
    if "required" in dictionary:
        required = dictionary.get("required", [])
        for key, field in fields.items():
            if key not in required:
                optional_type = fields[key][0]
                fields[key] = (Optional[optional_type], ...)
    custom_model = create_model(model_name, **fields)
    return custom_model
