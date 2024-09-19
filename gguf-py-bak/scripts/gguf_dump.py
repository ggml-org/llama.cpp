#!/usr/bin/env python3
from __future__ import annotations

import logging
import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf import GGUFReader, GGUFValueType, ReaderTensor  # noqa: E402

logger = logging.getLogger("gguf-dump")


def get_file_host_endian(reader: GGUFReader) -> tuple[str, str]:
    host_endian = 'LITTLE' if np.uint32(1) == np.uint32(1).newbyteorder("<") else 'BIG'
    if reader.byte_order == 'S':
        file_endian = 'BIG' if host_endian == 'LITTLE' else 'LITTLE'
    else:
        file_endian = host_endian
    return (host_endian, file_endian)


# For more information about what field.parts and field.data represent,
# please see the comments in the modify_gguf.py example.
def dump_metadata(reader: GGUFReader, args: argparse.Namespace) -> None:
    host_endian, file_endian = get_file_host_endian(reader)
    print(f'* File is {file_endian} endian, script is running on a {host_endian} endian host.')  # noqa: NP100
    print(f'* Dumping {len(reader.fields)} key/value pair(s)')  # noqa: NP100
    for n, field in enumerate(reader.fields.values(), 1):
        if not field.types:
            pretty_type = 'N/A'
        elif field.types[0] == GGUFValueType.ARRAY:
            nest_count = len(field.types) - 1
            pretty_type = '[' * nest_count + str(field.types[-1].name) + ']' * nest_count
        else:
            pretty_type = str(field.types[-1].name)

        log_message = f'  {n:5}: {pretty_type:10} | {len(field.data):8} | {field.name}'
        if len(field.types) == 1:
            curr_type = field.types[0]
            if curr_type == GGUFValueType.STRING:
                log_message += ' = {0}'.format(repr(str(bytes(field.parts[-1]), encoding='utf-8')[:60]))
            elif field.types[0] in reader.gguf_scalar_to_np:
                log_message += ' = {0}'.format(field.parts[-1][0])
        print(log_message)  # noqa: NP100
    if args.no_tensors:
        return
    print(f'* Dumping {len(reader.tensors)} tensor(s)')  # noqa: NP100
    for n, tensor in enumerate(reader.tensors, 1):
        prettydims = ', '.join('{0:5}'.format(d) for d in list(tensor.shape) + [1] * (4 - len(tensor.shape)))
        print(f'  {n:5}: {tensor.n_elements:10} | {prettydims} | {tensor.tensor_type.name:7} | {tensor.name}')  # noqa: NP100


def dump_metadata_json(reader: GGUFReader, args: argparse.Namespace) -> None:
    import json
    host_endian, file_endian = get_file_host_endian(reader)
    metadata: dict[str, Any] = {}
    tensors: dict[str, Any] = {}
    result = {
        "filename": args.model,
        "endian": file_endian,
        "metadata": metadata,
        "tensors": tensors,
    }
    for idx, field in enumerate(reader.fields.values()):
        curr: dict[str, Any] = {
            "index": idx,
            "type": field.types[0].name if field.types else 'UNKNOWN',
            "offset": field.offset,
        }
        metadata[field.name] = curr
        if field.types[:1] == [GGUFValueType.ARRAY]:
            curr["array_types"] = [t.name for t in field.types][1:]
            if not args.json_array:
                continue
            itype = field.types[-1]
            if itype == GGUFValueType.STRING:
                curr["value"] = [str(bytes(field.parts[idx]), encoding="utf-8") for idx in field.data]
            else:
                curr["value"] = [pv for idx in field.data for pv in field.parts[idx].tolist()]
        elif field.types[0] == GGUFValueType.STRING:
            curr["value"] = str(bytes(field.parts[-1]), encoding="utf-8")
        else:
            curr["value"] = field.parts[-1].tolist()[0]
    if not args.no_tensors:
        for idx, tensor in enumerate(reader.tensors):
            tensors[tensor.name] = {
                "index": idx,
                "shape": tensor.shape.tolist(),
                "type": tensor.tensor_type.name,
                "offset": tensor.field.offset,
            }
    json.dump(result, sys.stdout)


def markdown_table_with_alignment_support(header_map: list[dict[str, str]], data: list[dict[str, Any]]):
    # JSON to Markdown table formatting: https://stackoverflow.com/a/72983854/2850957

    # Alignment Utility Function
    def strAlign(padding: int, alignMode: str | None, strVal: str):
        if alignMode == 'center':
            return strVal.center(padding)
        elif alignMode == 'right':
            return strVal.rjust(padding - 1) + ' '
        elif alignMode == 'left':
            return ' ' + strVal.ljust(padding - 1)
        else: # default left
            return ' ' + strVal.ljust(padding - 1)

    def dashAlign(padding: int, alignMode: str | None):
        if alignMode == 'center':
            return ':' + '-' * (padding - 2) + ':'
        elif alignMode == 'right':
            return '-' * (padding - 1) + ':'
        elif alignMode == 'left':
            return ':' + '-' * (padding - 1)
        else: # default left
            return '-' * (padding)

    # Calculate Padding For Each Column Based On Header and Data Length
    rowsPadding = {}
    for index, columnEntry in enumerate(header_map):
        padCount = max([len(str(v)) for d in data for k, v in d.items() if k == columnEntry['key_name']], default=0) + 2
        headerPadCount = len(columnEntry['header_name']) + 2
        rowsPadding[index] = headerPadCount if padCount <= headerPadCount else padCount

    # Render Markdown Header
    rows = []
    rows.append('|'.join(strAlign(rowsPadding[index], columnEntry.get('align'), str(columnEntry['header_name'])) for index, columnEntry in enumerate(header_map)))
    rows.append('|'.join(dashAlign(rowsPadding[index], columnEntry.get('align')) for index, columnEntry in enumerate(header_map)))

    # Render Tabular Data
    for item in data:
        rows.append('|'.join(strAlign(rowsPadding[index], columnEntry.get('align'), str(item[columnEntry['key_name']])) for index, columnEntry in enumerate(header_map)))

    # Convert Tabular String Rows Into String
    tableString = ""
    for row in rows:
        tableString += f'|{row}|\n'

    return tableString


def element_count_rounded_notation(count: int) -> str:
    if count > 1e15 :
        # Quadrillion
        scaled_amount = count * 1e-15
        scale_suffix = "Q"
    elif count > 1e12 :
        # Trillions
        scaled_amount = count * 1e-12
        scale_suffix = "T"
    elif count > 1e9 :
        # Billions
        scaled_amount = count * 1e-9
        scale_suffix = "B"
    elif count > 1e6 :
        # Millions
        scaled_amount = count * 1e-6
        scale_suffix = "M"
    elif count > 1e3 :
        # Thousands
        scaled_amount = count * 1e-3
        scale_suffix = "K"
    else:
        # Under Thousands
        scaled_amount = count
        scale_suffix = ""
    return f"{'~' if count > 1e3 else ''}{round(scaled_amount)}{scale_suffix}"


def translate_tensor_name(name):
    words = name.split(".")

    # Source: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#standardized-tensor-names
    abbreviation_dictionary = {
        'token_embd': 'Token embedding',
        'pos_embd': 'Position embedding',
        'output_norm': 'Output normalization',
        'output': 'Output',
        'attn_norm': 'Attention normalization',
        'attn_norm_2': 'Attention normalization',
        'attn_qkv': 'Attention query-key-value',
        'attn_q': 'Attention query',
        'attn_k': 'Attention key',
        'attn_v': 'Attention value',
        'attn_output': 'Attention output',
        'ffn_norm': 'Feed-forward network normalization',
        'ffn_up': 'Feed-forward network "up"',
        'ffn_gate': 'Feed-forward network "gate"',
        'ffn_down': 'Feed-forward network "down"',
        'ffn_gate_inp': 'Expert-routing layer for the Feed-forward network in Mixture of Expert models',
        'ffn_gate_exp': 'Feed-forward network "gate" layer per expert in Mixture of Expert models',
        'ffn_down_exp': 'Feed-forward network "down" layer per expert in Mixture of Expert models',
        'ffn_up_exp': 'Feed-forward network "up" layer per expert in Mixture of Expert models',
        'ssm_in': 'State space model input projections',
        'ssm_conv1d': 'State space model rolling/shift',
        'ssm_x': 'State space model selective parametrization',
        'ssm_a': 'State space model state compression',
        'ssm_d': 'State space model skip connection',
        'ssm_dt': 'State space model time step',
        'ssm_out': 'State space model output projection',
        'blk': 'Block',
        'enc': 'Encoder',
        'dec': 'Decoder',
    }

    expanded_words = []
    for word in words:
        word_norm = word.strip().lower()
        if word_norm in abbreviation_dictionary:
            expanded_words.append(abbreviation_dictionary[word_norm].title())
        else:
            expanded_words.append(word.title())

    return ' '.join(expanded_words)


def dump_markdown_metadata(reader: GGUFReader, args: argparse.Namespace) -> None:
    host_endian, file_endian = get_file_host_endian(reader)
    markdown_content = ""
    markdown_content += f'# {args.model} - GGUF Internal File Dump\n\n'
    markdown_content += f'- Endian: {file_endian} endian\n'
    markdown_content += '\n'
    markdown_content += '## Key Value Metadata Store\n\n'
    markdown_content += f'There are {len(reader.fields)} key-value pairs in this file\n'
    markdown_content += '\n'

    kv_dump_table: list[dict[str, str | int]] = []
    for n, field in enumerate(reader.fields.values(), 1):
        if not field.types:
            pretty_type = 'N/A'
        elif field.types[0] == GGUFValueType.ARRAY:
            nest_count = len(field.types) - 1
            pretty_type = '[' * nest_count + str(field.types[-1].name) + ']' * nest_count
        else:
            pretty_type = str(field.types[-1].name)

        def escape_markdown_inline_code(value_string):
            # Find the longest contiguous sequence of backticks in the string then
            # wrap string with appropriate number of backticks required to escape it
            max_backticks = max((len(match.group(0)) for match in re.finditer(r'`+', value_string)), default=0)
            inline_code_marker = '`' * (max_backticks + 1)

            # If the string starts or ends with a backtick, add a space at the beginning and end
            if value_string.startswith('`') or value_string.endswith('`'):
                value_string = f" {value_string} "

            return f"{inline_code_marker}{value_string}{inline_code_marker}"

        total_elements = len(field.data)
        value = ""
        if len(field.types) == 1:
            curr_type = field.types[0]
            if curr_type == GGUFValueType.STRING:
                truncate_length = 60
                value_string = str(bytes(field.parts[-1]), encoding='utf-8')
                if len(value_string) > truncate_length:
                    head = escape_markdown_inline_code(value_string[:truncate_length // 2])
                    tail = escape_markdown_inline_code(value_string[-truncate_length // 2:])
                    value = "{head}...{tail}".format(head=head, tail=tail)
                else:
                    value = escape_markdown_inline_code(value_string)
            elif curr_type in reader.gguf_scalar_to_np:
                value = str(field.parts[-1][0])
        else:
            if field.types[0] == GGUFValueType.ARRAY:
                curr_type = field.types[1]
                array_elements = []

                if curr_type == GGUFValueType.STRING:
                    render_element = min(5, total_elements)
                    for element_pos in range(render_element):
                        truncate_length = 30
                        value_string = str(bytes(field.parts[-1 - (total_elements - element_pos - 1) * 2]), encoding='utf-8')
                        if len(value_string) > truncate_length:
                            head = escape_markdown_inline_code(value_string[:truncate_length // 2])
                            tail = escape_markdown_inline_code(value_string[-truncate_length // 2:])
                            value = "{head}...{tail}".format(head=head, tail=tail)
                        else:
                            value = escape_markdown_inline_code(value_string)
                        array_elements.append(value)

                elif curr_type in reader.gguf_scalar_to_np:
                    render_element = min(7, total_elements)
                    for element_pos in range(render_element):
                        array_elements.append(str(field.parts[-1 - (total_elements - element_pos - 1)][0]))

                value = f'[ {", ".join(array_elements).strip()}{", ..." if total_elements > len(array_elements) else ""} ]'

        kv_dump_table.append({"n":n, "pretty_type":pretty_type, "total_elements":total_elements, "field_name":field.name, "value":value})

    kv_dump_table_header_map = [
        {'key_name':'n',                'header_name':'POS',      'align':'right'},
        {'key_name':'pretty_type',      'header_name':'TYPE',     'align':'left'},
        {'key_name':'total_elements',   'header_name':'Count',    'align':'right'},
        {'key_name':'field_name',       'header_name':'Key',      'align':'left'},
        {'key_name':'value',            'header_name':'Value',    'align':'left'},
    ]

    markdown_content += markdown_table_with_alignment_support(kv_dump_table_header_map, kv_dump_table)

    markdown_content += "\n"

    if not args.no_tensors:
        # Group tensors by their prefix and maintain order
        tensor_prefix_order: list[str] = []
        tensor_name_to_key: dict[str, int] = {}
        tensor_groups: dict[str, list[ReaderTensor]] = {}
        total_elements = sum(tensor.n_elements for tensor in reader.tensors)

        # Parsing Tensors Record
        for key, tensor in enumerate(reader.tensors):
            tensor_components = tensor.name.split('.')

            # Classify Tensor Group
            tensor_group_name = "base"
            if tensor_components[0] == 'blk':
                tensor_group_name = f"{tensor_components[0]}.{tensor_components[1]}"
            elif tensor_components[0] in ['enc', 'dec'] and tensor_components[1] == 'blk':
                tensor_group_name = f"{tensor_components[0]}.{tensor_components[1]}.{tensor_components[2]}"
            elif tensor_components[0] in ['enc', 'dec']:
                tensor_group_name = f"{tensor_components[0]}"

            # Check if new Tensor Group
            if tensor_group_name not in tensor_groups:
                tensor_groups[tensor_group_name] = []
                tensor_prefix_order.append(tensor_group_name)

            # Record Tensor and Tensor Position
            tensor_groups[tensor_group_name].append(tensor)
            tensor_name_to_key[tensor.name] = key

        # Tensors Mapping Dump
        markdown_content += f'## Tensors Overview {element_count_rounded_notation(total_elements)} Elements\n\n'
        markdown_content += f'Total number of elements in all tensors: {total_elements} Elements\n'
        markdown_content += '\n'

        for group in tensor_prefix_order:
            tensors = tensor_groups[group]
            group_elements = sum(tensor.n_elements for tensor in tensors)
            markdown_content += f"- [{translate_tensor_name(group)} Tensor Group - {element_count_rounded_notation(group_elements)} Elements](#{group.replace('.', '_')})\n"

        markdown_content += "\n"

        markdown_content += "### Tensor Data Offset\n"
        markdown_content += '\n'
        markdown_content += 'This table contains the offset and data segment relative to start of file\n'
        markdown_content += '\n'

        tensor_mapping_table: list[dict[str, str | int]] = []
        for key, tensor in enumerate(reader.tensors):
            data_offset_pretty = '{0:#16x}'.format(tensor.data_offset)
            data_size_pretty = '{0:#16x}'.format(tensor.n_bytes)
            tensor_mapping_table.append({"t_id":key, "layer_name":tensor.name, "data_offset":data_offset_pretty, "data_size":data_size_pretty})

        tensors_mapping_table_header_map = [
            {'key_name':'t_id',         'header_name':'T_ID',               'align':'right'},
            {'key_name':'layer_name',   'header_name':'Tensor Layer Name',  'align':'left'},
            {'key_name':'data_offset',  'header_name':'Data Offset (B)',    'align':'right'},
            {'key_name':'data_size',    'header_name':'Data Size (B)',      'align':'right'},
        ]

        markdown_content += markdown_table_with_alignment_support(tensors_mapping_table_header_map, tensor_mapping_table)
        markdown_content += "\n"

        for group in tensor_prefix_order:
            tensors = tensor_groups[group]
            group_elements = sum(tensor.n_elements for tensor in tensors)
            group_percentage = group_elements / total_elements * 100
            markdown_content += f"### <a name=\"{group.replace('.', '_')}\">{translate_tensor_name(group)} Tensor Group : {element_count_rounded_notation(group_elements)} Elements</a>\n\n"

            # Precalculate column sizing for visual consistency
            prettify_element_est_count_size: int = 1
            prettify_element_count_size: int = 1
            prettify_dimension_max_widths: dict[int, int] = {}
            for tensor in tensors:
                prettify_element_est_count_size = max(prettify_element_est_count_size, len(str(element_count_rounded_notation(tensor.n_elements))))
                prettify_element_count_size = max(prettify_element_count_size, len(str(tensor.n_elements)))
                for i, dimension_size in enumerate(list(tensor.shape) + [1] * (4 - len(tensor.shape))):
                    prettify_dimension_max_widths[i] = max(prettify_dimension_max_widths.get(i,1), len(str(dimension_size)))

            # Generate Tensor Layer Table Content
            tensor_dump_table: list[dict[str, str | int]] = []
            for tensor in tensors:
                human_friendly_name = translate_tensor_name(tensor.name.replace(".weight", ".(W)").replace(".bias", ".(B)"))
                pretty_dimension = ' x '.join(f'{str(d):>{prettify_dimension_max_widths[i]}}' for i, d in enumerate(list(tensor.shape) + [1] * (4 - len(tensor.shape))))
                element_count_est = f"({element_count_rounded_notation(tensor.n_elements):>{prettify_element_est_count_size}})"
                element_count_string = f"{element_count_est} {tensor.n_elements:>{prettify_element_count_size}}"
                type_name_string = f"{tensor.tensor_type.name}"
                tensor_dump_table.append({"t_id":tensor_name_to_key[tensor.name], "layer_name":tensor.name, "human_layer_name":human_friendly_name, "element_count":element_count_string, "pretty_dimension":pretty_dimension, "tensor_type":type_name_string})

            tensor_dump_table_header_map = [
                {'key_name':'t_id',             'header_name':'T_ID',                             'align':'right'},
                {'key_name':'layer_name',       'header_name':'Tensor Layer Name',                'align':'left'},
                {'key_name':'human_layer_name', 'header_name':'Human Friendly Tensor Layer Name', 'align':'left'},
                {'key_name':'element_count',    'header_name':'Elements',                         'align':'left'},
                {'key_name':'pretty_dimension', 'header_name':'Shape',                            'align':'left'},
                {'key_name':'tensor_type',      'header_name':'Type',                             'align':'left'},
            ]

            markdown_content += markdown_table_with_alignment_support(tensor_dump_table_header_map, tensor_dump_table)

            markdown_content += "\n"
            markdown_content += f"- Total elements in {group}: ({element_count_rounded_notation(group_elements):>4}) {group_elements}\n"
            markdown_content += f"- Percentage of total elements: {group_percentage:.2f}%\n"
            markdown_content += "\n\n"

    print(markdown_content)  # noqa: NP100


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump GGUF file metadata")
    parser.add_argument("model",           type=str,            help="GGUF format model filename")
    parser.add_argument("--no-tensors", action="store_true", help="Don't dump tensor metadata")
    parser.add_argument("--json",       action="store_true", help="Produce JSON output")
    parser.add_argument("--json-array", action="store_true", help="Include full array values in JSON output (long)")
    parser.add_argument("--data-offset",    action="store_true", help="Start of data offset")
    parser.add_argument("--data-alignment", action="store_true", help="Data alignment applied globally to data field")
    parser.add_argument("--markdown",   action="store_true", help="Produce markdown output")
    parser.add_argument("--verbose",    action="store_true", help="increase output verbosity")

    args = parser.parse_args(None if len(sys.argv) > 1 else ["--help"])

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if not args.json and not args.markdown and not args.data_offset and not args.data_alignment:
        logger.info(f'* Loading: {args.model}')

    reader = GGUFReader(args.model, 'r')

    if args.json:
        dump_metadata_json(reader, args)
    elif args.markdown:
        dump_markdown_metadata(reader, args)
    elif args.data_offset:
        print(reader.data_offset)  # noqa: NP100
    elif args.data_alignment:
        print(reader.alignment)  # noqa: NP100
    else:
        dump_metadata(reader, args)


if __name__ == '__main__':
    main()
