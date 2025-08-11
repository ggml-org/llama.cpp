from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import os
import json


def fill_templated_filename(filename: str, output_type: str | None) -> str:
    # Given a file name fill in any type templates e.g. 'some-model-name.{ftype}.gguf'
    ftype_lowercase: str = output_type.lower() if output_type is not None else ""
    ftype_uppercase: str = output_type.upper() if output_type is not None else ""
    return filename.format(ftype_lowercase,
                           outtype=ftype_lowercase, ftype=ftype_lowercase,
                           OUTTYPE=ftype_uppercase, FTYPE=ftype_uppercase)


def model_weight_count_rounded_notation(model_params_count: int, min_digits: int = 2) -> str:
    if model_params_count > 1e12 :
        # Trillions Of Parameters
        scaled_model_params = model_params_count * 1e-12
        scale_suffix = "T"
    elif model_params_count > 1e9 :
        # Billions Of Parameters
        scaled_model_params = model_params_count * 1e-9
        scale_suffix = "B"
    elif model_params_count > 1e6 :
        # Millions Of Parameters
        scaled_model_params = model_params_count * 1e-6
        scale_suffix = "M"
    else:
        # Thousands Of Parameters
        scaled_model_params = model_params_count * 1e-3
        scale_suffix = "K"

    fix = max(min_digits - len(str(round(scaled_model_params)).lstrip('0')), 0)

    return f"{scaled_model_params:.{fix}f}{scale_suffix}"


def size_label(total_params: int, shared_params: int, expert_params: int, expert_count: int) -> str:

    if expert_count > 0:
        pretty_size = model_weight_count_rounded_notation(abs(shared_params) + abs(expert_params), min_digits=2)
        size_class = f"{expert_count}x{pretty_size}"
    else:
        size_class = model_weight_count_rounded_notation(abs(total_params), min_digits=2)

    return size_class


def naming_convention(model_name: str | None, base_name: str | None, finetune_string: str | None, version_string: str | None, size_label: str | None, output_type: str | None, model_type: Literal['vocab', 'LoRA'] | None = None) -> str:
    # Reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md#gguf-naming-convention

    if base_name is not None:
        name = base_name.strip().replace(' ', '-').replace('/', '-')
    elif model_name is not None:
        name = model_name.strip().replace(' ', '-').replace('/', '-')
    else:
        name = "ggml-model"

    parameters = f"-{size_label}" if size_label is not None else ""

    finetune = f"-{finetune_string.strip().replace(' ', '-')}" if finetune_string is not None else ""

    version = f"-{version_string.strip().replace(' ', '-')}" if version_string is not None else ""

    encoding = f"-{output_type.strip().replace(' ', '-').upper()}" if output_type is not None else ""

    kind = f"-{model_type.strip().replace(' ', '-')}" if model_type is not None else ""

    return f"{name}{parameters}{finetune}{version}{encoding}{kind}"


@dataclass
class RemoteTensor:
    dtype: str
    shape: tuple[int, ...]
    offset_start: int
    size: int
    url: str

    def data(self) -> bytearray:
        # TODO: handle request errors (maybe with limited retries?)
        # NOTE: using a bytearray, otherwise PyTorch complains the buffer is not writeable
        data = bytearray(SafetensorRemote.get_data_by_range(url=self.url, start=self.offset_start, size=self.size))
        return data


class SafetensorRemote:
    """
    Uility class to handle remote safetensor files.
    This class is designed to work with Hugging Face model repositories.

    Example (one model has single safetensor file, the other has multiple):
        for model_id in ["ngxson/TEST-Tiny-Llama4", "Qwen/Qwen2.5-7B-Instruct"]:
            tensors = SafetensorRemote.get_list_tensors_hf_model(model_id)
            print(tensors)

    Example reading tensor data:
        tensors = SafetensorRemote.get_list_tensors_hf_model(model_id)
        for name, meta in tensors.items():
            dtype, shape, offset_start, size, remote_safetensor_url = meta
            # read the tensor data
            data = SafetensorRemote.get_data_by_range(remote_safetensor_url, offset_start, size)
            print(data)
    """

    BASE_DOMAIN = "https://huggingface.co"
    ALIGNMENT = 8 # bytes

    @classmethod
    def get_list_tensors_hf_model(cls, model_id: str) -> dict[str, RemoteTensor]:
        """
        Get list of tensors from a Hugging Face model repository.

        Returns a dictionary of tensor names and their metadata.
        Each tensor is represented as a tuple of (dtype, shape, offset_start, size, remote_safetensor_url)
        """
        # case 1: model has only one single model.safetensor file
        is_single_file = cls.check_file_exist(f"{cls.BASE_DOMAIN}/{model_id}/resolve/main/model.safetensors")
        if is_single_file:
            url = f"{cls.BASE_DOMAIN}/{model_id}/resolve/main/model.safetensors"
            return cls.get_list_tensors(url)

        # case 2: model has multiple files
        index_url = f"{cls.BASE_DOMAIN}/{model_id}/resolve/main/model.safetensors.index.json"
        is_multiple_files = cls.check_file_exist(index_url)
        if is_multiple_files:
            # read the index file
            index_data = cls.get_data_by_range(index_url, 0)
            index_str = index_data.decode('utf-8')
            index_json = json.loads(index_str)
            assert index_json.get("weight_map") is not None, "weight_map not found in index file"
            weight_map = index_json["weight_map"]
            # get the list of files
            all_files = list(set(weight_map.values()))
            all_files.sort() # make sure we load shard files in order
            # get the list of tensors
            tensors: dict[str, RemoteTensor] = {}
            for file in all_files:
                url = f"{cls.BASE_DOMAIN}/{model_id}/resolve/main/{file}"
                for key, val in cls.get_list_tensors(url).items():
                    tensors[key] = val
            return tensors

        raise ValueError(
            f"No safetensor file has been found for model {model_id}."
            "If the repo has safetensor files, make sure the model is public or you have a "
            "valid Hugging Face token set in the environment variable HF_TOKEN."
        )

    @classmethod
    def get_list_tensors(cls, url: str) -> dict[str, RemoteTensor]:
        """
        Get list of tensors from a remote safetensor file.

        Returns a dictionary of tensor names and their metadata.
        Each tensor is represented as a tuple of (dtype, shape, offset_start, size)
        """
        metadata, data_start_offset = cls.get_metadata(url)
        res: dict[str, RemoteTensor] = {}

        for name, meta in metadata.items():
            if name == "__metadata__":
                continue
            if not isinstance(meta, dict):
                raise ValueError(f"Invalid metadata for tensor '{name}': {meta}")
            try:
                dtype = meta["dtype"]
                shape = meta["shape"]
                offset_start_relative, offset_end_relative = meta["data_offsets"]
                size = offset_end_relative - offset_start_relative
                offset_start = data_start_offset + offset_start_relative
                res[name] = RemoteTensor(dtype=dtype, shape=tuple(shape), offset_start=offset_start, size=size, url=url)
            except KeyError as e:
                raise ValueError(f"Missing key in metadata for tensor '{name}': {e}, meta = {meta}")

        return res

    @classmethod
    def get_metadata(cls, url: str) -> tuple[dict, int]:
        """
        Get JSON metadata from a remote safetensor file.

        Returns tuple of (metadata, data_start_offset)
        """
        # Request first 5MB of the file (hopefully enough for metadata)
        read_size = 5 * 1024 * 1024
        raw_data = cls.get_data_by_range(url, 0, read_size)

        # Parse header
        # First 8 bytes contain the metadata length as u64 little-endian
        if len(raw_data) < 8:
            raise ValueError("Not enough data to read metadata size")
        metadata_length = int.from_bytes(raw_data[:8], byteorder='little')

        # Calculate the data start offset
        data_start_offset = 8 + metadata_length
        alignment = SafetensorRemote.ALIGNMENT
        if data_start_offset % alignment != 0:
            data_start_offset += alignment - (data_start_offset % alignment)

        # Check if we have enough data to read the metadata
        if len(raw_data) < 8 + metadata_length:
            raise ValueError(f"Could not read complete metadata. Need {8 + metadata_length} bytes, got {len(raw_data)}")

        # Extract metadata bytes and parse as JSON
        metadata_bytes = raw_data[8:8 + metadata_length]
        metadata_str = metadata_bytes.decode('utf-8')
        try:
            metadata = json.loads(metadata_str)
            return metadata, data_start_offset
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse safetensor metadata as JSON: {e}")

    @classmethod
    def get_data_by_range(cls, url: str, start: int, size: int = -1) -> bytes:
        """
        Get raw byte data from a remote file by range.
        If size is not specified, it will read the entire file.
        """
        import requests
        from urllib.parse import urlparse

        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError(f"Invalid URL: {url}")

        headers = cls._get_request_headers()
        if size > -1:
            headers["Range"] = f"bytes={start}-{start + size}"
        response = requests.get(url, allow_redirects=True, headers=headers)
        response.raise_for_status()

        # Get raw byte data
        return response.content[slice(size if size > -1 else None)]

    @classmethod
    def check_file_exist(cls, url: str) -> bool:
        """
        Check if a file exists at the given URL.
        Returns True if the file exists, False otherwise.
        """
        import requests
        from urllib.parse import urlparse

        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError(f"Invalid URL: {url}")

        try:
            headers = cls._get_request_headers()
            headers["Range"] = "bytes=0-0"
            response = requests.head(url, allow_redirects=True, headers=headers)
            # Success (2xx) or redirect (3xx)
            return 200 <= response.status_code < 400
        except requests.RequestException:
            return False

    @classmethod
    def _get_request_headers(cls) -> dict[str, str]:
        """Prepare common headers for requests."""
        headers = {"User-Agent": "convert_hf_to_gguf"}
        if os.environ.get("HF_TOKEN"):
            headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
        return headers
