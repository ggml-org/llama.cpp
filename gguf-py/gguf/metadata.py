from __future__ import annotations

import re
import json
import yaml
import logging
from pathlib import Path
from typing import Any, Literal, Optional
from dataclasses import dataclass

from .constants import Keys

import gguf

logger = logging.getLogger("metadata")


@dataclass
class Metadata:
    # Authorship Metadata to be written to GGUF KV Store
    name: Optional[str] = None
    author: Optional[str] = None
    version: Optional[str] = None
    organization: Optional[str] = None
    finetune: Optional[str] = None
    basename: Optional[str] = None
    description: Optional[str] = None
    quantized_by: Optional[str] = None
    size_label: Optional[str] = None
    url: Optional[str] = None
    doi: Optional[str] = None
    uuid: Optional[str] = None
    repo_url: Optional[str] = None
    source_url: Optional[str] = None
    source_doi: Optional[str] = None
    source_uuid: Optional[str] = None
    source_repo_url: Optional[str] = None
    license: Optional[str] = None
    license_name: Optional[str] = None
    license_link: Optional[str] = None
    base_models: Optional[list[dict]] = None
    tags: Optional[list[str]] = None
    languages: Optional[list[str]] = None
    datasets: Optional[list[str]] = None

    @staticmethod
    def load(metadata_override_path: Optional[Path] = None, model_path: Optional[Path] = None, model_name: Optional[str] = None, total_params: int = 0) -> Metadata:
        # This grabs as many contextual authorship metadata as possible from the model repository
        # making any conversion as required to match the gguf kv store metadata format
        # as well as giving users the ability to override any authorship metadata that may be incorrect

        # Create a new Metadata instance
        metadata = Metadata()

        model_card = Metadata.load_model_card(model_path)
        hf_params = Metadata.load_hf_parameters(model_path)
        # TODO: load adapter_config.json when possible, it usually contains the base model of the LoRA adapter

        # heuristics
        metadata = Metadata.apply_metadata_heuristic(metadata, model_card, hf_params, model_path, total_params)

        # Metadata Override File Provided
        # This is based on LLM_KV_NAMES mapping in llama.cpp
        metadata_override = Metadata.load_metadata_override(metadata_override_path)

        metadata.name            = metadata_override.get(Keys.General.NAME,            metadata.name)
        metadata.author          = metadata_override.get(Keys.General.AUTHOR,          metadata.author)
        metadata.version         = metadata_override.get(Keys.General.VERSION,         metadata.version)
        metadata.organization    = metadata_override.get(Keys.General.ORGANIZATION,    metadata.organization)

        metadata.finetune        = metadata_override.get(Keys.General.FINETUNE,        metadata.finetune)
        metadata.basename        = metadata_override.get(Keys.General.BASENAME,        metadata.basename)

        metadata.description     = metadata_override.get(Keys.General.DESCRIPTION,     metadata.description)
        metadata.quantized_by    = metadata_override.get(Keys.General.QUANTIZED_BY,    metadata.quantized_by)

        metadata.size_label      = metadata_override.get(Keys.General.SIZE_LABEL,      metadata.size_label)
        metadata.license_name    = metadata_override.get(Keys.General.LICENSE_NAME,    metadata.license_name)
        metadata.license_link    = metadata_override.get(Keys.General.LICENSE_LINK,    metadata.license_link)

        metadata.url             = metadata_override.get(Keys.General.URL,             metadata.url)
        metadata.doi             = metadata_override.get(Keys.General.DOI,             metadata.doi)
        metadata.uuid            = metadata_override.get(Keys.General.UUID,            metadata.uuid)
        metadata.repo_url        = metadata_override.get(Keys.General.REPO_URL,        metadata.repo_url)

        metadata.source_url      = metadata_override.get(Keys.General.SOURCE_URL,      metadata.source_url)
        metadata.source_doi      = metadata_override.get(Keys.General.SOURCE_DOI,      metadata.source_doi)
        metadata.source_uuid     = metadata_override.get(Keys.General.SOURCE_UUID,     metadata.source_uuid)
        metadata.source_repo_url = metadata_override.get(Keys.General.SOURCE_REPO_URL, metadata.source_repo_url)

        # Base Models is received here as an array of models
        metadata.base_models     = metadata_override.get("general.base_models",        metadata.base_models)

        metadata.tags            = metadata_override.get(Keys.General.TAGS,            metadata.tags)
        metadata.languages       = metadata_override.get(Keys.General.LANGUAGES,       metadata.languages)
        metadata.datasets        = metadata_override.get(Keys.General.DATASETS,        metadata.datasets)

        # Direct Metadata Override (via direct cli argument)
        if model_name is not None:
            metadata.name = model_name

        return metadata

    @staticmethod
    def load_metadata_override(metadata_override_path: Optional[Path] = None) -> dict[str, Any]:
        if metadata_override_path is None or not metadata_override_path.is_file():
            return {}

        with open(metadata_override_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def load_model_card(model_path: Optional[Path] = None) -> dict[str, Any]:
        if model_path is None or not model_path.is_dir():
            return {}

        model_card_path = model_path / "README.md"

        if not model_card_path.is_file():
            return {}

        # The model card metadata is assumed to always be in YAML
        # ref: https://github.com/huggingface/transformers/blob/a5c642fe7a1f25d3bdcd76991443ba6ff7ee34b2/src/transformers/modelcard.py#L468-L473
        with open(model_card_path, "r", encoding="utf-8") as f:
            if f.readline() == "---\n":
                raw = f.read().partition("---\n")[0]
                data = yaml.safe_load(raw)
                if isinstance(data, dict):
                    return data
                else:
                    logger.error(f"while reading YAML model card frontmatter, data is {type(data)} instead of dict")
                    return {}
            else:
                return {}

    @staticmethod
    def load_hf_parameters(model_path: Optional[Path] = None) -> dict[str, Any]:
        if model_path is None or not model_path.is_dir():
            return {}

        config_path = model_path / "config.json"

        if not config_path.is_file():
            return {}

        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def id_to_title(string):
        # Convert capitalization into title form unless acronym or version number
        return ' '.join([w.title() if w.islower() and not re.match(r'^(v\d+(?:\.\d+)*|\d.*)$', w) else w for w in string.strip().replace('-', ' ').split()])

    @staticmethod
    def get_model_id_components(model_id: Optional[str] = None, total_params: int = 0) -> tuple[str | None, str | None, str | None, str | None, str | None, str | None]:
        # Huggingface often store model id as '<org>/<model name>'
        # so let's parse it and apply some heuristics if possible for model name components

        if model_id is None:
            # model ID missing
            return None, None, None, None, None, None

        if ' ' in model_id:
            # model ID is actually a normal human sentence
            # which means its most likely a normal model name only
            # not part of the hugging face naming standard, but whatever
            return model_id, None, None, None, None, None

        if '/' in model_id:
            # model ID (huggingface style)
            org_component, model_full_name_component = model_id.split('/', 1)
        else:
            # model ID but missing org components
            org_component, model_full_name_component = None, model_id

        # Check if we erroneously matched against './' or '../' etc...
        if org_component is not None and org_component[0] == '.':
            org_component = None

        name_parts: list[str] = model_full_name_component.split('-')

        # Remove empty parts
        for i in reversed(range(len(name_parts))):
            if len(name_parts[i]) == 0:
                del name_parts[i]

        name_types: list[
            set[Literal["basename", "size_label", "finetune", "version", "type"]]
        ] = [set() for _ in name_parts]

        # Annotate the name
        for i, part in enumerate(name_parts):
            # Version
            if re.fullmatch(r'(v|iter)?\d+([.]\d+)*', part, re.IGNORECASE):
                name_types[i].add("version")
            # Quant type (should not be there for base models, but still annotated)
            elif re.fullmatch(r'i?q\d(_\w)*|b?fp?(16|32)', part, re.IGNORECASE):
                name_types[i].add("type")
                name_parts[i] = part.upper()
            # Model size
            elif i > 0 and re.fullmatch(r'(([A]|\d+[x])?\d+([._]\d+)?[KMBT][\d]?|small|mini|medium|large|x?xl)', part, re.IGNORECASE):
                part = part.replace("_", ".")
                # Handle weird bloom-7b1 notation
                if part[-1].isdecimal():
                    part = part[:-2] + "." + part[-1] + part[-2]
                # Normalize the size suffixes
                if len(part) > 1 and part[-2].isdecimal():
                    if part[-1] in "kmbt":
                        part = part[:-1] + part[-1].upper()
                if total_params != 0:
                    try:
                        label_params = float(part[:-1]) * pow(1000, " KMBT".find(part[-1]))
                        # Only use it as a size label if it's close or bigger than the model size
                        # Note that LoRA adapters don't necessarily include all layers,
                        # so this is why bigger label sizes are accepted.
                        # Do not use the size label when it's smaller than 1/8 of the model size
                        if (total_params < 0 and label_params < abs(total_params) // 8) or (
                            # Check both directions when the current model isn't a LoRA adapter
                            total_params > 0 and abs(label_params - total_params) > 7 * total_params // 8
                        ):
                            # Likely a context length
                            name_types[i].add("finetune")
                            # Lowercase the size when it's a context length
                            part = part[:-1] + part[-1].lower()
                    except ValueError:
                        # Failed to convert the size label to float, use it anyway
                        pass
                if len(name_types[i]) == 0:
                    name_types[i].add("size_label")
                name_parts[i] = part
            # Some easy to recognize finetune names
            elif i > 0 and re.fullmatch(r'chat|instruct|vision|lora', part, re.IGNORECASE):
                if total_params < 0 and part.lower() == "lora":
                    # ignore redundant "lora" in the finetune part when the output is a lora adapter
                    name_types[i].add("type")
                else:
                    name_types[i].add("finetune")

        # Ignore word-based size labels when there is at least a number-based one present
        # TODO: should word-based size labels always be removed instead?
        if any(c.isdecimal() for n, t in zip(name_parts, name_types) if "size_label" in t for c in n):
            for n, t in zip(name_parts, name_types):
                if "size_label" in t:
                    if all(c.isalpha() for c in n):
                        t.remove("size_label")

        at_start = True
        # Find the basename through the annotated name
        for part, t in zip(name_parts, name_types):
            if at_start and ((len(t) == 0 and part[0].isalpha()) or "version" in t):
                t.add("basename")
            else:
                if at_start:
                    at_start = False
                if len(t) == 0:
                    t.add("finetune")

        # Remove the basename annotation from trailing version
        for part, t in zip(reversed(name_parts), reversed(name_types)):
            if "basename" in t and len(t) > 1:
                t.remove("basename")
            else:
                break

        basename = "-".join(n for n, t in zip(name_parts, name_types) if "basename" in t) or None
        # Deduplicate size labels using order-preserving 'dict' ('set' seems to sort the keys)
        size_label = "-".join(dict.fromkeys(s for s, t in zip(name_parts, name_types) if "size_label" in t).keys()) or None
        finetune = "-".join(f for f, t in zip(name_parts, name_types) if "finetune" in t) or None
        # TODO: should the basename version always be excluded?
        # NOTE: multiple finetune versions are joined together
        version = "-".join(v for v, t, in zip(name_parts, name_types) if "version" in t and "basename" not in t) or None

        if size_label is None and finetune is None and version is None:
            # Too ambiguous, output nothing
            basename = None

        return model_full_name_component, org_component, basename, finetune, version, size_label

    @staticmethod
    def apply_metadata_heuristic(metadata: Metadata, model_card: Optional[dict] = None, hf_params: Optional[dict] = None, model_path: Optional[Path] = None, total_params: int = 0) -> Metadata:
        # Reference Model Card Metadata: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1

        # Model Card Heuristics
        ########################
        if model_card is not None:

            if "model_name" in model_card and metadata.name is None:
                # Not part of huggingface model card standard but notice some model creator using it
                # such as TheBloke in 'TheBloke/Mistral-7B-Instruct-v0.2-GGUF'
                metadata.name = model_card.get("model_name")

            if "model_creator" in model_card and metadata.author is None:
                # Not part of huggingface model card standard but notice some model creator using it
                # such as TheBloke in 'TheBloke/Mistral-7B-Instruct-v0.2-GGUF'
                metadata.author = model_card.get("model_creator")

            if "model_type" in model_card and metadata.basename is None:
                # Not part of huggingface model card standard but notice some model creator using it
                # such as TheBloke in 'TheBloke/Mistral-7B-Instruct-v0.2-GGUF'
                metadata.basename = model_card.get("model_type")

            if "base_model" in model_card:
                # This represents the parent models that this is based on
                # Example: stabilityai/stable-diffusion-xl-base-1.0. Can also be a list (for merges)
                # Example of merges: https://huggingface.co/EmbeddedLLM/Mistral-7B-Merge-14-v0.1/blob/main/README.md
                metadata_base_models = []
                base_model_value = model_card.get("base_model", None)

                if base_model_value is not None:
                    if isinstance(base_model_value, str):
                        metadata_base_models.append(base_model_value)
                    elif isinstance(base_model_value, list):
                        metadata_base_models.extend(base_model_value)

                if metadata.base_models is None:
                    metadata.base_models = []

                for model_id in metadata_base_models:
                    # NOTE: model size of base model is assumed to be similar to the size of the current model
                    model_full_name_component, org_component, basename, finetune, version, size_label = Metadata.get_model_id_components(model_id, total_params)
                    base_model = {}
                    if model_full_name_component is not None:
                        base_model["name"] = Metadata.id_to_title(model_full_name_component)
                    if org_component is not None:
                        base_model["organization"] = Metadata.id_to_title(org_component)
                    if version is not None:
                        base_model["version"] = version
                    if org_component is not None and model_full_name_component is not None:
                        base_model["repo_url"] = f"https://huggingface.co/{org_component}/{model_full_name_component}"
                    metadata.base_models.append(base_model)

            if "license" in model_card and metadata.license is None:
                metadata.license = model_card.get("license")

            if "license_name" in model_card and metadata.license_name is None:
                metadata.license_name = model_card.get("license_name")

            if "license_link" in model_card and metadata.license_link is None:
                metadata.license_link = model_card.get("license_link")

            tags_value = model_card.get("tags", None)
            if tags_value is not None:

                if metadata.tags is None:
                    metadata.tags = []

                if isinstance(tags_value, str):
                    metadata.tags.append(tags_value)
                elif isinstance(tags_value, list):
                    metadata.tags.extend(tags_value)

            pipeline_tags_value = model_card.get("pipeline_tag", None)
            if pipeline_tags_value is not None:

                if metadata.tags is None:
                    metadata.tags = []

                if isinstance(pipeline_tags_value, str):
                    metadata.tags.append(pipeline_tags_value)
                elif isinstance(pipeline_tags_value, list):
                    metadata.tags.extend(pipeline_tags_value)

            language_value = model_card.get("languages", model_card.get("language", None))
            if language_value is not None:

                if metadata.languages is None:
                    metadata.languages = []

                if isinstance(language_value, str):
                    metadata.languages.append(language_value)
                elif isinstance(language_value, list):
                    metadata.languages.extend(language_value)

            dataset_value = model_card.get("datasets", model_card.get("dataset", None))
            if dataset_value is not None:

                if metadata.datasets is None:
                    metadata.datasets = []

                if isinstance(dataset_value, str):
                    metadata.datasets.append(dataset_value)
                elif isinstance(dataset_value, list):
                    metadata.datasets.extend(dataset_value)

        # Hugging Face Parameter Heuristics
        ####################################

        if hf_params is not None:

            hf_name_or_path = hf_params.get("_name_or_path")
            if hf_name_or_path is not None and hf_name_or_path.count('/') <= 1:
                # Use _name_or_path only if its actually a model name and not some computer path
                # e.g. 'meta-llama/Llama-2-7b-hf'
                model_id = hf_name_or_path
                model_full_name_component, org_component, basename, finetune, version, size_label = Metadata.get_model_id_components(model_id, total_params)
                if metadata.name is None and model_full_name_component is not None:
                    metadata.name = Metadata.id_to_title(model_full_name_component)
                if metadata.organization is None and org_component is not None:
                    metadata.organization = Metadata.id_to_title(org_component)
                if metadata.basename is None and basename is not None:
                    metadata.basename = basename
                if metadata.finetune is None and finetune is not None:
                    metadata.finetune = finetune
                if metadata.version is None and version is not None:
                    metadata.version = version
                if metadata.size_label is None and size_label is not None:
                    metadata.size_label = size_label

        # Directory Folder Name Fallback Heuristics
        ############################################
        if model_path is not None:
            model_id = model_path.name
            model_full_name_component, org_component, basename, finetune, version, size_label = Metadata.get_model_id_components(model_id, total_params)
            if metadata.name is None and model_full_name_component is not None:
                metadata.name = Metadata.id_to_title(model_full_name_component)
            if metadata.organization is None and org_component is not None:
                metadata.organization = Metadata.id_to_title(org_component)
            if metadata.basename is None and basename is not None:
                metadata.basename = basename
            if metadata.finetune is None and finetune is not None:
                metadata.finetune = finetune
            if metadata.version is None and version is not None:
                metadata.version = version
            if metadata.size_label is None and size_label is not None:
                metadata.size_label = size_label

        return metadata

    def set_gguf_meta_model(self, gguf_writer: gguf.GGUFWriter):
        assert self.name is not None
        gguf_writer.add_name(self.name)

        if self.author is not None:
            gguf_writer.add_author(self.author)
        if self.version is not None:
            gguf_writer.add_version(self.version)
        if self.organization is not None:
            gguf_writer.add_organization(self.organization)

        if self.finetune is not None:
            gguf_writer.add_finetune(self.finetune)
        if self.basename is not None:
            gguf_writer.add_basename(self.basename)

        if self.description is not None:
            gguf_writer.add_description(self.description)
        if self.quantized_by is not None:
            gguf_writer.add_quantized_by(self.quantized_by)

        if self.size_label is not None:
            gguf_writer.add_size_label(self.size_label)

        if self.license is not None:
            gguf_writer.add_license(self.license)
        if self.license_name is not None:
            gguf_writer.add_license_name(self.license_name)
        if self.license_link is not None:
            gguf_writer.add_license_link(self.license_link)

        if self.url is not None:
            gguf_writer.add_url(self.url)
        if self.doi is not None:
            gguf_writer.add_doi(self.doi)
        if self.uuid is not None:
            gguf_writer.add_uuid(self.uuid)
        if self.repo_url is not None:
            gguf_writer.add_repo_url(self.repo_url)

        if self.source_url is not None:
            gguf_writer.add_source_url(self.source_url)
        if self.source_doi is not None:
            gguf_writer.add_source_doi(self.source_doi)
        if self.source_uuid is not None:
            gguf_writer.add_source_uuid(self.source_uuid)
        if self.source_repo_url is not None:
            gguf_writer.add_source_repo_url(self.source_repo_url)

        if self.base_models is not None:
            gguf_writer.add_base_model_count(len(self.base_models))
            for key, base_model_entry in enumerate(self.base_models):
                if "name" in base_model_entry:
                    gguf_writer.add_base_model_name(key, base_model_entry["name"])
                if "author" in base_model_entry:
                    gguf_writer.add_base_model_author(key, base_model_entry["author"])
                if "version" in base_model_entry:
                    gguf_writer.add_base_model_version(key, base_model_entry["version"])
                if "organization" in base_model_entry:
                    gguf_writer.add_base_model_organization(key, base_model_entry["organization"])
                if "url" in base_model_entry:
                    gguf_writer.add_base_model_url(key, base_model_entry["url"])
                if "doi" in base_model_entry:
                    gguf_writer.add_base_model_doi(key, base_model_entry["doi"])
                if "uuid" in base_model_entry:
                    gguf_writer.add_base_model_uuid(key, base_model_entry["uuid"])
                if "repo_url" in base_model_entry:
                    gguf_writer.add_base_model_repo_url(key, base_model_entry["repo_url"])

        if self.tags is not None:
            gguf_writer.add_tags(self.tags)
        if self.languages is not None:
            gguf_writer.add_languages(self.languages)
        if self.datasets is not None:
            gguf_writer.add_datasets(self.datasets)
