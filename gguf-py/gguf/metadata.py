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
    datasets: Optional[list[dict]] = None

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

        # Datasets is received here as an array of datasets
        metadata.datasets        = metadata_override.get("general.datasets",           metadata.datasets)

        metadata.tags            = metadata_override.get(Keys.General.TAGS,            metadata.tags)
        metadata.languages       = metadata_override.get(Keys.General.LANGUAGES,       metadata.languages)

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

        # The model card metadata is assumed to always be in YAML (frontmatter)
        # ref: https://github.com/huggingface/transformers/blob/a5c642fe7a1f25d3bdcd76991443ba6ff7ee34b2/src/transformers/modelcard.py#L468-L473
        yaml_content: str = ""
        with open(model_card_path, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.splitlines()
            lines_yaml = []
            if len(lines) == 0:
                # Empty file
                return {}
            if len(lines) > 0 and lines[0] != "---":
                # No frontmatter
                return {}
            for line in lines[1:]:
                if line == "---":
                    break # End of frontmatter
                else:
                    lines_yaml.append(line)
            yaml_content = "\n".join(lines_yaml) + "\n"

        # Quick hack to fix the Norway problem
        # https://hitchdev.com/strictyaml/why/implicit-typing-removed/
        yaml_content = yaml_content.replace("- no\n", "- \"no\"\n")

        if yaml_content:
            data = yaml.safe_load(yaml_content)
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
        if org_component is not None and len(org_component) > 0 and org_component[0] == '.':
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

            def use_model_card_metadata(metadata_key: str, model_card_key: str):
                if model_card_key in model_card and getattr(metadata, metadata_key, None) is None:
                    setattr(metadata, metadata_key, model_card.get(model_card_key))

            def use_array_model_card_metadata(metadata_key: str, model_card_key: str):
                # Note: Will append rather than replace if already exist
                tags_value = model_card.get(model_card_key, None)
                if tags_value is None:
                    return

                current_value = getattr(metadata, metadata_key, None)
                if current_value is None:
                    current_value = []

                if isinstance(tags_value, str):
                    current_value.append(tags_value)
                elif isinstance(tags_value, list):
                    current_value.extend(tags_value)

                setattr(metadata, metadata_key, current_value)

            # LLAMA.cpp's direct internal convention
            # (Definitely not part of hugging face formal/informal standard)
            #########################################
            use_model_card_metadata("name", "name")
            use_model_card_metadata("author", "author")
            use_model_card_metadata("version", "version")
            use_model_card_metadata("organization", "organization")
            use_model_card_metadata("description", "description")
            use_model_card_metadata("finetune", "finetune")
            use_model_card_metadata("basename", "basename")
            use_model_card_metadata("size_label", "size_label")
            use_model_card_metadata("source_url", "url")
            use_model_card_metadata("source_doi", "doi")
            use_model_card_metadata("source_uuid", "uuid")
            use_model_card_metadata("source_repo_url", "repo_url")

            # LLAMA.cpp's huggingface style convention
            # (Definitely not part of hugging face formal/informal standard... but with model_ appended to match their style)
            ###########################################
            use_model_card_metadata("name", "model_name")
            use_model_card_metadata("author", "model_author")
            use_model_card_metadata("version", "model_version")
            use_model_card_metadata("organization", "model_organization")
            use_model_card_metadata("description", "model_description")
            use_model_card_metadata("finetune", "model_finetune")
            use_model_card_metadata("basename", "model_basename")
            use_model_card_metadata("size_label", "model_size_label")
            use_model_card_metadata("source_url", "model_url")
            use_model_card_metadata("source_doi", "model_doi")
            use_model_card_metadata("source_uuid", "model_uuid")
            use_model_card_metadata("source_repo_url", "model_repo_url")

            # Hugging Face Direct Convention
            #################################

            # Not part of huggingface model card standard but notice some model creator using it
            # such as TheBloke in 'TheBloke/Mistral-7B-Instruct-v0.2-GGUF'
            use_model_card_metadata("name", "model_name")
            use_model_card_metadata("author", "model_creator")
            use_model_card_metadata("basename", "model_type")

            if "base_model" in model_card or "base_models" in model_card or "base_model_sources" in model_card:
                # This represents the parent models that this is based on
                # Example: stabilityai/stable-diffusion-xl-base-1.0. Can also be a list (for merges)
                # Example of merges: https://huggingface.co/EmbeddedLLM/Mistral-7B-Merge-14-v0.1/blob/main/README.md
                metadata_base_models = []
                base_model_value = model_card.get("base_model", model_card.get("base_models", model_card.get("base_model_sources", None)))

                if base_model_value is not None:
                    if isinstance(base_model_value, str):
                        metadata_base_models.append(base_model_value)
                    elif isinstance(base_model_value, list):
                        metadata_base_models.extend(base_model_value)

                if metadata.base_models is None:
                    metadata.base_models = []

                for model_id in metadata_base_models:
                    # NOTE: model size of base model is assumed to be similar to the size of the current model
                    base_model = {}
                    if isinstance(model_id, str):
                        if model_id.startswith("http://") or model_id.startswith("https://") or model_id.startswith("ssh://"):
                            base_model["repo_url"] = model_id

                            # Check if Hugging Face ID is present in URL
                            if "huggingface.co" in model_id:
                                match = re.match(r"https?://huggingface.co/([^/]+/[^/]+)$", model_id)
                                if match:
                                    model_id_component = match.group(1)
                                    model_full_name_component, org_component, basename, finetune, version, size_label = Metadata.get_model_id_components(model_id_component, total_params)

                                    # Populate model dictionary with extracted components
                                    if model_full_name_component is not None:
                                        base_model["name"] = Metadata.id_to_title(model_full_name_component)
                                    if org_component is not None:
                                        base_model["organization"] = Metadata.id_to_title(org_component)
                                    if version is not None:
                                        base_model["version"] = version

                        else:
                            # Likely a Hugging Face ID
                            model_full_name_component, org_component, basename, finetune, version, size_label = Metadata.get_model_id_components(model_id, total_params)

                            # Populate model dictionary with extracted components
                            if model_full_name_component is not None:
                                base_model["name"] = Metadata.id_to_title(model_full_name_component)
                            if org_component is not None:
                                base_model["organization"] = Metadata.id_to_title(org_component)
                            if version is not None:
                                base_model["version"] = version
                            if org_component is not None and model_full_name_component is not None:
                                base_model["repo_url"] = f"https://huggingface.co/{org_component}/{model_full_name_component}"

                    elif isinstance(model_id, dict):
                        base_model = model_id

                    else:
                        logger.error(f"base model entry '{str(model_id)}' not in a known format")

                    metadata.base_models.append(base_model)

            if "datasets" in model_card or "dataset" in model_card or "dataset_sources" in model_card:
                # This represents the datasets that this was trained from
                metadata_datasets = []
                dataset_value = model_card.get("datasets", model_card.get("dataset", model_card.get("dataset_sources", None)))

                if dataset_value is not None:
                    if isinstance(dataset_value, str):
                        metadata_datasets.append(dataset_value)
                    elif isinstance(dataset_value, list):
                        metadata_datasets.extend(dataset_value)

                if metadata.datasets is None:
                    metadata.datasets = []

                for dataset_id in metadata_datasets:
                    # NOTE: model size of base model is assumed to be similar to the size of the current model
                    dataset = {}
                    if isinstance(dataset_id, str):
                        if dataset_id.startswith(("http://", "https://", "ssh://")):
                            dataset["repo_url"] = dataset_id

                            # Check if Hugging Face ID is present in URL
                            if "huggingface.co" in dataset_id:
                                match = re.match(r"https?://huggingface.co/([^/]+/[^/]+)$", dataset_id)
                                if match:
                                    dataset_id_component = match.group(1)
                                    dataset_name_component, org_component, basename, finetune, version, size_label = Metadata.get_model_id_components(dataset_id_component, total_params)

                                    # Populate dataset dictionary with extracted components
                                    if dataset_name_component is not None:
                                        dataset["name"] = Metadata.id_to_title(dataset_name_component)
                                    if org_component is not None:
                                        dataset["organization"] = Metadata.id_to_title(org_component)
                                    if version is not None:
                                        dataset["version"] = version

                        else:
                            # Likely a Hugging Face ID
                            dataset_name_component, org_component, basename, finetune, version, size_label = Metadata.get_model_id_components(dataset_id, total_params)

                            # Populate dataset dictionary with extracted components
                            if dataset_name_component is not None:
                                dataset["name"] = Metadata.id_to_title(dataset_name_component)
                            if org_component is not None:
                                dataset["organization"] = Metadata.id_to_title(org_component)
                            if version is not None:
                                dataset["version"] = version
                            if org_component is not None and dataset_name_component is not None:
                                dataset["repo_url"] = f"https://huggingface.co/{org_component}/{dataset_name_component}"

                    elif isinstance(dataset_id, dict):
                        dataset = dataset_id

                    else:
                        logger.error(f"dataset entry '{str(dataset_id)}' not in a known format")

                    metadata.datasets.append(dataset)

            use_model_card_metadata("license", "license")
            use_model_card_metadata("license_name", "license_name")
            use_model_card_metadata("license_link", "license_link")

            use_array_model_card_metadata("tags", "tags")
            use_array_model_card_metadata("tags", "pipeline_tag")

            use_array_model_card_metadata("languages", "languages")
            use_array_model_card_metadata("languages", "language")

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
            if isinstance(self.license, list):
                gguf_writer.add_license(",".join(self.license))
            else:
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
                if "description" in base_model_entry:
                    gguf_writer.add_base_model_description(key, base_model_entry["description"])
                if "url" in base_model_entry:
                    gguf_writer.add_base_model_url(key, base_model_entry["url"])
                if "doi" in base_model_entry:
                    gguf_writer.add_base_model_doi(key, base_model_entry["doi"])
                if "uuid" in base_model_entry:
                    gguf_writer.add_base_model_uuid(key, base_model_entry["uuid"])
                if "repo_url" in base_model_entry:
                    gguf_writer.add_base_model_repo_url(key, base_model_entry["repo_url"])

        if self.datasets is not None:
            gguf_writer.add_dataset_count(len(self.datasets))
            for key, dataset_entry in enumerate(self.datasets):
                if "name" in dataset_entry:
                    gguf_writer.add_dataset_name(key, dataset_entry["name"])
                if "author" in dataset_entry:
                    gguf_writer.add_dataset_author(key, dataset_entry["author"])
                if "version" in dataset_entry:
                    gguf_writer.add_dataset_version(key, dataset_entry["version"])
                if "organization" in dataset_entry:
                    gguf_writer.add_dataset_organization(key, dataset_entry["organization"])
                if "description" in dataset_entry:
                    gguf_writer.add_dataset_description(key, dataset_entry["description"])
                if "url" in dataset_entry:
                    gguf_writer.add_dataset_url(key, dataset_entry["url"])
                if "doi" in dataset_entry:
                    gguf_writer.add_dataset_doi(key, dataset_entry["doi"])
                if "uuid" in dataset_entry:
                    gguf_writer.add_dataset_uuid(key, dataset_entry["uuid"])
                if "repo_url" in dataset_entry:
                    gguf_writer.add_dataset_repo_url(key, dataset_entry["repo_url"])

        if self.tags is not None:
            gguf_writer.add_tags(self.tags)
        if self.languages is not None:
            gguf_writer.add_languages(self.languages)
