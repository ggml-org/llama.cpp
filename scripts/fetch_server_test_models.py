#!/usr/bin/env python
'''
    This script fetches all the models used in the server tests.

    This is useful for slow tests that use larger models, to avoid them timing out on the model downloads.

    It is meant to be run from the root of the repository.

    Example:
        python scripts/fetch_server_test_models.py
        ( cd examples/server/tests && ./tests.sh -v -x -m slow )
'''
import ast
import glob
import logging
import os
from typing import Generator
from pydantic import BaseModel
from typing import Optional
import subprocess


class HuggingFaceModel(BaseModel):
    hf_repo: str
    hf_file: Optional[str] = None

    class Config:
        frozen = True


def collect_hf_model_test_parameters(test_file) -> Generator[HuggingFaceModel, None, None]:
    try:
        with open(test_file) as f:
            tree = ast.parse(f.read())
    except Exception as e:
        logging.error(f'collect_hf_model_test_parameters failed on {test_file}: {e}')
        return

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr == 'parametrize':
                    param_names = ast.literal_eval(dec.args[0]).split(",")
                    if "hf_repo" not in param_names:
                        continue

                    raw_param_values = dec.args[1]
                    if not isinstance(raw_param_values, ast.List):
                        logging.warning(f'Skipping non-list parametrize entry at {test_file}:{node.lineno}')
                        continue

                    hf_repo_idx = param_names.index("hf_repo")
                    hf_file_idx = param_names.index("hf_file") if "hf_file" in param_names else None

                    for t in raw_param_values.elts:
                        if not isinstance(t, ast.Tuple):
                            logging.warning(f'Skipping non-tuple parametrize entry at {test_file}:{node.lineno}')
                            continue
                        yield HuggingFaceModel(
                            hf_repo=ast.literal_eval(t.elts[hf_repo_idx]),
                            hf_file=ast.literal_eval(t.elts[hf_file_idx]) if hf_file_idx is not None else None)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    models = sorted(list(set([
        model
        for test_file in glob.glob('examples/server/tests/unit/test_*.py')
        for model in collect_hf_model_test_parameters(test_file)
    ])), key=lambda m: (m.hf_repo, m.hf_file))

    logging.info(f'Found {len(models)} models in parameterized tests:')
    for m in models:
        logging.info(f'  - {m.hf_repo} / {m.hf_file}')

    cli_path = os.environ.get(
        'LLAMA_SERVER_BIN_PATH',
        os.path.join(
            os.path.dirname(__file__),
            '../build/bin/Release/llama-cli.exe' if os.name == 'nt' else '../build/bin/llama-cli'))

    for m in models:
        if '<' in m.hf_repo or (m.hf_file is not None and '<' in m.hf_file):
            continue
        if m.hf_file is not None and '-of-' in m.hf_file:
            logging.warning(f'Skipping model at {m.hf_repo} / {m.hf_file} because it is a split file')
            continue
        logging.info(f'Using llama-cli to ensure model {m.hf_repo}/{m.hf_file} was fetched')
        cmd = [
            cli_path,
            '-hfr', m.hf_repo,
            *([] if m.hf_file is None else ['-hff', m.hf_file]),
            '-n', '1',
            '-p', 'Hey',
            '--no-warmup',
            '--log-disable',
            '-no-cnv']
        if m.hf_file != 'tinyllamas/stories260K.gguf' and 'Mistral-Nemo' not in m.hf_repo:
            cmd.append('-fa')
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError:
            logging.error(f'Failed to fetch model at {m.hf_repo} / {m.hf_file} with command:\n  {" ".join(cmd)}')
            exit(1)
