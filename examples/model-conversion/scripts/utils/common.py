#!/usr/bin/env python3

import os
import sys

def get_model_name_from_env_path(env_path_name):
    model_path = os.getenv(env_path_name)
    if not model_path:
        print(f"Error: {env_path_name} environment variable not set")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    name = os.path.basename(os.path.normpath(model_path))
    if name.endswith(".gguf"):
        name = name[:-5]

    return name
