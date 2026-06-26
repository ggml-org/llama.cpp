#!/usr/bin/env python3

from huggingface_hub import HfApi
import argparse

# This script requires that the environment variable HF_TOKEN is set with your
# Hugging Face API token.
api = HfApi()

def load_template_and_substitute(template_path, **kwargs):
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()

        return template_content.format(**kwargs)
    except FileNotFoundError:
        print(f"Template file '{template_path}' not found!")
        return None
    except KeyError as e:
        print(f"Missing template variable: {e}")
        return None

parser = argparse.ArgumentParser(description='Create a new Hugging Face model repository')
parser.add_argument('--model-name', '-m', help='Name for the model', required=True)
parser.add_argument('--namespace', '-ns', help='Namespace to add the model to', required=True)
parser.add_argument('--org-base-model', '-b', help='Original Base model name', default="")
parser.add_argument('--no-card', action='store_true', help='Skip creating model card')
parser.add_argument('--private', '-p', action='store_true', help='Create private model')
parser.add_argument('--embedding', '-e', action='store_true', help='Use embedding model card template')
parser.add_argument('--dry-run', '-d', action='store_true', help='Print repository info and template without creating repository')

args = parser.parse_args()

repo_id = f"{args.namespace}/{args.model_name}-GGUF"
print("Repository ID: ", repo_id)

repo_url = None
if not args.dry_run:
    repo_url = api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=False
    )

if not args.no_card:
    if args.embedding:
        template_path = "scripts/embedding/modelcard.template"
    else:
        template_path = "scripts/causal/modelcard.template"

    print("Template path: ", template_path)

    model_card_content = load_template_and_substitute(
        template_path,
        model_name=args.model_name,
        namespace=args.namespace,
        base_model=args.org_base_model,
    )

    if args.dry_run:
        print("\nTemplate Content:\n")
        print(model_card_content)
    else:
        if model_card_content:
            api.upload_file(
                path_or_fileobj=model_card_content.encode('utf-8'),
                path_in_repo="README.md",
                repo_id=repo_id
            )
            print("Model card created successfully.")
        else:
            print("Failed to create model card.")

if not args.dry_run and repo_url:
    print(f"Repository created: {repo_url}")


