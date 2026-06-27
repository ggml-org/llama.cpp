#!/usr/bin/env python3

from huggingface_hub import HfApi
import argparse
import os
import sys


def create_collection(title, description, private=False, namespace=None, return_slug=False):
    """
    Create a new collection on Hugging Face

    Args:
        title: Collection title
        description: Collection description
        private: Whether the collection should be private (default: False)
        namespace: Optional namespace (defaults to your username)

    Returns:
        Collection object if successful, None if failed
    """

    # Check if HF_TOKEN is available
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        print("❌ No HF_TOKEN or HUGGINGFACE_HUB_TOKEN found in environment variables")
        print("Please set your Hugging Face token as an environment variable")
        return None

    # Initialize API
    api = HfApi()

    try:
        # Test authentication first
        user_info = api.whoami()
        if not return_slug:
            print(f"✅ Authenticated as: {user_info['name']}")

        # Create the collection
        if not return_slug:
            print(f"📚 Creating collection: '{title}'...")
        collection = api.create_collection(
            title=title,
            description=description,
            private=private,
            namespace=namespace
        )

        if not return_slug:
            print(f"✅ Collection created successfully!")
            print(f"📋 Collection slug: {collection.slug}")
            print(f"🔗 Collection URL: https://huggingface.co/collections/{collection.slug}")

        return collection

    except Exception as e:
        print(f"❌ Error creating collection: {e}")
        return None

def main():
    # This script requires that the environment variable HF_TOKEN is set with your
    # Hugging Face API token.
    api = HfApi()

    parser = argparse.ArgumentParser(description='Create a Huggingface Collection')
    parser.add_argument('--name', '-n', help='The name/title of the Collection', required=True)
    parser.add_argument('--description', '-d', help='The description for the Collection', required=True)
    parser.add_argument('--namespace', '-ns', help='The namespace to add the Collection to', required=True)
    parser.add_argument('--private', '-p', help='Create a private Collection', action='store_true')  # Fixed
    parser.add_argument('--return-slug', '-s', help='Only output the collection slug', action='store_true')  # Fixed

    args = parser.parse_args()

    name = args.name
    description = args.description
    private = args.private
    namespace = args.namespace
    return_slug = args.return_slug

    if not return_slug:
        print("🚀 Creating Hugging Face Collection")
        print(f"Title: {name}")
        print(f"Description: {description}")
        print(f"Namespace: {namespace}")
        print(f"Private: {private}")

    collection = create_collection(
        title=name,
        description=description,
        private=private,
        namespace=namespace,
        return_slug=return_slug
    )

    if collection:
        if return_slug:
            print(collection.slug)
        else:
            print("\n🎉 Collection created successfully!")
            print(f"Use this slug to add models: {collection.slug}")
    else:
        print("\n❌ Failed to create collection")
        sys.exit(1)

if __name__ == "__main__":
    main()
