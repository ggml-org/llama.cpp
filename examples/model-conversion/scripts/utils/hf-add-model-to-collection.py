#!/usr/bin/env python3

from huggingface_hub import HfApi
import argparse
import sys

def add_model_to_collection(collection_slug, model_id, note=""):
    """
    Add a model to an existing collection

    Args:
        collection_slug: The slug of the collection (e.g., "username/collection-name-12345")
        model_id: The model repository ID (e.g., "username/model-name")
        note: Optional note about the model

    Returns:
        True if successful, False if failed
    """

    # Initialize API
    api = HfApi()

    try:
        user_info = api.whoami()
        print(f"✅ Authenticated as: {user_info['name']}")

        # Verify the model exists
        print(f"🔍 Checking if model exists: {model_id}")
        try:
            model_info = api.model_info(model_id)
        except Exception as e:
            print(f"❌ Model not found or not accessible: {model_id}")
            print(f"Error: {e}")
            return False

        print(f"📚 Adding model to collection...")
        api.add_collection_item(
            collection_slug=collection_slug,
            item_id=model_id,
            item_type="model",
            note=note
        )

        print(f"✅ Model added to collection successfully!")
        print(f"🔗 Collection URL: https://huggingface.co/collections/{collection_slug}")

        return True

    except Exception as e:
        print(f"❌ Error adding model to collection: {e}")
        return False

def main():
    # This script requires that the environment variable HF_TOKEN is set with your
    # Hugging Face API token.
    api = HfApi()

    parser = argparse.ArgumentParser(description='Add model to a Huggingface Collection')
    parser.add_argument('--collection', '-c', help='The collection slug username/collection-hash', required=True)
    parser.add_argument('--model', '-m', help='The model to add to the Collection', required=True)
    parser.add_argument('--note', '-n', help='An optional note/description', required=False)
    args = parser.parse_args()

    collection = args.collection
    model = args.model
    note = args.note

    success = add_model_to_collection(
        collection_slug=collection,
        model_id=model,
        note=note
    )

    if success:
        print("\n🎉 Model added successfully!")
    else:
        print("\n❌ Failed to add model to collection")
        sys.exit(1)
if __name__ == "__main__":
    main()
