
#!/usr/bin/env bash

COLLECTION_SLUG=$(python ./create_collection.py --return-slug)
echo "Created collection: $COLLECTION_SLUG"

# Use it in the next command
python add_model_to_collection.py "$COLLECTION_SLUG" "username/my-model"
