# mtmd-debug

## Debugging encode pass

Example of debugging an input gray image (raw, not preprocessed):

```py
from transformers import AutoModel

model = AutoModel.from_pretrained(...)

def test_vision():
  img_size = 896 # number of patches per side
  pixel_values = torch.zeros(1, 3, img_size, img_size) + 0.5 # gray image
  with torch.no_grad():
    outputs = model.model.get_image_features(pixel_values=pixel_values)
  print("last_hidden_state shape:", outputs.last_hidden_state.shape)
  print("last_hidden_state:", outputs.last_hidden_state)

test_vision()
```

## Debugging preprocess pass

(TODO)
