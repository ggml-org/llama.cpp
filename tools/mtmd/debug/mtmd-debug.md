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

Example of debugging a rainbow image:

```py
import torch
import math

def make_rainbow(img_size):
    cx, cy = img_size / 2.0, img_size / 2.0
    max_dist = math.sqrt(cx * cx + cy * cy)
    img = torch.zeros(1, 3, img_size, img_size)
    for y in range(img_size):
        for x in range(img_size):
            dx, dy = x - cx, y - cy
            hue = math.atan2(dy, dx) / (2 * math.pi)
            if hue < 0:
                hue += 1
            sat = math.sqrt(dx * dx + dy * dy) / max_dist
            sat = min(sat, 1.0)
            h6 = hue * 6
            i6 = int(h6)
            f = h6 - i6
            p = 1 - sat
            q = 1 - sat * f
            t = 1 - sat * (1 - f)
            rgb = [(1,t,p),(q,1,p),(p,1,t),(p,q,1),(t,p,1),(1,p,q)][i6 % 6]
            img[0, 0, y, x] = rgb[0]
            img[0, 1, y, x] = rgb[1]
            img[0, 2, y, x] = rgb[2]
    return img

img_size = 896
pixel_values = make_rainbow(img_size)
with torch.no_grad():
    outputs = model.model.get_image_features(pixel_values=pixel_values)
print("last_hidden_state:", outputs.last_hidden_state)
```

## Debugging preprocess pass

(TODO)
