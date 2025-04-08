# Evaluation-code-for-text-to-3d
Evaluation code for text-to-3D using huggingface model weight (because my remote server can only download weight from huggingface hub) :laughing:

**Note:** image rendered by method in threestudio is usually concat with depth map and alpha image. 

# Installation

## Option 1: Install as a package
```bash
# Clone the repository
git clone https://github.com/yourusername/evaluation-code-for-text-to-3d.git
cd evaluation-code-for-text-to-3d

# Install in development mode
pip install -e .
```

## Option 2: Install dependencies only
```bash
pip install -r requirements.txt
```

# Usage

## As a Python Package
```python
import eval_3d
from eval_3d.src.model import get_feature_extractor
from eval_3d.utils import clip_score_helper_function

clip_model = get_feature_extractor("clip", pretrained=True)
# Example: Compute CLIP score
clip_score = clip_score_helper_function(
    clip_model,
    text_prompt="your text prompt",
    image_path="path/to/image_generated",
)
```

## Command Line Interface

### CLIP Score
We can compute CLIP score between text prompt and images rendered by threestudio. 
```python
import eval_3d
from eval_3d.src.model import get_feature_extractor
from eval_3d.utils import clip_score_helper_function

clip_model = get_feature_extractor("clip", pretrained=True)
# Example: Compute CLIP score
clip_score = clip_score_helper_function(
    clip_model,
    text_prompt="your text prompt",
    image_path="path/to/image_generated",
)
```

### 3D-FID Score
We can compute 3D-FID score between 2 image folders, the first folder is rendered by threestudio, the second folder is generated by Diffusion model.

```python
import eval_3d
from eval_3d.src.model import get_feature_extractor
from eval_3d.utils import fid_score_helper_function

model = get_feature_extractor("fid", pretrained=True)
# Example: Compute FID score
fid_score = fid_score_helper_function(
    model,
    generate_image_dir="generated path",
    real_image_dir="real path",
)
```

### Clip-MMD Score
We can compute CMMD score between 2 image folders, the first folder is rendered by threestudio, the second folder is generated by Diffusion model.

```python
import eval_3d
from eval_3d.src.model import get_feature_extractor
from eval_3d.utils import cmmd_score_helper_function

model = get_feature_extractor("clip", pretrained=True)
# Example: Compute FID score
fid_score = cmmd_score_helper_function(
    model,
    generate_image_dir="generated path",
    real_image_dir="real path",
)
```

# Background
## Inception Variety
- Link paper: [Taming Mode Collapse in Score Distillation for Text-to-3D Generation](https://arxiv.org/abs/2401.00909)

The formulation of IV score 
$$
IV(\theta) = H[\mathbb{E}_c[p_{cls}(y|g(\theta, c))]]
$$
The higher IV signifies that each rendered view is likely to have a distinct label prediction, meaning the 3D creation has higher view diversity => less Janus (???)

The problem with IV score is that if the rendered image is not good => the Inception backbone don't label it certainly => the entropy is high. Combining the IV and IQ score, we have IG (Information Gain) $IG=(IV - IQ)/IQ$ 

