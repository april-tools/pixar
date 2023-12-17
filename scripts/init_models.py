from LatentPixel.modeling.compressors.cnn_blocks import ResnetBlock
from LatentPixel.modeling.llama import LlamaForPatchCausalInference, LlamaConfig, LlamaConfig
from LatentPixel import LatentGPT2
import torch
from pixel import PIXELForPreTraining

from LatentPixel import LatentLlama


config = LlamaConfig.from_pretrained('storage/llama_large/config.json')
setattr(config, 'num_channel', 1)
setattr(config, 'patch_len', 2)
setattr(config, 'patch_size', 8)
model = LlamaForPatchCausalInference(config)
model.save_pretrained('storage/llama_large/')
n = 0
for p in model.parameters():
		n += p.numel()
print(n)
config = LlamaConfig.from_pretrained('storage/llama_medium/config.json')
setattr(config, 'num_channel', 1)
setattr(config, 'patch_len', 2)
setattr(config, 'patch_size', 8)
model = LlamaForPatchCausalInference(config)
model.save_pretrained('storage/llama_medium/')
n = 0
for p in model.parameters():
		n += p.numel()
print(n)
