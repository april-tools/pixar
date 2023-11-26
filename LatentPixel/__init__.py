from .modeling import (
    LPixelForMLM,
    LatentGPT2,
    LPixelForClassification,
    Discriminator,
    DiscriminatorConfig,
    CNNAutoencoderConfig,
    CNNAutoencoder,
    LatentLlama,
    SDAutoencoder,
    LatentLlamaForSequenceClassification
)
from .utils import (
    init_render,
    init_timestamp,
    timestamp,
    params2dict,
    timeit
)
from .config import (
    ModelType,
    RenderConfig
)
from .text_graph import (
    TGraph
)
from .dataprocess import (
    get_pixel_pretrain_dataloader,
    get_glue_dataset
)
from .metrics import EditDistance
from .training import GLUE_META

BIN_FONT = 'PixeloidSans-mLxMm.ttf'

DEFAULT_BINARY_RENDERING = {
    'dpi': 80,
    'font_size': 8,
    'pixels_per_patch': 8,
    'pad_size': 3,
    'font_file': BIN_FONT,
    'path': 'storage/pixel-base',
    'rgb':  False,
    'binary': True,
    'max_seq_length': 720,
    'mask_ratio': 0.25,
    'patch_len': 2,
    'num_workers': 1
}

__all__ = [
    'ModelType'
    'TGraph'
    'init_render'
    'init_timestamp'
    'timestamp'
    'get_pixel_pretrain_dataloader'
    'RenderConfig'
    'LPixelForMLM'
    'EditDistance'
    'params2dict'
    'timeit'
    'LatentGPT2'
    'LPixelForClassification'
    'get_glue_dataset'
    'GLUE_META'
    'Discriminator'
    'DiscriminatorConfig'
    'CNNAutoencoderConfig'
    'CNNAutoencoder'
    'LatentLlama'
    'SDAutoencoder'
    'BIN_FONT'
    'LatentLlamaForSequenceClassification'
    'DEFAULT_BINARY_RENDERING'
]
