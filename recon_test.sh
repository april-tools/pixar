#!/bin/bash

# python recognition_test.py \
#     --font 'GoNotoCurrent.ttf' 'GoNotoCurrent.ttf'  'GoNotoCurrent.ttf' 'GoNotoCurrent.ttf' 'GoNotoCurrent.ttf' 'GoNotoCurrent.ttf' 'PixeloidSans-mLxMm.ttf' 'PixeloidSans-mLxMm.ttf' 'PixeloidSans-mLxMm.ttf' 'PixeloidSans-mLxMm.ttf' 'PixeloidSans-mLxMm.ttf' 'PixeloidSans-mLxMm.ttf'  \
#     --pixels_per_patch 32 24 16 8 8 8 32 24 16 8 8 8 \
#     --dpi 240 180 120 60 60 60 240 180 120 60 60 60 \
#     --font_size 8 8 8 8 8 8 8 8 8 8 8 8 \
#     --rgb True True True True True False True True True True True False  \
#     --binary False False False False False True False False False False False True  \

# python recognition_test.py \
#     --font 'GoNotoCurrent.ttf' 'GoNotoCurrent.ttf'  'GoNotoCurrent.ttf' 'GoNotoCurrent.ttf' 'GoNotoCurrent.ttf' 'GoNotoCurrent.ttf' 'PixeloidSans-mLxMm.ttf' 'PixeloidSans-mLxMm.ttf' 'PixeloidSans-mLxMm.ttf' 'PixeloidSans-mLxMm.ttf' 'PixeloidSans-mLxMm.ttf' 'PixeloidSans-mLxMm.ttf'  \
#     --pixels_per_patch 32 24 16 8 8 8 32 24 16 8 8 8 \
#     --dpi 240 180 120 60 60 60 240 180 120 60 60 60 \
#     --font_size 8 8 8 8 8 8 8 8 8 8 8 8 \
#     --rgb True True True True True False True True True True True False  \
#     --binary False False False False False True False False False False False True  \
#     --coder_path storage/SD2_VQGAN

python recognition_test.py \
    --font 'GoNotoCurrent.ttf' 'GoNotoCurrent.ttf' 'GoNotoCurrent.ttf' 'GoNotoCurrent.ttf' 'GoNotoCurrent.ttf' 'GoNotoCurrent.ttf' 'GoNotoCurrent.ttf' 'PixeloidSans-mLxMm.ttf' 'PixeloidSans-mLxMm.ttf' 'PixeloidSans-mLxMm.ttf' 'PixeloidSans-mLxMm.ttf' 'PixeloidSans-mLxMm.ttf' 'PixeloidSans-mLxMm.ttf' 'PixeloidSans-mLxMm.ttf' 'PixeloidSans-mLxMm.ttf' \
    --pixels_per_patch 32 24 16 12 8 8 8 32 24 16 16 12 8 8 8 \
    --dpi 240 180 120 80 60 60 60 240 180 120 120 80 60 60 60 \
    --font_size 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 \
    --rgb True True True True True True False True True True False True True True False  \
    --binary False False False False False False True False False False True False False False True  \
    --coder_path storage/SD2_VQGAN

# python recognition_test.py \
#     --font 'GoNotoCurrent.ttf' 'GoNotoCurrent.ttf' 'GoNotoCurrent.ttf' 'GoNotoCurrent.ttf' 'GoNotoCurrent.ttf' 'GoNotoCurrent.ttf' 'GoNotoCurrent.ttf' 'PixeloidSans-mLxMm.ttf' 'PixeloidSans-mLxMm.ttf' 'PixeloidSans-mLxMm.ttf' 'PixeloidSans-mLxMm.ttf' 'PixeloidSans-mLxMm.ttf' 'PixeloidSans-mLxMm.ttf' 'PixeloidSans-mLxMm.ttf' \
#     --pixels_per_patch 32 24 16 12 8 8 8 32 24 16 12 8 8 8 \
#     --dpi 240 180 120 80 60 60 60 240 180 120 80 60 60 60 \
#     --font_size 8 8 8 8 8 8 8 8 8 8 8 8 8 8 \
#     --rgb True True True True True True False True True True True True True False  \
#     --binary False False False False False False True False False False False False False True  \
