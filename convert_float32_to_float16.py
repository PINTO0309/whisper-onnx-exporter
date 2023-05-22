# pip install onnxconverter-common==1.13.0

import os
import onnx
from onnxconverter_common.float16 import convert_float_to_float16_model_path

input_onnx_files = [
    "base_decoder_11",
    "base_encoder_11",
    "base.en_decoder_11",
    "base.en_encoder_11",
    "medium_decoder_11",
    "medium_encoder_11",
    "medium.en_decoder_11",
    "medium.en_encoder_11",
    "small_decoder_11",
    "small_encoder_11",
    "small.en_decoder_11",
    "small.en_encoder_11",
    "tiny_decoder_11",
    "tiny_encoder_11",
    "tiny.en_decoder_11",
    "tiny.en_encoder_11",
]

os.makedirs('onnx-models-float16', exist_ok=True)

for input_onnx_file in input_onnx_files:
    output_onnx_model = f'onnx-models-float16/{input_onnx_file.split("/")[0] if len(input_onnx_file.split("/")) == 1 else input_onnx_file.split("/")[1]}_float16.onnx'
    model_float16 = \
        convert_float_to_float16_model_path(
            model_path=f'onnx-models/{input_onnx_file}.onnx',
            keep_io_types=True,
        )
    onnx.save(model_float16, output_onnx_model)
