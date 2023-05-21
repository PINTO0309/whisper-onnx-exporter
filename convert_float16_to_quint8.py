from onnxruntime.quantization import quantize_dynamic, QuantType

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
    "large-v1/large-v1_decoder_11",
    "large-v1/large-v1_encoder_11",
    "large-v2/large-v2_decoder_11",
    "large-v2/large-v2_encoder_11",
]


for input_onnx_file in input_onnx_files:
    quantized_model = quantize_dynamic(
        model_input=f'onnx-models_/{input_onnx_file}.onnx',
        model_output=f'onnx-models-dynamic-int8/{input_onnx_file}_int8.onnx',
        weight_type=QuantType.QUInt8,
    )

