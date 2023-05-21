sne4onnx \
--input_onnx_file_path onnx-models/large-v2/large-v2_encoder_11.onnx \
--input_op_names mel \
--output_op_names /blocks.14/Add_output_0 \
--output_onnx_file_path onnx-models/large-v2/large-v2_encoder_11_pre.onnx

sne4onnx \
--input_onnx_file_path onnx-models/large-v2/large-v2_encoder_11.onnx \
--input_op_names /blocks.14/Add_output_0 \
--output_op_names output \
--output_onnx_file_path onnx-models/large-v2/large-v2_encoder_11_post.onnx



onnx2json \
--input_onnx_file_path onnx-models/large-v2/large-v2_encoder_11.onnx \
--output_json_path onnx-models/large-v2/large-v2_encoder_11.json \
--json_indent 2


wc -l onnx-models/large-v2/large-v2_encoder_11.json
# 39779 onnx-models/large-v2/large-v2_encoder_11.json


cat onnx-models/large-v2/large-v2_encoder_11.json | split -a5

cat xa* > onnx-models/large-v2/large-v2_encoder_11_.json
