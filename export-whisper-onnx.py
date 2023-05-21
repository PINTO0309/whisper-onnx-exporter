import sys
import os
import torch
import onnx # v1.13.1
from onnxsim import simplify # v0.4.17
from model import Whisper, ModelDimensions

print("Using PyTorch version: {}\n".format(torch.__version__))

if len(sys.argv) <= 1:
	print("Usage: python export-whisper-onnx.py [whisper-model-name]")
	exit(0)

modelName = sys.argv[1]

validModelNames = [
	"tiny",
	"tiny.en",
	"base",
	"base.en",
	"small",
	"small.en",
	"medium",
	"medium.en",
	"large",
	"large-v1",
	"large-v2",
]

if not modelName in validModelNames:
	print("Error: model name must be one of {}".format(", ".join(validModelNames)))
	exit(1)

checkpoint = torch.load(f"pytorch-models/{modelName}.pt", map_location=torch.device('cpu'))

modelDims = ModelDimensions(**checkpoint["dims"])
whisper = Whisper(modelDims, modelName)
whisper.load_state_dict(checkpoint["model_state_dict"])
# whisper = whisper.to("cpu")
whisper = whisper.to("cuda")
batchSize = 1
audioEncoder = whisper.encoder
audioEncoderRandomInputs = \
	torch.randn(
		batchSize,
		modelDims.n_mels,
		modelDims.n_audio_ctx * 2
	).cuda()
encodedFeatures = whisper.encoder(audioEncoderRandomInputs)
outputDir = f"onnx-models"
os.makedirs(outputDir, exist_ok=True)

OPSET = 11

######### Encoder
ENCODER_FILE = f"{outputDir}/{modelName}_encoder_{OPSET}.onnx"
torch.onnx.export(
	model=audioEncoder,
	args=(audioEncoderRandomInputs),
	f=ENCODER_FILE,
	input_names=["mel"],
	output_names=["output"],
	opset_version=OPSET,
)
if modelName not in ["large", "large-v1", "large-v2"]:
	model_onnx2 = onnx.load(ENCODER_FILE)
	model_simp, check = simplify(model_onnx2)
	onnx.save(model_simp, ENCODER_FILE)
	model_onnx2 = onnx.load(ENCODER_FILE)
	model_simp, check = simplify(model_onnx2)
	onnx.save(model_simp, ENCODER_FILE)
	model_onnx2 = onnx.load(ENCODER_FILE)
	model_simp, check = simplify(model_onnx2)
	onnx.save(model_simp, ENCODER_FILE)


######### Decoder
with torch.autocast("cuda", dtype=torch.float16):
	textDecoder = whisper.decoder
	tokens = torch.tensor([[0]], dtype=torch.int64).cuda()
	kvCache = torch.from_numpy(whisper.new_kv_cache(batchSize, 1)).cuda()
	offset = torch.tensor(0).cuda()

	DECODER_FILE = f"{outputDir}/{modelName}_decoder_{OPSET}.onnx"
	torch.onnx.export(
		model=textDecoder,
		args=(tokens, encodedFeatures, kvCache, offset),
		f=DECODER_FILE,
		input_names=["tokens", "audio_features", "kv_cache", "offset"],
		output_names=["logits", "output_kv_cache", "cross_attention_qks"],
		# output_names=["x2"],
		dynamic_axes={
			"tokens": [0, 1],
			"audio_features": [0],
			"kv_cache": [1, 2],
			"output_kv_cache": [1, 2],
			"cross_attention_qks": [1, 3, 4],
		},
		opset_version=OPSET,
	)
	if modelName not in ["large", "large-v1", "large-v2"]:
		model_onnx2 = onnx.load(DECODER_FILE)
		model_simp, check = simplify(model_onnx2)
		onnx.save(model_simp, DECODER_FILE)
		model_onnx2 = onnx.load(DECODER_FILE)
		model_simp, check = simplify(model_onnx2)
		onnx.save(model_simp, DECODER_FILE)
		model_onnx2 = onnx.load(DECODER_FILE)
		model_simp, check = simplify(model_onnx2)
		onnx.save(model_simp, DECODER_FILE)