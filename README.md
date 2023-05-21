# Whisper ONNX exporter

A tool to export OpenAI Whisper speech recognition models to ONNX.

The core model file (`model.py`) has been isolated from the [original Whisper codebase](https://github.com/openai/whisper). Other files are not included or needed.

Taking some of the code in [`whisper-openvino`](https://github.com/zhuzilin/whisper-openvino) as a starting point, the model's key-value structure has been modified to be passed as an input or output, removing the need for hooks.

The `TextDecoder`, `ResidualAttentionBlock` and `MultiHeadAttention` classes have also been further modified to directly output the cross-attention weights, without any hooks.

The exported ONNX models are primarily intended to be used with [Echogarden](https://github.com/echogarden-project/echogarden), which has its own implementation of the higher-level Whisper API, and is written in TypeScript. The code doesn't include a way to use the exported models from Python. However, since it is closely related to the code on [`whisper-openvino`](https://github.com/zhuzilin/whisper-openvino), which adapts the higher-level Python API to use it, it should be possible to make it work with it, with some modifications.

## Downloading pre-exported models

You can download pre-exported models for all sizes, except `large`, `large-v1` and `large-v2`, from the releases section of the [`whisper-onnx-models` repository](https://github.com/echogarden-project/whisper-onnx-models).

## Usage

Ensure you have PyTorch installed.

Copy the official Whisper model files (`.pt`) to the `pytorch-models` subdirectory.
https://github.com/openai/whisper/blob/248b6cb124225dd263bb9bd32d060b6517e067f8/whisper/__init__.py#L17-L29

```bash
wget -O pytorch-models/tiny.en.pt "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/tiny.en.pt"
wget -O pytorch-models/tiny.pt "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/tiny.pt"
wget -O pytorch-models/base.en.pt "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/base.en.pt"
wget -O pytorch-models/base.pt "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/base.pt"
wget -O pytorch-models/small.en.pt "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/small.en.pt"
wget -O pytorch-models/small.pt "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/small.pt"
wget -O pytorch-models/medium.en.pt "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/medium.en.pt"
wget -O pytorch-models/medium.pt "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/medium.pt"
wget -O pytorch-models/large-v1.pt "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/large-v1.pt"
wget -O pytorch-models/large-v2.pt "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/large-v2.pt"
wget -O pytorch-models/large.pt "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/large.pt"
```

Run:
```
python export-whisper-onnx.py [whisper-model-name]
```

For example:
```
python export-whisper-onnx.py tiny
```

The exported encoder and decoder ONNX models would be located at:
```
onnx-models/tiny/encoder.onnx
onnx-models/tiny/decoder.onnx
```

Float32 to Float16 convert:

https://github.com/quanvuhust/Export-ONNX-float-16

https://zenn.dev/pinto0309/scraps/dc3e7b8ec32492

## Converted Models
https://github.com/PINTO0309/PINTO_model_zoo/tree/main/381_Whisper

## License

MIT
