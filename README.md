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
curl -o pytorch-models/tiny.en.pt "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt"
curl -o pytorch-models/tiny.pt "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
curl -o pytorch-models/base.en.pt "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt"
curl -o pytorch-models/base.pt "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt"
curl -o pytorch-models/small.en.pt "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt"
curl -o pytorch-models/small.pt "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt"
curl -o pytorch-models/medium.en.pt "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt"
curl -o pytorch-models/medium.pt "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt"
curl -o pytorch-models/large-v1.pt "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt"
curl -o pytorch-models/large-v2.pt "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt"
curl -o pytorch-models/large.pt "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt"
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

## License

MIT
