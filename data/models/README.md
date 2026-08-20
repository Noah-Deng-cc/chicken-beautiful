# Runtime Model Manifest

The model binary is intentionally not tracked in Git. Provision it on the Raspberry Pi as
`data/models/emotion.onnx`.

Current validation source:

- URL: <https://github.com/Shohruh72/Emotion_onnx/releases/download/v.1.0.0/emotion.onnx>
- SHA-256: `7863032d8c0c0bc259aa0dd7b8ebf6b607af454e7a0ea82be2b37767914ecd0f`
- OpenCV DNN output: `(1, 7)` classification logits
- Assumed label order: `angry, disgusted, fearful, happy, neutral, sad, surprised`

The label order is an integration assumption based on the seven-class FER convention and must
be rechecked against any replacement model before deployment. The runtime applies stable softmax
to raw seven-value logits, then applies the configured confidence threshold.
