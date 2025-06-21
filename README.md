# `ppu-yolo-onnx-inference`

![ppu-yolo-onnx-inference](https://raw.githubusercontent.com/PT-Perkasa-Pilar-Utama/ppu-yolo-onnx-inference/refs/heads/main/assets/ppu-yolo-onnx-inference.png)

Easily run YOLOv11 object detection models in a TypeScript Bun environment. No Python, PyTorch, or heavy dependencies needed. Supports multiple independent instances of YOLOv11 models, each with its own inference session.

See: [Demo repo](https://github.com/PT-Perkasa-Pilar-Utama/yolo-onnx-bun-demo)

YOLO in javascript runtime should as easy as:

```ts
import { YoloDetectionInference } from "ppu-yolo-onnx-inference";

const model = new YoloDetectionInference({
  model: {
    path: "./model.onnx",
    classNames: ["person", "car", "bicycle"],
  },
  thresholds: {
    confidence: 0.5,
  },
});

await model.init();
const detections = await model.detect(imageBuffer);
await model.destroy();
```

### Why use this library?

- ✅ **Lightweight & Fast**: Inference runs with onnxruntime-web or onnxruntime-node in a JS/TS environment. No Python or PyTorch required.
- ✅ **Multi-instance Ready**: You can load and run multiple YOLO models (even different sizes) independently and concurrently.
- ✅ **Flexible Deployment**: Ideal for server-side Bun inference or potential browser/WebAssembly support in the future.
- ✅ **Easy Integration**: Minimal configuration, with out-of-the-box support for ONNX models.
- ✅ **Bun Optimized**: Designed for Bun’s performance, though can be extended for Node.js with community help.

## Installation

Install using your preferred package manager:

```bash
npm install ppu-yolo-onnx-inference
yarn add ppu-yolo-onnx-inference
bun add ppu-yolo-onnx-inference
```

> [!NOTE]
> This project is developed and tested primarily with Bun.  
> Support for Node.js, Deno, or browser environments is **not guaranteed**.
>
> If you choose to use it outside of Bun and encounter any issues, feel free to report them.  
> I'm open to fixing bugs for other runtimes with community help.

## Getting the onnx and class names

See [`yolo-convert-onnx.py`](./examples/yolo-convert-onnx.py) to get the onnx file and class name list.

## Configuration

All options are grouped under the YoloDetectionOptions interface:

```ts
export interface YoloDetectionOptions {
  /** File paths to the required Onnx YOLO model and class name list. */
  model: ModelOptions;

  /** Controls threshold for object detection. */
  thresholds?: ModelThresholds;

  /** Controls model input output tensor metadata. */
  modelMetadata?: ModelMetadata;

  /** Controls logging and image dump behavior for debugging. */
  debug?: DebuggingOptions;
}
```

#### `ModelOptions`

| Property     | Type          | Description                                                 |
| ------------ | ------------- | ----------------------------------------------------------- |
| `onnx`       | `ArrayBuffer` | The YOLOv11 ONNX model file.                                |
| `classNames` | `string[]`    | Array of class names corresponding to model output indices. |

#### `ModelThresholds`

| Property           | Type     | Description                                        |
| ------------------ | -------- | -------------------------------------------------- |
| `confidence?`      | `number` | Minimum confidence for a detection (default 0.75). |
| `iou?`             | `number` | IOU threshold for NMS filtering (default 0.5).     |
| `classConfidence?` | `number` | Per-class confidence threshold (default 0.2).      |

#### `ModelMetadata`

| Property           | Type               | Description                                                       |
| ------------------ | ------------------ | ----------------------------------------------------------------- |
| `inputShape`       | `[number, number]` | Input image shape (e.g., [640, 640]). Defaults to model metadata. |
| `inputTensorName`  | `string`           | Output tensor name (default from model metadata).                 |
| `outputTensorName` | `string`           | Input tensor name (default from model metadata).                  |

#### `DebuggingOptions`

| Property      |   Type    | Default | Description                                              |
| ------------- | :-------: | :-----: | :------------------------------------------------------- |
| `verbose`     | `boolean` | `false` | Turn on detailed console logs of each processing step.   |
| `debug`       | `boolean` | `false` | Write intermediate image frames to disk.                 |
| `debugFolder` | `string`  | `"out"` | Directory (relative to CWD) to save debug image outputs. |

## Result example

```ts
[
  {
    box: {
      x: 275,
      y: 6,
      width: 24,
      height: 38,
    },
    className: "person",
    classId: 0,
    confidence: 0.9873744249343872,
  },
  {
    box: {
      x: 5,
      y: 2,
      width: 24,
      height: 38,
    },
    className: "car",
    classId: 1,
    confidence: 0.9779072999954224,
  },
  {
    box: {
      x: 247,
      y: 6,
      width: 24,
      height: 39,
    },
    className: "bicycle",
    classId: 2,
    confidence: 0.9770053625106812,
  },
  {
    box: {
      x: 32,
      y: 3,
      width: 23,
      height: 38,
    },
    className: "car",
    classId: 1,
    confidence: 0.9710473418235779,
  },
];
```

## Contributing

Contributions are welcome! If you would like to contribute, please follow these steps:

1. **Fork the Repository:** Create your own fork of the project.
2. **Create a Feature Branch:** Use a descriptive branch name for your changes.
3. **Implement Changes:** Make your modifications, add tests, and ensure everything passes.
4. **Submit a Pull Request:** Open a pull request to discuss your changes and get feedback.

### Running Tests

This project uses Bun for testing. To run the tests locally, execute:

```bash
bun test
```

Ensure that all tests pass before submitting your pull request.

## Scripts

Recommended development environment is in linux-based environment.  
Library template: https://github.com/aquapi/lib-template

All script sources and usage.

### [Build](./scripts/build.ts)

Emit `.js` and `.d.ts` files to [`lib`](./lib).

### [Publish](./scripts/publish.ts)

Move [`package.json`](./package.json), [`README.md`](./README.md) to [`lib`](./lib) and publish the package.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Support

If you encounter any issues or have suggestions, please open an issue in the repository.

Happy coding!
