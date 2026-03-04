# `ppu-yolo-onnx-inference`

![ppu-yolo-onnx-inference](https://raw.githubusercontent.com/PT-Perkasa-Pilar-Utama/ppu-yolo-onnx-inference/refs/heads/main/assets/ppu-yolo-onnx-inference.png)

Run YOLOv11 object detection in TypeScript — server-side (Bun / Node.js) or client-side (browser). No Python, no PyTorch. Supports multiple independent model instances with separate inference sessions.

### Features

- **Dual platform** — single codebase, works in Node.js/Bun and the browser
- **Multi-instance** — load and run multiple YOLO models concurrently
- **Lightweight** — powered by `onnxruntime-node` (server) or `onnxruntime-web` (browser)
- **Zero config** — sensible defaults, minimal setup required

## Installation

```bash
npm install ppu-yolo-onnx-inference
```

### Platform dependencies

Install the runtime for your target platform:

```bash
# Server (Node.js / Bun)
npm install onnxruntime-node

# Browser
npm install onnxruntime-web
```

Both are declared as optional peer dependencies — install only what you need.

## Quick start

### Server-side (Node.js / Bun)

```ts
import { YoloDetectionInference } from "ppu-yolo-onnx-inference";
import { readFileSync } from "fs";

const modelBuffer = readFileSync("./coco128.onnx").buffer;

const detector = new YoloDetectionInference({
  model: {
    onnx: modelBuffer,
    classNames: ["person", "car", "bicycle"],
  },
  thresholds: { confidence: 0.5 },
});

await detector.init();
const detections = await detector.detect(imageBuffer);
await detector.destroy();
```

See the [server-side demo repo](https://github.com/PT-Perkasa-Pilar-Utama/yolo-onnx-bun-demo) for a complete example.

### Client-side (browser)

```ts
import { YoloDetectionInference } from "ppu-yolo-onnx-inference/web";

const response = await fetch("/model.onnx");
const modelBuffer = await response.arrayBuffer();

const detector = new YoloDetectionInference({
  model: {
    onnx: modelBuffer,
    classNames: ["person", "car", "bicycle"],
  },
  thresholds: { confidence: 0.5 },
});

await detector.init();
const detections = await detector.detect(imageBuffer);
await detector.destroy();
```

Try the [live client-side demo](https://pt-perkasa-pilar-utama.github.io/ppu-yolo-onnx-inference/) — runs entirely in the browser with webcam support.

### Using via CDN (no bundler)

For plain HTML pages, use an import map to resolve bare specifiers:

```html
<script type="importmap">
{
  "imports": {
    "onnxruntime-web": "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.2/dist/ort.all.bundle.min.mjs",
    "onnxruntime-common": "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.2/dist/ort.all.bundle.min.mjs",
    "ppu-ocv/web": "https://cdn.jsdelivr.net/npm/ppu-ocv@2/index.web.js",
    "@techstark/opencv-js": "https://cdn.jsdelivr.net/npm/@aspect-build/aspect-opencv-js@4.10.0-release.2/opencv.js"
  }
}
</script>
<script type="module">
  import { YoloDetectionInference } from "https://cdn.jsdelivr.net/npm/ppu-yolo-onnx-inference@2/web/index.js";

  // ... same API as above
</script>
```

## Getting the ONNX model and class names

See [`yolo-convert-onnx.py`](./examples/yolo-convert-onnx.py) to export a YOLO model to ONNX format and extract the class name list.

## Configuration

All options are grouped under the `YoloDetectionOptions` interface:

```ts
interface YoloDetectionOptions {
  model: ModelOptions;
  thresholds?: ModelThresholds;
  modelMetadata?: ModelMetadata;
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
| `inputTensorName`  | `string`           | Input tensor name (default from model metadata).                  |
| `outputTensorName` | `string`           | Output tensor name (default from model metadata).                 |

#### `DebuggingOptions`

| Property      |   Type    | Default | Description                                              |
| ------------- | :-------: | :-----: | :------------------------------------------------------- |
| `verbose`     | `boolean` | `false` | Turn on detailed console logs of each processing step.   |
| `debug`       | `boolean` | `false` | Write intermediate image frames to disk (server only).   |
| `debugFolder` | `string`  | `"out"` | Directory (relative to CWD) to save debug image outputs. |

## Result format

```ts
[
  {
    box: { x: 275, y: 6, width: 24, height: 38 },
    className: "person",
    classId: 0,
    confidence: 0.987,
  },
  {
    box: { x: 5, y: 2, width: 24, height: 38 },
    className: "car",
    classId: 1,
    confidence: 0.978,
  },
];
```

## Architecture

The library uses a Platform Provider pattern to share all business logic between server and browser:

```
src/
├── core/              # Platform-agnostic (single source of truth)
│   ├── platform.ts    # PlatformProvider interface
│   └── base-yolo-inference.ts
├── processor/         # Node.js wrapper (onnxruntime-node + ppu-ocv)
├── web/               # Browser wrapper (onnxruntime-web + ppu-ocv/web)
├── interface.ts       # Shared types
├── constant.ts        # Default thresholds
└── index.ts           # Node.js entrypoint
```

Import paths:

| Environment | Import                                  |
| ----------- | --------------------------------------- |
| Node / Bun  | `ppu-yolo-onnx-inference`               |
| Browser     | `ppu-yolo-onnx-inference/web`           |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes and add tests
4. Submit a pull request

### Running tests

```bash
bun test
```

## Scripts

Library template: https://github.com/aquapi/lib-template

### [Build](./scripts/build.ts)

Emit `.js` and `.d.ts` files to [`lib`](./lib).

### [Publish](./scripts/publish.ts)

Move [`package.json`](./package.json), [`README.md`](./README.md) to [`lib`](./lib) and publish the package.

## License

MIT — see [LICENSE](LICENSE).
