/**
 * @module ppu-yolo-onnx-inference
 *
 * Node.js / Bun entrypoint for YOLOv11 object detection inference.
 * Uses `onnxruntime-node` for ONNX model execution and `ppu-ocv`
 * for image processing.
 *
 * @example
 * ```ts
 * import { YoloDetectionInference } from "ppu-yolo-onnx-inference";
 *
 * const detector = new YoloDetectionInference({
 *   model: { onnx: modelBuffer, classNames: ["person", "car"] },
 * });
 *
 * await detector.init();
 * const detections = await detector.detect(imageBuffer);
 * await detector.destroy();
 * ```
 */
export type {
  Box,
  DebuggingOptions,
  DetectedObject,
  ModelMetadata,
  ModelOptions,
  ModelThresholds,
  PreprocessYoloResult,
  TensorValueMetadata,
  ValueMetadataBase,
  YoloDetectionOptions,
} from "./interface.js";

export { YoloDetectionInference } from "./processor/yolo-inference.service.js";

export {
  DEFAULT_DEBUG_OPTIONS,
  DEFAULT_THRESHOLDS,
  STANDARD_MODEL_INPUT_SHAPE,
} from "./constant.js";
