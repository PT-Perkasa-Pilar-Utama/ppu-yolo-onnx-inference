/**
 * @module ppu-yolo-onnx-inference/web
 *
 * Browser entrypoint for YOLOv11 object detection inference.
 * Uses `onnxruntime-web` for ONNX model execution and `ppu-ocv/web`
 * for image processing, enabling fully client-side object detection.
 *
 * @example
 * ```ts
 * import { YoloDetectionInference } from "ppu-yolo-onnx-inference/web";
 *
 * const modelResponse = await fetch("/model.onnx");
 * const modelBuffer = await modelResponse.arrayBuffer();
 *
 * const detector = new YoloDetectionInference({
 *   model: { onnx: modelBuffer, classNames: ["person", "car"] },
 * });
 *
 * await detector.init();
 * const detections = await detector.detect(imageArrayBuffer);
 * ```
 */
export { YoloDetectionInference } from "./yolo-inference.web.js";

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
} from "../interface.js";

export {
  DEFAULT_DEBUG_OPTIONS,
  DEFAULT_THRESHOLDS,
  STANDARD_MODEL_INPUT_SHAPE,
} from "../constant.js";
