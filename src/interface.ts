import type { Tensor } from "onnxruntime-node";

/**
 * Controls verbose output and image dumps for debugging OCR.
 */
export interface DebuggingOptions {
  /**
   * Enable detailed logging of each processing step.
   * @default false
   */
  verbose?: boolean;

  /**
   * Save intermediate image data to disk for inspection.
   * @default false
   */
  debug?: boolean;

  /**
   * Directory where debug images will be written.
   * Relative to the current working directory.
   * @default "out"
   */
  debugFolder?: string;
}

/**
 * Simple rectangle representation.
 */
export interface Box {
  /** X-coordinate of the top-left corner. */
  x: number;
  /** Y-coordinate of the top-left corner. */
  y: number;
  /** Width of the box in pixels. */
  width: number;
  /** Height of the box in pixels. */
  height: number;
}

export interface ModelOptions {
  path: string; // Path to the ONNX model file
  classNames: string[]; // List of class names for object detection
}

export interface ModelThresholds {
  confidence?: number; // Minimum confidence score for a detection to be considered valid
  iou?: number; // Intersection over Union threshold for non-max suppression
  classConfidence?: number; // Minimum confidence score for a class to be considered valid
}

export interface ModelMetadata {
  inputShape: [number, number]; // Input shape of the model (width, height)
  inputTensorName: string; // Name of the input tensor in the ONNX model
  outputTensorName: string; // Name of the output tensor in the ONNX model
}

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

export interface PreprocessYoloResult {
  tensor: Float32Array;
  modelInputWidth: number;
  modelInputHeight: number;
  originalWidth: number;
  originalHeight: number;
  scaleRatio: number;
}

export interface DetectedObject {
  box: Box; // x, y, width, height in original image coordinates
  className: string;
  classId: number;
  confidence: number;
}

// Re-export interfaces for ValueMetadata and TensorValueMetadata because can't import them directly from onnxruntime-node

/**
 * The common part of the value metadata type for both tensor and non-tensor values.
 */
export interface ValueMetadataBase {
  /**
   * The name of the specified input or output.
   */
  readonly name: string;
}

/**
 * Represents the metadata of a tensor value.
 */
export interface TensorValueMetadata extends ValueMetadataBase {
  /**
   * Get a value indicating whether the value is a tensor.
   */
  readonly isTensor: true;
  /**
   * Get the data type of the tensor.
   */
  readonly type: Tensor.Type;
  /**
   * Get the shape of the tensor.
   *
   * If the shape is not defined, the value will an empty array. Otherwise, it will be an array representing the shape
   * of the tensor. Each element in the array can be a number or a string. If the element is a number, it represents
   * the corresponding dimension size. If the element is a string, it represents a symbolic dimension.
   */
  readonly shape: ReadonlyArray<number | string>;
}
