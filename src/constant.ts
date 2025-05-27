import type { DebuggingOptions, ModelThresholds } from "./interface";

export const DEFAULT_THRESHOLDS: Required<ModelThresholds> = {
  confidence: 0.75,
  iou: 0.5,
  classConfidence: 0.2,
};

export const DEFAULT_DEBUG_OPTIONS: Required<DebuggingOptions> = {
  verbose: false,
  debug: false,
  debugFolder: "./out",
};

export const STANDARD_MODEL_INPUT_SHAPE: [number, number] = [640, 640];
