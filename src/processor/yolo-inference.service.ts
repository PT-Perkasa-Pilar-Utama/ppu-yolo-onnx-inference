import { ImageProcessor } from "ppu-ocv";
import { BaseYoloDetectionInference } from "../core/base-yolo-inference.js";
import type { CoreCanvas } from "../core/platform.js";
import type { YoloDetectionOptions } from "../interface.js";
import { NodePlatformProvider } from "./platform.node.js";

/**
 * YOLOv11 Object Detection Inference Engine for Node.js / Bun.
 *
 * @example
 * ```ts
 * import { YoloDetectionInference } from "ppu-yolo-onnx-inference";
 *
 * const detector = new YoloDetectionInference({
 *   model: { onnx: modelBuffer, classNames: ["person", "car"] },
 *   thresholds: { confidence: 0.5 },
 * });
 *
 * await detector.init();
 * const detections = await detector.detect(imageBuffer);
 * await detector.destroy();
 * ```
 */
export class YoloDetectionInference extends BaseYoloDetectionInference {
  constructor(options: YoloDetectionOptions) {
    super(options, new NodePlatformProvider());
  }

  /**
   * Convert an ArrayBuffer to a Canvas.
   * @param buffer - The input image as ArrayBuffer.
   * @returns A canvas containing the decoded image.
   */
  static async convertBufferToCanvas(buffer: ArrayBuffer): Promise<CoreCanvas> {
    return ImageProcessor.prepareCanvas(buffer) as Promise<CoreCanvas>;
  }
}
