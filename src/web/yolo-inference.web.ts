import { ImageProcessor } from "ppu-ocv/web";
import { BaseYoloDetectionInference } from "../core/base-yolo-inference.js";
import type { CoreCanvas } from "../core/platform.js";
import type { YoloDetectionOptions } from "../interface.js";
import { WebPlatformProvider } from "./platform.web.js";

/**
 * YOLOv11 Object Detection Inference Engine for the browser.
 *
 * @example
 * ```ts
 * import { YoloDetectionInference } from "ppu-yolo-onnx-inference/web";
 *
 * const detector = new YoloDetectionInference({
 *   model: { onnx: modelArrayBuffer, classNames: ["person", "car"] },
 *   thresholds: { confidence: 0.5 },
 * });
 *
 * await detector.init();
 * const detections = await detector.detect(imageArrayBuffer);
 * await detector.destroy();
 * ```
 */
export class YoloDetectionInference extends BaseYoloDetectionInference {
  constructor(options: YoloDetectionOptions) {
    super(options, new WebPlatformProvider());
  }

  /**
   * Convert an ArrayBuffer to a canvas element.
   * @param buffer - The input image as ArrayBuffer.
   * @returns A canvas containing the decoded image.
   */
  static async convertBufferToCanvas(buffer: ArrayBuffer): Promise<CoreCanvas> {
    return ImageProcessor.prepareCanvas(buffer) as Promise<CoreCanvas>;
  }
}
