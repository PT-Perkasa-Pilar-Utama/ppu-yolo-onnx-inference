import type { InferenceSession, Tensor } from "onnxruntime-common";
import * as ort from "onnxruntime-node";
import type { CanvasLike } from "ppu-ocv";
import { CanvasToolkit, createCanvas, ImageProcessor } from "ppu-ocv";
import type { CoreCanvas, PlatformProvider } from "../core/platform.js";

/**
 * Node.js platform provider using `onnxruntime-node` and `ppu-ocv`.
 */
export class NodePlatformProvider implements PlatformProvider {
  createInferenceSession(model: ArrayBuffer): Promise<InferenceSession> {
    return ort.InferenceSession.create(
      model,
    ) as unknown as Promise<InferenceSession>;
  }

  createTensor(
    type: Tensor.Type,
    data: Float32Array,
    dims: readonly number[],
  ): Tensor {
    return new ort.Tensor(type, data, dims as number[]) as unknown as Tensor;
  }

  createCanvas(width: number, height: number): CoreCanvas {
    return createCanvas(width, height) as unknown as CoreCanvas;
  }

  isCanvasLike(value: unknown): value is CoreCanvas {
    return (
      value != null &&
      typeof value === "object" &&
      "width" in value &&
      "height" in value &&
      "getContext" in value &&
      typeof (value as Record<string, unknown>).getContext === "function"
    );
  }

  async prepareCanvas(buffer: ArrayBuffer): Promise<CoreCanvas> {
    return ImageProcessor.prepareCanvas(buffer) as Promise<CoreCanvas>;
  }

  resizeImage(source: CoreCanvas, width: number, height: number): CoreCanvas {
    const processor = new ImageProcessor(source as unknown as CanvasLike);
    const resized = processor.resize({ width, height }).toCanvas();
    processor.destroy();
    return resized as unknown as CoreCanvas;
  }

  getContext2D(canvas: CoreCanvas): CanvasRenderingContext2D {
    return (canvas as unknown as CanvasLike).getContext(
      "2d",
    ) as unknown as CanvasRenderingContext2D;
  }

  async saveDebugImage(
    canvas: CoreCanvas,
    filename: string,
    outputDir: string,
  ): Promise<void> {
    await CanvasToolkit.getInstance().saveImage({
      canvas: canvas as unknown as CanvasLike,
      filename,
      path: outputDir,
    });
  }

  async initRuntime(): Promise<void> {
    await ImageProcessor.initRuntime();
  }

  scheduleYield(): Promise<void> {
    return new Promise((resolve) => setImmediate(resolve));
  }
}
