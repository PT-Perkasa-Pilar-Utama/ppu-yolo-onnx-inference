import type { InferenceSession, Tensor } from "onnxruntime-common";
import * as ort from "onnxruntime-web";
import { ImageProcessor } from "ppu-ocv/web";
import type { CoreCanvas, PlatformProvider } from "../core/platform.js";

/* ───── Set WASM paths at module load time ───── */
if (typeof globalThis !== "undefined" && !ort.env.wasm.wasmPaths) {
  ort.env.wasm.wasmPaths =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/";
}

/**
 * Browser platform provider using `onnxruntime-web` and `ppu-ocv/web`.
 */
export class WebPlatformProvider implements PlatformProvider {
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
    if (typeof OffscreenCanvas !== "undefined") {
      return new OffscreenCanvas(width, height) as unknown as CoreCanvas;
    }
    const c = document.createElement("canvas");
    c.width = width;
    c.height = height;
    return c as unknown as CoreCanvas;
  }

  isCanvasLike(value: unknown): value is CoreCanvas {
    if (
      typeof HTMLCanvasElement !== "undefined" &&
      value instanceof HTMLCanvasElement
    )
      return true;
    if (
      typeof OffscreenCanvas !== "undefined" &&
      value instanceof OffscreenCanvas
    )
      return true;
    return false;
  }

  async prepareCanvas(buffer: ArrayBuffer): Promise<CoreCanvas> {
    return ImageProcessor.prepareCanvas(buffer) as Promise<CoreCanvas>;
  }

  resizeImage(source: CoreCanvas, width: number, height: number): CoreCanvas {
    const processor = new ImageProcessor(
      source as unknown as HTMLCanvasElement,
    );
    const resized = processor.resize({ width, height }).toCanvas();
    processor.destroy();
    return resized as unknown as CoreCanvas;
  }

  getContext2D(canvas: CoreCanvas): CanvasRenderingContext2D {
    const ctx = (canvas as unknown as HTMLCanvasElement).getContext("2d", {
      willReadFrequently: true,
    });
    if (!ctx) throw new Error("Failed to get 2D context from canvas");
    return ctx;
  }

  async saveDebugImage(
    _canvas: CoreCanvas,
    _filename: string,
    _outputDir: string,
  ): Promise<void> {
    /* no-op in browser — cannot write to disk */
  }

  async initRuntime(): Promise<void> {
    await ImageProcessor.initRuntime();
  }

  scheduleYield(): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, 0));
  }
}
