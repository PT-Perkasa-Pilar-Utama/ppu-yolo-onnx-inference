import type { InferenceSession, Tensor } from "onnxruntime-common";

/**
 * Minimal canvas interface compatible with both `@napi-rs/canvas` (Node)
 * and `HTMLCanvasElement` / `OffscreenCanvas` (Browser).
 */
export interface CoreCanvas {
  /** Canvas width in pixels. */
  width: number;
  /** Canvas height in pixels. */
  height: number;
}

/**
 * Platform abstraction for running YOLO inference across Node.js and browser.
 *
 * Platform-specific modules (`onnxruntime-node`/`onnxruntime-web`,
 * `ppu-ocv`/`ppu-ocv/web`) are injected through this interface,
 * keeping all business logic platform-agnostic.
 */
export interface PlatformProvider {
  /** Create an ONNX inference session from a model buffer. */
  createInferenceSession(model: ArrayBuffer): Promise<InferenceSession>;

  /** Create a typed tensor for model input. */
  createTensor(
    type: Tensor.Type,
    data: Float32Array,
    dims: readonly number[],
  ): Tensor;

  /** Create a blank canvas with the given dimensions. */
  createCanvas(width: number, height: number): CoreCanvas;

  /** Check whether a value is a platform canvas. */
  isCanvasLike(value: unknown): value is CoreCanvas;

  /** Decode an image buffer into a canvas. */
  prepareCanvas(buffer: ArrayBuffer): Promise<CoreCanvas>;

  /**
   * Resize a canvas to the given dimensions, returning a new canvas.
   * Implementations must handle internal resource cleanup.
   */
  resizeImage(source: CoreCanvas, width: number, height: number): CoreCanvas;

  /**
   * Get a 2D rendering context from a canvas.
   * Implementations should pass `{ willReadFrequently: true }` where appropriate.
   */
  getContext2D(canvas: CoreCanvas): CanvasRenderingContext2D;

  /** Save a canvas as a debug image file (no-op in browser). */
  saveDebugImage(
    canvas: CoreCanvas,
    filename: string,
    outputDir: string,
  ): Promise<void>;

  /** Initialize the image processing runtime (OpenCV). */
  initRuntime(): Promise<void>;

  /** Yield the event loop (`setImmediate` on Node, `setTimeout` in browser). */
  scheduleYield(): Promise<void>;
}
