import { describe, expect, test } from "bun:test";

// ── 1. Node module exports ───────────────────────────────────────────
describe("Node module exports", () => {
  test("src/index.ts exports YoloDetectionInference", async () => {
    const mod = await import("../src/index.js");
    expect(mod.YoloDetectionInference).toBeDefined();
    expect(typeof mod.YoloDetectionInference).toBe("function");
  });

  test("src/index.ts exports shared constants", async () => {
    const mod = await import("../src/index.js");
    expect(mod.DEFAULT_THRESHOLDS).toBeDefined();
    expect(mod.DEFAULT_DEBUG_OPTIONS).toBeDefined();
    expect(mod.STANDARD_MODEL_INPUT_SHAPE).toBeDefined();
  });

  test("DEFAULT_THRESHOLDS has expected defaults", async () => {
    const mod = await import("../src/index.js");
    expect(mod.DEFAULT_THRESHOLDS).toEqual({
      confidence: 0.75,
      iou: 0.5,
      classConfidence: 0.2,
    });
  });

  test("STANDARD_MODEL_INPUT_SHAPE is [640, 640]", async () => {
    const mod = await import("../src/index.js");
    expect(mod.STANDARD_MODEL_INPUT_SHAPE).toEqual([640, 640]);
  });
});

// ── 2. Instantiation & lifecycle ─────────────────────────────────────
describe("Node service instantiation", () => {
  test("can create instance with default options", async () => {
    const { YoloDetectionInference } = await import("../src/index.js");
    const service = new YoloDetectionInference({
      model: { onnx: new ArrayBuffer(10), classNames: ["test"] },
    });
    expect(service).toBeDefined();
    expect(service.isInitialized()).toBe(false);
  });

  test("can create instance with custom thresholds", async () => {
    const { YoloDetectionInference } = await import("../src/index.js");
    const service = new YoloDetectionInference({
      model: { onnx: new ArrayBuffer(10), classNames: ["a", "b", "c"] },
      thresholds: { confidence: 0.9, iou: 0.3, classConfidence: 0.5 },
      debug: { verbose: false, debug: false, debugFolder: "./test-out" },
    });
    expect(service).toBeDefined();
    expect(service.isInitialized()).toBe(false);
  });

  test("detect returns empty array when not initialized", async () => {
    const { YoloDetectionInference } = await import("../src/index.js");
    const service = new YoloDetectionInference({
      model: { onnx: new ArrayBuffer(10), classNames: ["test"] },
    });
    const result = await service.detect(new ArrayBuffer(10));
    expect(result).toEqual([]);
  });

  test("destroy is safe to call without initialization", async () => {
    const { YoloDetectionInference } = await import("../src/index.js");
    const service = new YoloDetectionInference({
      model: { onnx: new ArrayBuffer(10), classNames: ["test"] },
    });
    await service.destroy();
    expect(service.isInitialized()).toBe(false);
  });
});

// ── 3. Node services use correct platform modules ────────────────────
describe("Node services use correct platform modules", () => {
  test("platform.node.ts uses onnxruntime-node", async () => {
    const content = await Bun.file("./src/processor/platform.node.ts").text();
    expect(content).toContain("onnxruntime-node");
    expect(content).not.toContain("onnxruntime-web");
  });

  test("platform.node.ts uses ppu-ocv (not ppu-ocv/web)", async () => {
    const content = await Bun.file("./src/processor/platform.node.ts").text();
    expect(content).toContain('from "ppu-ocv"');
    expect(content).not.toContain("ppu-ocv/web");
  });

  test("node service extends base class", async () => {
    const content = await Bun.file(
      "./src/processor/yolo-inference.service.ts",
    ).text();
    expect(content).toContain("BaseYoloDetectionInference");
    expect(content).toContain("NodePlatformProvider");
  });
});

// ── 4. Architecture — single source of truth ─────────────────────────
describe("Architecture", () => {
  test("base class exists in core directory", async () => {
    const content = await Bun.file(
      "./src/core/base-yolo-inference.ts",
    ).text();
    expect(content).toContain("class BaseYoloDetectionInference");
    expect(content).toContain("PlatformProvider");
  });

  test("base class does not import any platform-specific module", async () => {
    const content = await Bun.file(
      "./src/core/base-yolo-inference.ts",
    ).text();
    expect(content).not.toContain("onnxruntime-node");
    expect(content).not.toContain("onnxruntime-web");
    expect(content).not.toContain('from "ppu-ocv"');
    expect(content).not.toContain("ppu-ocv/web");
    expect(content).not.toContain('from "fs"');
    expect(content).not.toContain('from "path"');
  });

  test("platform interface exists in core directory", async () => {
    const content = await Bun.file("./src/core/platform.ts").text();
    expect(content).toContain("interface PlatformProvider");
  });
});
