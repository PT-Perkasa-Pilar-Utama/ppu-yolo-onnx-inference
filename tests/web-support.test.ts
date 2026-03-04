import { describe, expect, test } from "bun:test";

// ── 1. Web module exports ────────────────────────────────────────────
describe("Web module exports", () => {
  test("src/web/index.ts exports YoloDetectionInference", async () => {
    const mod = await import("../src/web/index.js");
    expect(mod.YoloDetectionInference).toBeDefined();
    expect(typeof mod.YoloDetectionInference).toBe("function");
  });

  test("src/web/index.ts exports shared constants", async () => {
    const mod = await import("../src/web/index.js");
    expect(mod.DEFAULT_THRESHOLDS).toBeDefined();
    expect(mod.DEFAULT_DEBUG_OPTIONS).toBeDefined();
    expect(mod.STANDARD_MODEL_INPUT_SHAPE).toBeDefined();
  });

  test("DEFAULT_THRESHOLDS has expected shape", async () => {
    const mod = await import("../src/web/index.js");
    const opts = mod.DEFAULT_THRESHOLDS;
    expect(opts).toHaveProperty("confidence");
    expect(opts).toHaveProperty("iou");
    expect(opts).toHaveProperty("classConfidence");
  });

  test("DEFAULT_DEBUG_OPTIONS has expected shape", async () => {
    const mod = await import("../src/web/index.js");
    const opts = mod.DEFAULT_DEBUG_OPTIONS;
    expect(opts).toHaveProperty("verbose");
    expect(opts).toHaveProperty("debug");
    expect(opts).toHaveProperty("debugFolder");
  });

  test("STANDARD_MODEL_INPUT_SHAPE is [640, 640]", async () => {
    const mod = await import("../src/web/index.js");
    expect(mod.STANDARD_MODEL_INPUT_SHAPE).toEqual([640, 640]);
  });
});

// ── 2. Instantiation & lifecycle ─────────────────────────────────────
describe("Web service instantiation", () => {
  test("can create instance with default options", async () => {
    const { YoloDetectionInference } = await import("../src/web/index.js");
    const service = new YoloDetectionInference({
      model: { onnx: new ArrayBuffer(10), classNames: ["test"] },
    });
    expect(service).toBeDefined();
    expect(service.isInitialized()).toBe(false);
  });

  test("can create instance with custom options", async () => {
    const { YoloDetectionInference } = await import("../src/web/index.js");
    const service = new YoloDetectionInference({
      model: { onnx: new ArrayBuffer(10), classNames: ["a", "b"] },
      thresholds: { confidence: 0.8, iou: 0.3 },
      debug: { verbose: true },
    });
    expect(service).toBeDefined();
  });

  test("detect returns empty array when not initialized", async () => {
    const { YoloDetectionInference } = await import("../src/web/index.js");
    const service = new YoloDetectionInference({
      model: { onnx: new ArrayBuffer(10), classNames: ["test"] },
    });
    const result = await service.detect(new ArrayBuffer(10));
    expect(result).toEqual([]);
  });

  test("destroy is safe to call without initialization", async () => {
    const { YoloDetectionInference } = await import("../src/web/index.js");
    const service = new YoloDetectionInference({
      model: { onnx: new ArrayBuffer(10), classNames: ["test"] },
    });
    await service.destroy(); // should not throw
    expect(service.isInitialized()).toBe(false);
  });
});

// ── 3. No Node-specific imports ──────────────────────────────────────
describe("Web services do not import Node-specific modules", () => {
  const WEB_FILES = [
    "./src/web/platform.web.ts",
    "./src/web/yolo-inference.web.ts",
    "./src/web/index.ts",
  ];

  for (const filePath of WEB_FILES) {
    test(`${filePath} has no Node imports`, async () => {
      const content = await Bun.file(filePath).text();
      expect(content).not.toContain('from "fs"');
      expect(content).not.toContain('from "path"');
      expect(content).not.toContain('from "os"');
      expect(content).not.toContain("onnxruntime-node");
    });
  }

  test("web platform uses onnxruntime-web", async () => {
    const content = await Bun.file("./src/web/platform.web.ts").text();
    expect(content).toContain("onnxruntime-web");
  });

  test("web platform uses ppu-ocv/web", async () => {
    const content = await Bun.file("./src/web/platform.web.ts").text();
    expect(content).toContain("ppu-ocv/web");
  });
});

// ── 4. Shared modules are reused (single source of truth) ────────────
describe("Shared modules are reused in web path", () => {
  test("web index re-exports shared interface types", async () => {
    const content = await Bun.file("./src/web/index.ts").text();
    expect(content).toContain("../interface.js");
    expect(content).toContain("../constant.js");
  });

  test("core base service is used by web service", async () => {
    const content = await Bun.file("./src/web/yolo-inference.web.ts").text();
    expect(content).toContain("../core/base-yolo-inference.js");
  });

  test("constants exports match between main and web", async () => {
    const mainMod = await import("../src/constant.js");
    const webMod = await import("../src/web/index.js");
    expect(webMod.DEFAULT_THRESHOLDS).toEqual(mainMod.DEFAULT_THRESHOLDS);
    expect(webMod.DEFAULT_DEBUG_OPTIONS).toEqual(mainMod.DEFAULT_DEBUG_OPTIONS);
    expect(webMod.STANDARD_MODEL_INPUT_SHAPE).toEqual(
      mainMod.STANDARD_MODEL_INPUT_SHAPE,
    );
  });

  test("core base service is also used by node service", async () => {
    const content = await Bun.file(
      "./src/processor/yolo-inference.service.ts",
    ).text();
    expect(content).toContain("../core/base-yolo-inference.js");
  });

  test("interface uses onnxruntime-common, not onnxruntime-node", async () => {
    const content = await Bun.file("./src/interface.ts").text();
    expect(content).toContain("onnxruntime-common");
    expect(content).not.toContain('from "onnxruntime-node"');
  });
});
