import type { InferenceSession, Tensor } from "onnxruntime-common";
import {
  DEFAULT_DEBUG_OPTIONS,
  DEFAULT_THRESHOLDS,
  STANDARD_MODEL_INPUT_SHAPE,
} from "../constant.js";
import type {
  Box,
  DebuggingOptions,
  DetectedObject,
  ModelMetadata,
  ModelThresholds,
  PreprocessYoloResult,
  TensorValueMetadata,
  YoloDetectionOptions,
} from "../interface.js";
import type { CoreCanvas, PlatformProvider } from "./platform.js";

/**
 * Platform-agnostic YOLOv11 Object Detection Inference Engine.
 *
 * Contains all business logic for image preprocessing, model inference,
 * and NMS post-processing. Platform-specific behaviour (canvas creation,
 * ONNX runtime, file I/O) is delegated to the injected {@link PlatformProvider}.
 */
export class BaseYoloDetectionInference {
  private readonly model: ArrayBuffer;
  /** Class names for object detection labels. */
  protected readonly classNames: string[];
  /** Merged threshold configuration. */
  protected readonly thresholds: Required<ModelThresholds>;
  /** Merged debugging configuration. */
  protected readonly debugging: Required<DebuggingOptions>;
  /** Platform abstraction layer. */
  protected readonly platform: PlatformProvider;

  private modelMetadata: ModelMetadata | null = null;
  private session: InferenceSession | null = null;
  private static readonly CHANNELS = 3;

  constructor(options: YoloDetectionOptions, platform: PlatformProvider) {
    this.model = options.model.onnx;
    this.classNames = options.model.classNames;
    this.thresholds = { ...DEFAULT_THRESHOLDS, ...options.thresholds };
    this.debugging = { ...DEFAULT_DEBUG_OPTIONS, ...options.debug };
    this.modelMetadata = options.modelMetadata || null;
    this.platform = platform;
  }

  /**
   * Whether the model session has been initialised.
   * @returns `true` after a successful {@link init} call.
   */
  public isInitialized(): boolean {
    return this.session !== null;
  }

  /**
   * Initialize the YOLO model and prepare for inference.
   */
  public async init(): Promise<void> {
    if (!this.model) {
      throw new Error("Model ArrayBuffer is required");
    }

    try {
      this.log("init", "Loading model");
      this.session = await this.platform.createInferenceSession(this.model);
      await this.platform.scheduleYield();

      this.log(
        "init",
        `Model loaded successfully\n\tinputNames: ${
          this.session.inputNames
        }\n\toutputNames: ${this.session.outputNames}`,
      );

      if (!this.modelMetadata) {
        let inputShape = STANDARD_MODEL_INPUT_SHAPE;
        try {
          /* inputMetadata may not exist on every ORT backend */
          const meta = (this.session as unknown as Record<string, unknown[]>)
            .inputMetadata;
          if (Array.isArray(meta) && meta.length > 0) {
            const shape = (meta[0] as TensorValueMetadata).shape.slice(2) as [
              number,
              number,
            ];
            if (shape.length === 2) inputShape = shape;
          }
        } catch {
          /* fall back to standard shape */
        }

        this.modelMetadata = {
          inputTensorName: this.session.inputNames[0],
          outputTensorName: this.session.outputNames[0],
          inputShape,
        };
      } else {
        this.modelMetadata = {
          inputTensorName:
            this.modelMetadata.inputTensorName || this.session.inputNames[0],
          outputTensorName:
            this.modelMetadata.outputTensorName || this.session.outputNames[0],
          inputShape:
            this.modelMetadata.inputShape || STANDARD_MODEL_INPUT_SHAPE,
        };
      }

      await this.platform.initRuntime();
      this.log("init", "ImageProcessor runtime initialized");
    } catch (error) {
      throw new Error(`Failed to load model: ${error}`);
    }
  }

  /**
   * Detect objects in an image.
   * @param image - The input image as ArrayBuffer or platform canvas.
   * @returns An array of detected objects with bounding boxes, class names, and confidence scores.
   * @throws Error if the model is not initialized or detection fails.
   */
  public async detect(
    image: ArrayBuffer | CoreCanvas,
  ): Promise<DetectedObject[]> {
    this.log("detect", "Starting object detection");

    try {
      const preprocessed = await this.preprocessImage(image);
      const output = await this.runInference(preprocessed);

      if (!output) return [];

      if (this.debugging.verbose) {
        this.debugTensorData(
          output.data as Float32Array,
          output.dims[2],
          output.dims[1],
        );
      }

      const detections = this.postprocessOutput(output, preprocessed);
      if (this.debugging.debug) {
        await this.saveDebugImages(preprocessed, image, detections);
      }

      this.log("detect", `Detected ${detections.length} objects\n`);
      return detections;
    } catch (error) {
      console.error("Detection failed:", error);
      return [];
    }
  }

  private async preprocessImage(
    image: ArrayBuffer | CoreCanvas,
  ): Promise<PreprocessYoloResult> {
    if (!this.modelMetadata) {
      throw new Error("Model not initialized. Call init() first");
    }

    const canvas = this.platform.isCanvasLike(image)
      ? image
      : await this.platform.prepareCanvas(image);

    const { width: originalWidth, height: originalHeight } = canvas;
    const [modelInputWidth, modelInputHeight] = this.modelMetadata.inputShape;

    const scaleRatio = Math.min(
      modelInputWidth / originalWidth,
      modelInputHeight / originalHeight,
    );
    const scaledWidth = Math.round(originalWidth * scaleRatio);
    const scaledHeight = Math.round(originalHeight * scaleRatio);

    const padX = (modelInputWidth - scaledWidth) / 2;
    const padY = (modelInputHeight - scaledHeight) / 2;

    const scaledCanvas = this.platform.createCanvas(
      modelInputWidth,
      modelInputHeight,
    );
    const ctx = this.platform.getContext2D(scaledCanvas);

    ctx.fillStyle = "rgb(114, 114, 114)";
    ctx.fillRect(0, 0, modelInputWidth, modelInputHeight);

    const resized = this.platform.resizeImage(
      canvas,
      scaledWidth,
      scaledHeight,
    );

    ctx.drawImage(
      resized as unknown as CanvasImageSource,
      padX,
      padY,
      scaledWidth,
      scaledHeight,
    );

    const tensor = this.canvasToTensor(
      scaledCanvas,
      modelInputWidth,
      modelInputHeight,
    );

    this.log(
      "preprocessImage",
      `Preprocessed: ${originalWidth}x${originalHeight} → ${modelInputWidth}x${modelInputHeight} (ratio: ${scaleRatio.toFixed(
        3,
      )}, padX: ${padX.toFixed(1)}, padY: ${padY.toFixed(1)})`,
    );

    return {
      tensor,
      modelInputWidth,
      modelInputHeight,
      originalWidth,
      originalHeight,
      scaleRatio,
      padX,
      padY,
    };
  }

  private canvasToTensor(
    canvas: CoreCanvas,
    width: number,
    height: number,
  ): Float32Array {
    const tensor = new Float32Array(
      BaseYoloDetectionInference.CHANNELS * height * width,
    );
    const ctx = this.platform.getContext2D(canvas);
    const imageData = ctx.getImageData(0, 0, width, height).data;

    for (let h = 0; h < height; h++) {
      for (let w = 0; w < width; w++) {
        const pixelIndex = h * width + w;
        const rgbaIndex = pixelIndex * 4;

        tensor[pixelIndex] = imageData[rgbaIndex] / 255.0;
        tensor[height * width + pixelIndex] = imageData[rgbaIndex + 1] / 255.0;
        tensor[2 * height * width + pixelIndex] =
          imageData[rgbaIndex + 2] / 255.0;
      }
    }

    return tensor;
  }

  private async runInference(
    preprocessed: PreprocessYoloResult,
  ): Promise<Tensor | null> {
    if (!this.session || !this.modelMetadata) {
      throw new Error("Model not initialized. Call init() first");
    }

    try {
      const inputTensor = this.platform.createTensor(
        "float32",
        preprocessed.tensor,
        [1, 3, preprocessed.modelInputHeight, preprocessed.modelInputWidth],
      );

      const feeds = {
        [this.modelMetadata.inputTensorName]: inputTensor,
      } as unknown as InferenceSession.FeedsType;
      const results = await this.session.run(feeds);

      return (results[this.modelMetadata.outputTensorName] as Tensor) || null;
    } catch (error) {
      console.error("Inference error:", error);
      throw error;
    }
  }

  private postprocessOutput(
    tensor: Tensor,
    preprocessed: PreprocessYoloResult,
  ): DetectedObject[] {
    const data = tensor.data as Float32Array;
    const [, numParams, numPredictions] = tensor.dims;

    this.log(
      "postprocessOutput",
      `Post-processing output: numParams=${numParams}, numPredictions=${numPredictions}`,
    );

    if (numParams < 4) {
      console.error(
        `Invalid tensor shape: expected ≥4 parameters per box, got ${numParams}`,
      );
      return [];
    }

    const candidates = this.extractCandidates(
      data,
      numPredictions,
      numParams,
      preprocessed,
    );
    const nmsIndices = this.applyNMS(candidates);

    return this.scaleCandidates(candidates, nmsIndices, preprocessed);
  }

  private extractCandidates(
    data: Float32Array,
    numPredictions: number,
    numParams: number,
    _preprocessed: PreprocessYoloResult,
  ): Array<{ box: Box; score: number; classId: number }> {
    const candidates: Array<{ box: Box; score: number; classId: number }> = [];
    const numClasses = numParams - 4;
    const isSingleClass = numClasses <= 1;

    this.log(
      "extractCandidates",
      `YOLOv11 format: numClasses=${numClasses}, isSingleClass=${isSingleClass}`,
    );

    let debugCount = 0;
    let highConfidenceCount = 0;

    for (let i = 0; i < numPredictions; i++) {
      const cx = data[i];
      const cy = data[numPredictions + i];
      const w = data[2 * numPredictions + i];
      const h = data[3 * numPredictions + i];

      let finalConfidence: number;
      let bestClassId: number;

      if (isSingleClass) {
        finalConfidence = data[4 * numPredictions + i];
        bestClassId = 0;
      } else {
        let maxClassScore = -Infinity;
        bestClassId = 0;

        for (let c = 0; c < numClasses; c++) {
          const classScore = data[(4 + c) * numPredictions + i];
          if (classScore > maxClassScore) {
            maxClassScore = classScore;
            bestClassId = c;
          }
        }
        finalConfidence = maxClassScore;
      }

      if (finalConfidence > 0.1) highConfidenceCount++;

      if (finalConfidence > 0.1 && debugCount < 5) {
        this.log(
          "extractCandidates",
          `Debug candidate ${i}: confidence=${finalConfidence.toFixed(4)}, ` +
            `box=[${cx.toFixed(1)}, ${cy.toFixed(1)}, ${w.toFixed(
              1,
            )}, ${h.toFixed(1)}], ` +
            `classId=${bestClassId}`,
        );
        debugCount++;
      }

      if (finalConfidence < this.thresholds.confidence || w <= 0 || h <= 0)
        continue;

      const x = cx - w / 2;
      const y = cy - h / 2;

      if (!this.modelMetadata) continue;

      if (
        x < 0 ||
        y < 0 ||
        x + w > this.modelMetadata.inputShape[0] ||
        y + h > this.modelMetadata.inputShape[1]
      ) {
        continue;
      }

      candidates.push({
        box: { x, y, width: w, height: h },
        score: finalConfidence,
        classId: bestClassId,
      });
    }

    this.log(
      "extractCandidates",
      `Total candidates with confidence > 0.1: ${highConfidenceCount}\n[YOLO:extractCandidates] ` +
        `Final candidates after filtering: ${candidates.length}`,
    );

    if (candidates.length < 3 && this.thresholds.confidence > 0.1) {
      this.log(
        "extractCandidates",
        "Few candidates found, trying with lower threshold...",
      );

      return this.extractWithLowerThreshold(data, numPredictions, numParams);
    }

    return candidates;
  }

  private extractWithLowerThreshold(
    data: Float32Array,
    numPredictions: number,
    numParams: number,
  ): Array<{ box: Box; score: number; classId: number }> {
    const candidates: Array<{ box: Box; score: number; classId: number }> = [];
    const numClasses = numParams - 4;
    const isSingleClass = numClasses <= 1;
    const lowerThreshold = Math.max(0.05, this.thresholds.confidence * 0.5);

    this.log(
      "extractWithLowerThreshold",
      `Trying lower threshold: ${lowerThreshold}`,
    );

    for (let i = 0; i < numPredictions; i++) {
      const cx = data[i];
      const cy = data[numPredictions + i];
      const w = data[2 * numPredictions + i];
      const h = data[3 * numPredictions + i];

      let finalConfidence: number;
      let bestClassId: number;

      if (isSingleClass) {
        finalConfidence = data[4 * numPredictions + i];
        bestClassId = 0;
      } else {
        let maxClassScore = -Infinity;
        bestClassId = 0;

        for (let c = 0; c < numClasses; c++) {
          const classScore = data[(4 + c) * numPredictions + i];
          if (classScore > maxClassScore) {
            maxClassScore = classScore;
            bestClassId = c;
          }
        }

        finalConfidence = maxClassScore;
      }

      if (finalConfidence >= lowerThreshold && w > 0 && h > 0) {
        const x = cx - w / 2;
        const y = cy - h / 2;

        if (!this.modelMetadata) continue;

        if (
          x >= 0 &&
          y >= 0 &&
          x + w <= this.modelMetadata.inputShape[0] &&
          y + h <= this.modelMetadata.inputShape[1]
        ) {
          candidates.push({
            box: { x, y, width: w, height: h },
            score: finalConfidence,
            classId: bestClassId,
          });
        }
      }
    }

    this.log(
      "extractWithLowerThreshold",
      `Candidates with lower threshold: ${candidates.length}`,
    );
    return candidates;
  }

  private debugTensorData(
    data: Float32Array,
    numPredictions: number,
    numParams: number,
  ): void {
    this.log("debugTensorData", "\n=== TENSOR DEBUG ===");

    const confidences: Array<{ index: number; confidence: number }> = [];

    for (let i = 0; i < numPredictions; i++) {
      if (numParams === 5) {
        const conf = data[4 * numPredictions + i];
        confidences.push({ index: i, confidence: conf });
      } else {
        let maxConf = -Infinity;
        for (let c = 0; c < numParams - 4; c++) {
          const classScore = data[(4 + c) * numPredictions + i];
          maxConf = Math.max(maxConf, classScore);
        }
        confidences.push({ index: i, confidence: maxConf });
      }
    }

    confidences.sort((a, b) => b.confidence - a.confidence);

    this.log("debugTensorData", "Top 10 detections by confidence:");
    for (let j = 0; j < Math.min(10, confidences.length); j++) {
      const { index: i, confidence } = confidences[j];
      const cx = data[i];
      const cy = data[numPredictions + i];
      const w = data[2 * numPredictions + i];
      const h = data[3 * numPredictions + i];

      this.log(
        "debugTensorData",
        `${j + 1}. Index ${i}: conf=${confidence.toFixed(4)}, box=[${cx.toFixed(
          1,
        )}, ${cy.toFixed(1)}, ${w.toFixed(1)}, ${h.toFixed(1)}]`,
      );
    }

    const ranges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    const counts = new Array(ranges.length - 1).fill(0) as number[];

    confidences.forEach(({ confidence }) => {
      for (let r = 0; r < ranges.length - 1; r++) {
        if (confidence >= ranges[r] && confidence < ranges[r + 1]) {
          counts[r]++;
          break;
        }
      }
    });

    this.log("debugTensorData", "Confidence distribution:");
    for (let r = 0; r < counts.length; r++) {
      this.log(
        "debugTensorData",
        `${ranges[r].toFixed(1)}-${ranges[r + 1].toFixed(1)}: ${
          counts[r]
        } detections`,
      );
    }
  }

  private applyNMS(
    candidates: Array<{ box: Box; score: number; classId: number }>,
  ): number[] {
    const indices = candidates
      .map((_, i) => i)
      .sort((a, b) => candidates[b].score - candidates[a].score);

    const keep: number[] = [];

    while (indices.length > 0) {
      const current = indices.shift()!;
      keep.push(current);

      for (let i = indices.length - 1; i >= 0; i--) {
        const iou = this.calculateIoU(
          candidates[current].box,
          candidates[indices[i]].box,
        );
        if (iou > this.thresholds.iou) {
          indices.splice(i, 1);
        }
      }
    }

    return keep;
  }

  private scaleCandidates(
    candidates: Array<{ box: Box; score: number; classId: number }>,
    indices: number[],
    preprocessed: PreprocessYoloResult,
  ): DetectedObject[] {
    return indices
      .map((i) => {
        const candidate = candidates[i];
        const { scaleRatio, originalWidth, originalHeight, padX, padY } =
          preprocessed;

        const unpaddedX = candidate.box.x - padX;
        const unpaddedY = candidate.box.y - padY;

        const x = Math.max(0, Math.round(unpaddedX / scaleRatio));
        const y = Math.max(0, Math.round(unpaddedY / scaleRatio));

        const width = Math.min(
          originalWidth - x,
          Math.round(candidate.box.width / scaleRatio),
        );
        const height = Math.min(
          originalHeight - y,
          Math.round(candidate.box.height / scaleRatio),
        );

        if (width <= 0 || height <= 0) return null;

        return {
          box: { x, y, width, height },
          className:
            this.classNames[candidate.classId] || `class_${candidate.classId}`,
          classId: candidate.classId,
          confidence: candidate.score,
        };
      })
      .filter(Boolean) as DetectedObject[];
  }

  private calculateIoU(box1: Box, box2: Box): number {
    const x1 = Math.max(box1.x, box2.x);
    const y1 = Math.max(box1.y, box2.y);
    const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
    const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);

    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const union =
      box1.width * box1.height + box2.width * box2.height - intersection;

    return union > 0 ? intersection / union : 0;
  }

  private async saveDebugImages(
    preprocessed: PreprocessYoloResult,
    originalImage: ArrayBuffer | CoreCanvas,
    detections: DetectedObject[],
  ): Promise<void> {
    try {
      await this.savePreprocessedImage(preprocessed);
      await this.saveDetectionVisualization(originalImage, detections);
    } catch (error) {
      console.error("Debug image save failed:", error);
    }
  }

  private async savePreprocessedImage(
    preprocessed: PreprocessYoloResult,
  ): Promise<void> {
    const { modelInputWidth, modelInputHeight, tensor } = preprocessed;
    const canvas = this.platform.createCanvas(
      modelInputWidth,
      modelInputHeight,
    );
    const ctx = this.platform.getContext2D(canvas);
    const imageData = ctx.createImageData(modelInputWidth, modelInputHeight);

    for (let i = 0; i < modelInputWidth * modelInputHeight; i++) {
      const rgbaIndex = i * 4;
      imageData.data[rgbaIndex] = tensor[i] * 255;
      imageData.data[rgbaIndex + 1] =
        tensor[modelInputWidth * modelInputHeight + i] * 255;
      imageData.data[rgbaIndex + 2] =
        tensor[2 * modelInputWidth * modelInputHeight + i] * 255;
      imageData.data[rgbaIndex + 3] = 255;
    }

    ctx.putImageData(imageData, 0, 0);

    await this.platform.saveDebugImage(
      canvas,
      "yolo-preprocessed",
      this.debugging.debugFolder,
    );
  }

  private async saveDetectionVisualization(
    image: ArrayBuffer | CoreCanvas,
    detections: DetectedObject[],
  ): Promise<void> {
    const canvas = this.platform.isCanvasLike(image)
      ? image
      : await this.platform.prepareCanvas(image);

    const ctx = this.platform.getContext2D(canvas);

    detections.forEach((detection) => {
      const { x, y, width, height } = detection.box;

      ctx.strokeStyle = "yellow";
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, width, height);

      ctx.fillStyle = "yellow";
      ctx.font = "16px Arial";
      ctx.fillText(
        `${detection.className} ${(detection.confidence * 100).toFixed(1)}%`,
        x,
        y > 20 ? y - 5 : y + height + 15,
      );
    });

    await this.platform.saveDebugImage(
      canvas,
      "yolo-detections",
      this.debugging.debugFolder,
    );
  }

  /** Log a message when verbose debugging is enabled. */
  protected log(caller: string, message: string): void {
    if (this.debugging.verbose) {
      console.log(`[YOLO:${caller}] ${message}`);
    }
  }

  /**
   * Releases the ONNX runtime session and cleans up resources.
   * Safe to call even if the model was never initialised.
   */
  async destroy(): Promise<void> {
    this.log("destroy", "Cleaning up resources");

    if (this.session) {
      await this.session.release();
      this.session = null;
    }

    this.log("destroy", "Resources cleaned up");
  }
}
