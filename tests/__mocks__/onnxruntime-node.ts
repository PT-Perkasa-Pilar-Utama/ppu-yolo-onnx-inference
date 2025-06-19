// Mock for onnxruntime-node
export class InferenceSession {
  static async create(modelPath: string, options?: any): Promise<InferenceSession> {
    console.log(`Mock InferenceSession.create called with modelPath: ${modelPath}`);
    return new InferenceSession();
  }

  async run(feeds: any, options?: any): Promise<any> {
    console.log('Mock InferenceSession.prototype.run called');
    // Simulate some output based on expected behavior if necessary
    return { output: { data: new Float32Array([1, 2, 3]) } };
  }

  async release(): Promise<void> {
    console.log('Mock InferenceSession.prototype.release called');
  }

  // Mock properties if needed, e.g., inputNames, outputNames
  get inputNames(): string[] {
    return ['input'];
  }

  get outputNames(): string[] {
    return ['output'];
  }
}

// If there are other exports from onnxruntime-node that are used, mock them here too.
// For example:
// export const OrtEnv = { /* ... mock OrtEnv properties and methods ... */ };
// export const Tensor = { /* ... mock Tensor properties and methods ... */ };
// export const InferenceSessionHandler = { /* ... mock InferenceSessionHandler ... */ };
// export const TrainingSession = { /* ... mock TrainingSession ... */ };
// export const ONNX = { /* ... mock ONNX ... */ };
// export const env = { /* ... mock env ... */ };
// ... and so on for any other named exports.
// For now, we only have InferenceSession as per the requirement.
