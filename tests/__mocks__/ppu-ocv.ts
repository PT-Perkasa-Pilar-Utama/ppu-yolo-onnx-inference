// Mock for ppu-ocv
export class ImageProcessor {
  static async initRuntime(config?: any): Promise<void> {
    console.log('Mock ImageProcessor.initRuntime called');
  }

  static prepareCanvas(canvasId: string): HTMLCanvasElement {
    console.log(`Mock ImageProcessor.prepareCanvas called with canvasId: ${canvasId}`);
    // Return a mock canvas element
    const mockCanvas = {
      getContext: (contextId: string) => {
        console.log(`Mock canvas.getContext called with contextId: ${contextId}`);
        return {
          drawImage: () => {},
          getImageData: () => ({ data: new Uint8ClampedArray(100) }),
          putImageData: () => {},
          // Add other context methods if needed by the code under test
        };
      },
      width: 300,
      height: 150,
      // Add other canvas properties if needed
    } as unknown as HTMLCanvasElement; // Use type assertion for complex mocks
    return mockCanvas;
  }

  async resize(width: number, height: number): Promise<void> {
    console.log(`Mock ImageProcessor.prototype.resize called with width: ${width}, height: ${height}`);
  }

  async toCanvas(canvasId: string): Promise<void> {
    console.log(`Mock ImageProcessor.prototype.toCanvas called with canvasId: ${canvasId}`);
  }

  destroy(): void {
    console.log('Mock ImageProcessor.prototype.destroy called');
  }

  // Add any other methods or properties of ImageProcessor that are used
}

export function createCanvas(width: number, height: number): HTMLCanvasElement {
  console.log(`Mock createCanvas called with width: ${width}, height: ${height}`);
  const mockCanvas = {
    getContext: (contextId: string) => {
      console.log(`Mock canvas.getContext called with contextId: ${contextId}`);
      return {
        drawImage: () => {},
        getImageData: () => ({ data: new Uint8ClampedArray(width * height * 4) }),
        putImageData: () => {},
      };
    },
    width: width,
    height: height,
    toDataURL: () => 'data:image/png;base64,mock', // Example data URL
  } as unknown as HTMLCanvasElement;
  return mockCanvas;
}

export class CanvasToolkit {
  private static instance: CanvasToolkit;

  private constructor() {
    // private constructor to prevent direct instantiation
  }

  public static getInstance(): CanvasToolkit {
    if (!CanvasToolkit.instance) {
      CanvasToolkit.instance = new CanvasToolkit();
      console.log('Mock CanvasToolkit.getInstance called - new instance created');
    } else {
      console.log('Mock CanvasToolkit.getInstance called - existing instance returned');
    }
    return CanvasToolkit.instance;
  }

  // Add any methods of CanvasToolkit that are used
  // For example:
  // public someMethod(): void {
  //   console.log('Mock CanvasToolkit.someMethod called');
  // }
}

// If there are other exports from ppu-ocv that are used, mock them here too.
// For example:
// export const anotherFunction = () => { /* ... */ };
// export class AnotherClass { /* ... */ }
// For now, we have ImageProcessor, createCanvas, and CanvasToolkit as per the requirement.
