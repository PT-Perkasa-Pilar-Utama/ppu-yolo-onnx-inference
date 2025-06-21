import { YoloDetectionInference } from "../src/yolo-inference";
import { COCO128_CLASS } from "./coco128/coco128-class";
// import { YoloDetectionInference } from "ppu-yolo-onnx-inference";

async function main() {
  const imageFile = Bun.file("./examples/coco128/person.png");
  const imageBuffer = await imageFile.arrayBuffer();

  const modelFile = Bun.file("./examples/coco128/coco128.onnx");
  const modelBuffer = await modelFile.arrayBuffer();

  try {
    let start = Date.now();
    const model = new YoloDetectionInference({
      model: {
        onnx: modelBuffer,
        classNames: COCO128_CLASS,
      },
      thresholds: {
        confidence: 0.5,
      },
    });

    await model.init();
    console.log("Initialized in ", Date.now() - start, "ms");

    // You can also use Canvas, there's static method to convert buffer to canvas
    // YoloDetectionInference.convertBufferToCanvas()

    start = Date.now();
    const detections = await model.detect(imageBuffer);

    // clean up session if not used anymore
    await model.destroy();

    console.log("Detections:", detections);
    console.log("Detection time:", Date.now() - start, "ms");
  } catch (error) {
    console.error("Error in main execution:", error);
  }
}

main();
