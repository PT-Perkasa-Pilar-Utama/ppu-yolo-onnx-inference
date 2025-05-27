import * as path from "path";

/**
 * Utility class for handling path and file operations
 */
export class PathManager {
  /**
   * Resolves a path to an absolute path, handling both relative and absolute paths
   */
  static resolvePath(inputPath: string): string {
    if (path.isAbsolute(inputPath)) {
      return inputPath;
    }

    return path.resolve(process.cwd(), inputPath);
  }

  /**
   * Validates that a file exists at the given path using Node.js fs API
   */
  static async validateFilePath(filePath: string): Promise<void> {
    try {
      const fs = await import("fs");
      const stats = await fs.promises.stat(filePath);

      if (!stats.isFile()) {
        throw new Error(`Path exists but is not a file: ${filePath}`);
      }

      if (stats.size === 0) {
        console.warn(`Warning: File ${filePath} is empty`);
      }
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") {
        throw new Error(`File not found: ${filePath}`);
      }
      throw new Error(`Failed to validate file path ${filePath}: ${error}`);
    }
  }

  /**
   * Creates directory if it doesn't exist
   */
  static async ensureDirectoryExists(dirPath: string): Promise<void> {
    try {
      const fs = await import("fs");
      await fs.promises.mkdir(dirPath, { recursive: true });
    } catch (error) {
      throw new Error(`Failed to create directory ${dirPath}: ${error}`);
    }
  }
}
