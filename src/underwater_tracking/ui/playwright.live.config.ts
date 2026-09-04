import { defineConfig, devices } from "@playwright/test";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const configDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(configDir, "../../..");
const port = Number(process.env.PLAYWRIGHT_PORT ?? 8000);
const externalBaseURL = process.env.PLAYWRIGHT_BASE_URL;
const baseURL = externalBaseURL ?? `http://127.0.0.1:${port}`;
const executablePath = process.env.PLAYWRIGHT_EXECUTABLE_PATH;
const pythonExecutable = process.env.PYTHON_EXECUTABLE ?? "python";
const quote = (value: string): string => `"${value.replaceAll("\"", "\\\"")}"`;

const backendCommand = [
  quote(pythonExecutable),
  quote(resolve(repoRoot, "main.py")),
  "--config",
  quote(resolve(repoRoot, "configs", "scenario", "uuv_only_single_target.yaml")),
  "--steps",
  "0",
  "--seed",
  "20260904",
  "--host",
  "127.0.0.1",
  "--port",
  String(port),
  "--ui-dist",
  quote(resolve(configDir, "dist")),
  "--output-root",
  quote(resolve(configDir, "test-results", "live-server-outputs")),
  "--verification-audit",
  "--acceptance-fixture",
].join(" ");

export default defineConfig({
  testDir: "e2e",
  testMatch: "three-uuv-tracking-modes.spec.ts",
  timeout: 5 * 60 * 1000,
  fullyParallel: false,
  workers: 1,
  reporter: [["list"], ["html", { open: "never" }]],
  use: {
    baseURL,
    trace: "retain-on-failure",
    screenshot: "off",
    video: "on",
    launchOptions: executablePath
      ? { executablePath }
      : undefined,
  },
  projects: [
    {
      name: "desktop",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 1440, height: 900 },
      },
    },
    {
      name: "compact-desktop",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 1280, height: 720 },
      },
    },
    {
      name: "mobile",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 390, height: 844 },
      },
    },
  ],
  webServer: externalBaseURL
    ? undefined
    : {
        command: backendCommand,
        url: baseURL,
        reuseExistingServer: false,
        timeout: 120_000,
        stdout: "pipe",
        stderr: "pipe",
      },
});
