import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const sourceDir = dirname(fileURLToPath(import.meta.url));

describe("production data boundary", () => {
  it("does not expose a mock live or replay source", () => {
    const appSource = readFileSync(resolve(sourceDir, "App.tsx"), "utf8");

    expect(appSource).not.toMatch(/mockEnabled|useMock|VITE_MOCK|mock:\/\//);
    expect(existsSync(resolve(sourceDir, "mock"))).toBe(false);
  });
});
