import { createHash } from "node:crypto";

if (!("Bun" in globalThis)) {
  class CryptoHasher {
    private readonly hash;

    constructor(algorithm: string) {
      this.hash = createHash(algorithm);
    }

    update(data: string | ArrayBuffer | ArrayBufferView) {
      if (typeof data === "string") {
        this.hash.update(data);
      } else if (ArrayBuffer.isView(data)) {
        this.hash.update(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
      } else {
        this.hash.update(new Uint8Array(data));
      }
      return this;
    }

    digest(encoding: "hex" | "base64" = "hex") {
      return this.hash.digest(encoding);
    }
  }

  Object.defineProperty(globalThis, "Bun", {
    value: { CryptoHasher },
    configurable: true,
  });
}
