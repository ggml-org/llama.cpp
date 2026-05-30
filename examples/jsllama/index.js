import { initBackend, freeBackend, loadModel, freeModel, tokenize, decode, decodeSingle, sample, isEogToken, tokenToPiece, kvCacheClear, getVocabSize, getLogits } from "./dll.js";

export class Llama {
  #handle = null;
  #initialized = false;
  #nCtx = 2048;
  #nBatch = 2048;
  #nThreads = 4;
  #nGpuLayers = -1;
  static #staticInitialized = false;

  static init() {
    if (!this.#staticInitialized) {
      initBackend();
      this.#staticInitialized = true;
    }
  }

  static free() {
    if (this.#staticInitialized) {
      freeBackend();
      this.#staticInitialized = false;
    }
  }

  constructor(options = {}) {
    this.#nCtx = options.nCtx || 2048;
    this.#nBatch = options.nBatch || 2048;
    this.#nThreads = options.nThreads || 4;
    this.#nGpuLayers = options.nGpuLayers ?? -1;
  }

  loadModel(modelPath) {
    Llama.init();

    this.#handle = loadModel(new TextEncoder().encode(modelPath), this.#nCtx, this.#nBatch, this.#nThreads, this.#nGpuLayers);
    
    if (!this.#handle || this.#handle === 0n) {
      throw new Error(`Failed to load model: ${modelPath}`);
    }

    this.#initialized = true;
    return true;
  }

  unloadModel() {
    if (this.#handle) {
      freeModel(this.#handle);
      this.#handle = null;
    }
    this.#initialized = false;
  }

  *generate(prompt, options = {}) {
    if (!this.#initialized) {
      throw new Error("Model not loaded");
    }

    const maxTokens = options.maxTokens || 256;

    kvCacheClear(this.#handle);

    const tokens = tokenize(this.#handle, prompt, true, true);
    
    let nPast = 0;
    let nConsumed = 0;
    let generated = 0;

    try {
      while (generated < maxTokens) {
        if (nConsumed < tokens.length) {
          const nTake = Math.min(this.#nBatch, tokens.length - nConsumed);
          const promptTokens = new Int32Array(nTake);
          for (let i = 0; i < nTake; i++) {
            promptTokens[i] = tokens[nConsumed + i];
          }
          
          const ret = decode(this.#handle, promptTokens, nPast);
          if (ret !== 0) {
            throw new Error(`Decode failed with code ${ret}`);
          }
          
          nPast += nTake;
          nConsumed += nTake;
        } else {
          const nextToken = sample(this.#handle);
          
          if (isEogToken(this.#handle, nextToken)) {
            break;
          }
          
          const piece = tokenToPiece(this.#handle, nextToken);
          generated++;
          yield piece;
          
          const ret = decodeSingle(this.#handle, nextToken, nPast);
          if (ret !== 0) {
            throw new Error(`Decode failed with code ${ret}`);
          }
          
          nPast++;
        }
      }
    } catch (error) {
      console.error("Generation error:", error);
    }
  }

  async generateStream(prompt, callback, options = {}) {
    for (const token of this.generate(prompt, options)) {
      callback(token);
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }

  get isLoaded() {
    return this.#initialized;
  }
}

export default Llama;
