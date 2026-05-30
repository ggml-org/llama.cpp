import { dlopen, ptr, toBuffer } from "bun:ffi";

const libPath = import.meta.dirname + "/bin/libjsllama.so";

const libs = dlopen(libPath, {
  jsllama_backend_init: {
    args: [],
    returns: "void",
  },
  jsllama_backend_free: {
    args: [],
    returns: "void",
  },
  jsllama_load_model: {
    args: ["cstring", "i32", "i32", "i32", "i32"],
    returns: "ptr",
  },
  jsllama_free_model: {
    args: ["ptr"],
    returns: "void",
  },
  jsllama_tokenize: {
    args: ["ptr", "cstring", "i32", "ptr", "i32", "i32", "i32"],
    returns: "i32",
  },
  jsllama_decode: {
    args: ["ptr", "buffer", "i32", "i32"],
    returns: "i32",
  },
  jsllama_decode_single: {
    args: ["ptr", "i32", "i32"],
    returns: "i32",
  },
  jsllama_sample: {
    args: ["ptr"],
    returns: "i32",
  },
  jsllama_is_eog: {
    args: ["ptr", "i32"],
    returns: "i32",
  },
  jsllama_token_to_piece: {
    args: ["ptr", "i32", "ptr", "i32"],
    returns: "i32",
  },
  jsllama_kv_cache_clear: {
    args: ["ptr"],
    returns: "void",
  },
  jsllama_get_vocab_size: {
    args: ["ptr"],
    returns: "i32",
  },
  jsllama_get_logits: {
    args: ["ptr", "i32"],
    returns: "ptr",
  },
});

export const jsllama = libs.symbols;

export function initBackend() {
  jsllama.jsllama_backend_init();
}

export function freeBackend() {
  jsllama.jsllama_backend_free();
}

export function loadModel(path, nCtx, nBatch, nThreads, nGpuLayers) {
  return jsllama.jsllama_load_model(path, nCtx, nBatch, nThreads, nGpuLayers);
}

export function freeModel(handle) {
  jsllama.jsllama_free_model(handle);
}

export function tokenize(handle, text, addSpecial, parseSpecial) {
  const maxTokens = text.length + 256;
  const tokens = new Int32Array(maxTokens);
  const tokensPtr = ptr(tokens);
  const textBuf = new TextEncoder().encode(text);
  
  const result = jsllama.jsllama_tokenize(
    handle,
    textBuf,
    textBuf.length,
    tokensPtr,
    maxTokens,
    addSpecial ? 1 : 0,
    parseSpecial ? 1 : 0
  );
  
  if (result < 0) {
    const needed = -result;
    const newTokens = new Int32Array(needed);
    const newTokensPtr = ptr(newTokens);
    const result2 = jsllama.jsllama_tokenize(
      handle,
      textBuf,
      textBuf.length,
      newTokensPtr,
      needed,
      addSpecial ? 1 : 0,
      parseSpecial ? 1 : 0
    );
    if (result2 < 0) {
      throw new Error("Tokenization failed");
    }
    const finalTokens = new Int32Array(result2);
    finalTokens.set(newTokens.slice(0, result2));
    return finalTokens;
  }
  
  const finalTokens = new Int32Array(result);
  finalTokens.set(tokens.slice(0, result));
  return finalTokens;
}

export function decode(handle, tokens, nPast) {
  return jsllama.jsllama_decode(handle, tokens, tokens.length, nPast);
}

export function decodeSingle(handle, token, nPast) {
  return jsllama.jsllama_decode_single(handle, token, nPast);
}

export function sample(handle) {
  return jsllama.jsllama_sample(handle);
}

export function isEogToken(handle, token) {
  return jsllama.jsllama_is_eog(handle, token) !== 0;
}

export function tokenToPiece(handle, token) {
  const buf = new Int8Array(32);
  const bufPtr = ptr(buf);
  const result = jsllama.jsllama_token_to_piece(handle, token, bufPtr, 32);
  
  if (result < 0) {
    const needed = -result;
    const newBuf = new Int8Array(needed);
    const newBufPtr = ptr(newBuf);
    const result2 = jsllama.jsllama_token_to_piece(handle, token, newBufPtr, needed);
    if (result2 < 0) {
      return "";
    }
    return new TextDecoder().decode(newBuf.slice(0, result2));
  }
  
  return new TextDecoder().decode(buf.slice(0, result));
}

export function kvCacheClear(handle) {
  jsllama.jsllama_kv_cache_clear(handle);
}

export function getVocabSize(handle) {
  return jsllama.jsllama_get_vocab_size(handle);
}

export function getLogits(handle, idx) {
  const logitsPtr = jsllama.jsllama_get_logits(handle, idx);
  const vocabSize = getVocabSize(handle);
  const logitsBuf = toBuffer(logitsPtr, vocabSize * 4);
  return new Float32Array(logitsBuf);
}
