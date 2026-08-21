/**
 * Run this CLI on node.js for testing
 * hanging video mtmd in windows.
 *
 * `node mtmd-hang-test.ts --help` for help.
 *
 * exitcode 0: test success
 *
 * else: test failed
 */

import fs from "node:fs/promises"
import fsSync from "node:fs"
import { spawn } from "node:child_process"
import { resolve } from "node:path"
import { platform } from "node:os"
import crypto from "node:crypto"

async function exists(path: string) {
  try {
    await fs.access(path, fs.constants.R_OK)
    return true
  } catch {
    return false
  }
}

async function writeBlob(url: string, filename?: string) {
  if (filename == null) {
    filename = url.substring(url.lastIndexOf("/") + 1)
  }

  const response = await fetch(url)

  if (!response.ok || response.body == null) {
    throw new Error(`Response failed with code: ${response.status}`)
  }

  const stream = fsSync.createWriteStream(`${currentDir}/${filename}`)

  for await (const chunk of response.body) {
    stream.write(chunk)
  }

  return new Promise<void>((res) => {
    stream.end(() => {
      res()
    })
  })
}

async function createHash(path: string) {
  // Requires some memory
  const data = await fs.readFile(path, { encoding: null })
  return crypto.createHash("sha256").update(data).digest("hex")
}

function hasParamInCLI(...params: string[]) {
  for (const param of params) {
    if (process.argv.includes(param)) {
      return true
    }
  }
  return false
}

if (!import.meta.main) {
  console.error("Please execute this file manually.")
  process.exit(-1)
}

if (platform() !== "win32") {
  console.error("This regression test is only for windows!")
  process.exit(-1)
}

if (hasParamInCLI("-h", "--help")) {
  console.log(`
How to use:
node.exe mtmd-hang-test.ts [llama-cli Path]

Default llama-cli path is
'<workspace>/build-x64-windows-llvm-debug/bin/llama-cli.exe'

Additional options:
 -v, --verbose: Verbose logging (in llama-cli)
 -h, --help: Show help (this)

Result:
exitCode 0 -> success
exitCode 1 -> timeout or error
exitCode -1 -> Not supported environment
else -> Unknown error
`)
  process.exit(0)
}

let verbose = false
const currentDir = import.meta.dirname
const relativePath = "build-x64-windows-llvm-debug/bin/llama-cli.exe"
let cliPath = resolve(import.meta.dirname, `../../../../${relativePath}`)
if (process.argv[2] != null && !process.argv[2].startsWith("-")) {
  cliPath = resolve(process.cwd(), process.argv[2])
}

if (hasParamInCLI("-v", "--verbose")) {
  console.log(`Verbose mode is ON`)
  verbose = true
}

console.log(`Cli: ${cliPath}`)

// Download models

const baseURL = "https://huggingface.co/ggml-org/SmolVLM2-500M-Video-Instruct-GGUF"

const modelName = "SmolVLM2-500M-Video-Instruct-Q8_0.gguf"
const modelURL = `${baseURL}/resolve/main/${modelName}`
const modelHash = `6f67b8036b2469fcd71728702720c6b51aebd759b78137a8120733b4d66438bc`

const mmprojName = `mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf`
const mmprojURL = `${baseURL}/resolve/main/${mmprojName}`
const mmprojHash = `921dc7e259f308e5b027111fa185efcbf33db13f6e35749ddf7f5cdb60ef520b`

const modelPath = resolve(currentDir, modelName)
const mmprojPath = resolve(currentDir, mmprojName)
const mp4Path = resolve(currentDir, "circle.mp4")
const userPrompt = `
Describe video (with audio) as 1 sentence.
`
const timeoutSec = 120

if (!await exists(modelPath)) {
  console.log("Fetching model...")
  await writeBlob(modelURL)
  console.log(`Model ${modelName} downloaded!`)
}
const modelHashResult = await createHash(modelPath)
if (modelHashResult !== modelHash) {
  console.error(`Model hash isn't match! Model hash: ${modelHashResult}`)
  process.exit(-1)
}

console.log(`Check: model is valid.`)

if (!await exists(mmprojPath)) {
  console.log("Fetching mmproj...")
  await writeBlob(mmprojURL, mmprojName)
  console.log(`Mmproj downloaded!`)
}
const mmprojHashResult = await createHash(mmprojPath)
if (mmprojHashResult !== mmprojHash) {
  console.error(`Mmproj hash isn't match! Mmproj hash: ${mmprojHashResult}`)
  process.exit(-1)
}

console.log(`Check: mmproj is valid.`)

console.log(`\n\nExecuting llama-cli...\n\n`)

const llamaCliParams = [
  "-m", modelPath,
  "-mm", mmprojPath,
  "--video", mp4Path,
  "-p", userPrompt,
  "--reasoning", "off",
  "--single-turn",
]

if (verbose) {
  llamaCliParams.push("-v")
}

const exec = spawn(
  cliPath,
  llamaCliParams,
)

exec.stdout.on("data", (data: Uint8Array) => process.stdout.write(data))
exec.stderr.on("data", (data: Uint8Array) => process.stderr.write(data))
exec.on("error", (err) => {
  console.error(err)
})

const killDaemon = setTimeout(() => {
  console.error(`Llama-cli timeout for ${timeoutSec} sec.`)
  exec.kill()
  process.exit(1)
}, timeoutSec * 1000)

process.on("SIGINT", () => {
  if (!exec.killed) {
    exec.kill()
  }
  clearTimeout(killDaemon)
  process.exit(1)
})

exec.once("exit", (code, signal) => {
  console.log(`Process exited with ${code}`)
  console.log(`
##########################
Test result: ${code === 0 ? "success" : "failed"}!  
##########################
`)
  clearTimeout(killDaemon)
  process.exit(code ?? 1)
})