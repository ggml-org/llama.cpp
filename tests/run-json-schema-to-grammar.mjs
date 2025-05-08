import { readFileSync } from "fs"
import { SchemaConverter } from "../tools/server/public_legacy/json-schema-to-grammar.mjs"

const [, , file] = process.argv
const url = `file://${file}`
let schema = JSON.parse(readFileSync(file, "utf8"));
const converter = new SchemaConverter({})
schema = await converter.resolveRefs(schema, url)
converter.visit(schema, '')
console.log(converter.formatGrammar())
