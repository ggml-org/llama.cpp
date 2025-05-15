import * as readline from 'node:readline'
import { stdin, stdout } from 'node:process'
import { readFileSync } from 'node:fs'
import { SchemaConverter }  from './public_legacy/json-schema-to-grammar.mjs'

const args = process.argv.slice(2);
const grammarJsonSchemaFile = args.find(
    (_, index) => args[index - 1] === "--grammar-json-schema"
);

const no_cached_prompt = args.find(
    (_, index) => args[index - 1] === "--no-cache-prompt"
) ?? "false";

const grammarFile = args.find((_, index) => args[index - 1] === "--grammar");

// Example usage: function,arguments
const grammarJsonSchemaPropOrder = args.find(
    (_, index) => args[index - 1] === "--grammar-json-schema-prop-order"
);
const propOrder = grammarJsonSchemaPropOrder
    ? grammarJsonSchemaPropOrder
          .split(",")
          .reduce((acc, cur, index) => ({ ...acc, [cur]: index }), {})
    : {};

let grammar = null
if (grammarJsonSchemaFile) {
    let schema = JSON.parse(readFileSync(grammarJsonSchemaFile, 'utf-8'))
    const converter = new SchemaConverter({prop_order: propOrder, allow_fetch: true})
    schema = await converter.resolveRefs(schema, grammarJsonSchemaFile)
    converter.visit(schema, '')
    grammar = converter.formatGrammar()
}
if (grammarFile) {
    grammar = readFileSync(grammarFile, 'utf-8')
}

// for cached prompt
let slot_id = -1;

const API_URL = 'http://127.0.0.1:8080'

const chat = [
    {
        human: "Hello, Assistant.",
        assistant: "Hello. How may I help you today?"
    },
    {
        human: "Please tell me the largest city in Europe.",
        assistant: "Sure. The largest city in Europe is Moscow, the capital of Russia."
    },
]

const instruction = `A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.`

function format_prompt(question) {
    return `${instruction}\n${
        chat.map(m =>`### Human: ${m.human}\n### Assistant: ${m.assistant}`).join("\n")
    }\n### Human: ${question}\n### Assistant:`
}

async function tokenize(content) {
    const result = await fetch(`${API_URL}/tokenize`, {
        method: 'POST',
        body: JSON.stringify({ content })
    })

    if (!result.ok) {
        return []
    }

    return await result.json().tokens
}

const n_keep = await tokenize(instruction).length

async function chat_completion(question) {
    const result = await fetch(`${API_URL}/completion`, {
        method: 'POST',
        body: JSON.stringify({
            prompt: format_prompt(question),
            temperature: 0.2,
            top_k: 40,
            top_p: 0.9,
            n_keep: n_keep,
            n_predict: 256,
            cache_prompt: no_cached_prompt === "false",
            slot_id: slot_id,
            stop: ["\n### Human:"], // stop completion after generating this
            grammar,
            stream: true,
        })
    })

    if (!result.ok) {
        return
    }

    let answer = ''

    for await (var chunk of result.body) {
        const t = Buffer.from(chunk).toString('utf8')
        if (t.startsWith('data: ')) {
            const message = JSON.parse(t.substring(6))
            slot_id = message.slot_id
            answer += message.content
            process.stdout.write(message.content)
            if (message.stop) {
                if (message.truncated) {
                    chat.shift()
                }
                break
            }
        }
    }

    process.stdout.write('\n')
    chat.push({ human: question, assistant: answer.trimStart() })
}

const rl = readline.createInterface({ input: stdin, output: stdout });

const readlineQuestion = (rl, query, options) => new Promise((resolve, reject) => {
    rl.question(query, options, resolve)
});

while(true) {
    const question = await readlineQuestion(rl, '> ')
    await chat_completion(question)
}
