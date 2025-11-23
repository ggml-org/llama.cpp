//@ts-check
// ALERT - Simple Stupid flow - Using from a discardable VM is better
// Helpers to handle tools/functions calling wrt
// * calling a external / independent ai session
// by Humans for All
//

import * as mChatMagic from './simplechat.js'
import * as mToolsMgr from './tools.mjs'


let gMe = /** @type{mChatMagic.Me} */(/** @type {unknown} */(null));


let externalai_meta = {
    "type": "function",
    "function": {
        "name": "external_ai",
        "description": "Delegates a task to another AI instance using a custom system prompt and user message, that you as the caller define. Useful for tasks like summarization, structured data generation, or any custom AI workflow. This external ai doesnt have access to internet or tool calls",
        "parameters": {
            "type": "object",
            "properties": {
                "system_prompt": {
                    "type": "string",
                    "description": "The system prompt to define the role and expected behavior of the external AI.",
                    "required": true,
                    "example": "You are a professional summarizer. Summarize the following text with up to around 500 words, or as the case may be based on the context:"
                },
                "user_message": {
                    "type": "string",
                    "description": "The detailed message with all the needed context to be processed by the external AI.",
                    "required": true,
                    "example": "This is a long document about climate change. It discusses rising temperatures, policy responses, and future projections. The remaining part of the document is captured here..."
                },
                "model_name": {
                    "type": "string",
                    "description": "Optional identifier for the specific AI model to use (e.g., 'gpt-4', 'claude-3').",
                    "required": false,
                    "example": "gpt-4"
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum number of tokens to generate in the response.",
                    "required": false,
                    "example": 500
                },
            },
            required: [ "system_prompt", "user_message" ]
        },
        "examples": [
            {
                "description": "Custom summarization",
                "tool_call": {
                    "name": "external_ai",
                    "arguments": {
                    "system_prompt": "You are a professional summarizer. Summarize the following text in 100 words:",
                    "user_message": "The long text to summarise is passed here..."
                    }
                }
            },
            {
                "description": "Structured data generation",
                "tool_call": {
                    "name": "external_ai",
                    "arguments": {
                    "system_prompt": "You are a data structurer. Convert the following text into a JSON object with fields: title, author, year, and summary.",
                    "user_message": "The Indian epic 'Ramayana' by Valmiki is from eons back. It explores the fight of good against evil as well as dharma including how kings should conduct themselves and their responsibilities."
                    }
                }
            },
            {
                "description": "Literary critic",
                "tool_call": {
                    "name": "external_ai",
                    "arguments": {
                    "system_prompt": "You are a professional literary critic. Evaluate the provided summary of the Ramayana against key criteria: accuracy of core themes, completeness of major elements, and clarity. Provide a concise assessment.",
                    "user_message": "The Indian epic 'Ramayana' by Valmiki is from eons back. It explores the fight of good against evil as well as dharma including how kings should conduct themselves and their responsibilities."
                    }
                }
            },
        ]
    }
};


/**
 * Implementation of the external ai tool call.
 * @param {string} chatid
 * @param {string} toolcallid
 * @param {string} toolname
 * @param {any} obj
 */
function externalai_run(chatid, toolcallid, toolname, obj) {
    let sc = gMe.multiChat.simpleChats[mChatMagic.AI_TC_SESSIONNAME];
    sc.clear()
    sc.add_system_anytime(obj['system_prompt'], 'TC:ExternalAI')
    sc.add(new mChatMagic.ChatMessageEx(new mChatMagic.NSChatMessage(mChatMagic.Roles.User, obj['user_message'])))
    sc.cfg.tools.enabled = false
    sc.handle_chat_hs(sc.cfg.baseURL, mChatMagic.ApiEP.Type.Chat, gMe.multiChat.elDivStreams).then((resp)=>{
        gMe.toolsMgr.workers_postmessage_for_main(gMe.toolsMgr.workers.js, chatid, toolcallid, toolname, resp.content_equiv());
    }).catch((err)=>{
        gMe.toolsMgr.workers_postmessage_for_main(gMe.toolsMgr.workers.js, chatid, toolcallid, toolname, `Error:TC:ExternalAI:${err}`);
    })
}


/**
 * @type {mToolsMgr.TCSwitch}
 */
let tc_switch = {
    "external_ai": {
        "handler": externalai_run,
        "meta": externalai_meta,
        "result": ""
    },
}


/**
 * Helps to get hold of the below, when needed later
 * * needed Ai SimpleChat instance
 * * the web worker path to use for returning result of tool call
 * @param {mChatMagic.Me} me
 */
export async function init(me) {
    gMe = me
}


/**
 * Return the tool call switch with supported / enabled / available tool calls
 * Allows to verify / setup tool calls, which need to cross check things at runtime
 * before getting allowed, like maybe bcas they depend on a config wrt specified
 * chat session.
 * @param {string} chatId
 */
export async function setup(chatId) {
    return tc_switch;
}
