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
                "description": "Task decomposition and planning",
                "tool_call": {
                    "name": "external_ai",
                    "arguments": {
                        "system_prompt": `
                        You are an expert task decomposition / planning assistant.
                        Your primary role is to understand the user's complex request and break it down into a series of manageable, sequential sub tasks.
                        Prioritize clarity and efficiency in your breakdown.
                        Present your plan as a numbered list, detailing each task along with its required tools and corresponding inputs/outputs.
                        End with a concise Next Step recommendation.
                        Focus on creating a robust execution plan that another ai can follow.`,
                        "user_message": "Find the last happening in the field of medicine focussed around ai and robotics",
                    }
                },
                "tool_response": {
                    "content": `
                    1. Clarify scope
                    1.1 Confirm desired depth (overview or deep dive).
                    1.2 Suggest possible focus areas (diagnostics AI, surgical robotics, rehab robotics).

                    2. Define suitable categories based on user clarification, for example
                    2.1 Medical AI - diagnostics, predictive analytics. drug discovery.
                    2.3 Medical robotics - surgical robots, rehabilitation and assistive robotics.

                    3. Identify authoritative sources
                    3.1 Peer reviewed journals (Nature medicine, The lancet, ...).
                    3.2 Major conferences.
                    3.3 Industry press releases from leading companies in these domains.

                    4. Research workflow / Sourcing strategy
                    4.1 Use search tool to gather recent news articles on “AI in medicine”, “robotic surgery”, ...
                    4.2 Fetch the top papers and clinical trials wrt each category.
                    4.3 Collate conference proceedings from last year on emerging research.

                    5. Extract & synthesize
                    5.1 List key papers/patents with main findings.
                    5.2 Summarize key clinical trials and their outcomes.
                    5.3 Highlight notable patents and product launches.
                    5.4 Note limitations, ethical concerns, regulatory status, ...

                    6. Structure output
                    6.1 Create Sections - Diagnostics AI, Treatment AI, Surgical Robotics, Rehab Robotics, ...
                    6.2 Present sub topics under each section with bullet points and concise explanations.

                    7. Review for accuracy and balance
                    7.1 Cross check facts with at least two independent sources.
                    7.2 Ensure representation of both benefits and current limitations/challenges.

                    8. Format the final output
                    8.1 Use Markdown for headings and bullet lists.
                    8.2 Include citations or links where appropriate.
                    8.3 Add an executive summary at the beginning.`
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
    if (gMe.tcexternalaiForceIsolatingDefaultsAlways) {
        sc.default_isolating()
    }
    sc.add_system_anytime(obj['system_prompt'], 'TC:ExternalAI')
    sc.add(new mChatMagic.ChatMessageEx(new mChatMagic.NSChatMessage(mChatMagic.Roles.User, obj['user_message'])))
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
