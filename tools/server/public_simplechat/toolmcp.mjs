//@ts-check
// ALERT - Simple Stupid flow - Using from a discardable VM is better
// simple mcpish client to handle tool/function calling provided by bundled simplemcp.py server logic.
// Currently it provides tool calls related to local/web access, pdf, etal
// by Humans for All
//

//
// The simplemcp.py mcpish server is expected to provide the below on /mcp service path
// tools/list - to get the meta of list of functions supported through simplemcp
// tools/call - to run the specified tool call
//


import * as mChatMagic from './simplechat.js'
import * as mToolsMgr from './tools.mjs'


/**
 * @type {mChatMagic.Me}
 */
let gMe = /** @type{mChatMagic.Me} */(/** @type {unknown} */(null));


/**
 * For now hash the shared secret with the year.
 * @param {mChatMagic.SimpleChat} chat
 */
async function bearer_transform(chat) {
    let data = `${new Date().getUTCFullYear()}${chat.cfg.tools.proxyAuthInsecure}`
    const ab = await crypto.subtle.digest('sha-256', new TextEncoder().encode(data));
    return Array.from(new Uint8Array(ab)).map(b => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Helper http get logic wrt the bundled SimpleProxy server,
 * which helps execute a given proxy dependent tool call.
 * Expects the simple minded proxy server to be running locally
 * * listening on a configured port
 * * expecting http requests
 *   * with a predefined query token and value wrt a predefined path
 * NOTE: Initial go, handles textual data type.
 * ALERT: Accesses a seperate/external web proxy/caching server, be aware and careful
 * @param {string} chatid
 * @param {string} toolcallid
 * @param {string} toolname
 * @param {any} objSearchParams
 * @param {string} path
 * @param {any} objHeaders
 */
async function proxyserver_get_anyargs(chatid, toolcallid, toolname, objSearchParams, path, objHeaders={}) {
    let chat = gMe.multiChat.simpleChats[chatid]
    if (gMe.toolsMgr.workers.js.onmessage != null) {
        let params = new URLSearchParams(objSearchParams)
        let newUrl = `${chat.cfg.tools.proxyUrl}/${path}?${params}`
        let headers = new Headers(objHeaders)
        let btoken = await bearer_transform(chat)
        headers.append('Authorization', `Bearer ${btoken}`)
        fetch(newUrl, { headers: headers}).then(resp => {
            if (!resp.ok) {
                throw new Error(`${resp.status}:${resp.statusText}`);
            }
            return resp.text()
        }).then(data => {
            gMe.toolsMgr.workers_postmessage_for_main(gMe.toolsMgr.workers.js, chatid, toolcallid, toolname, data);
        }).catch((err)=>{
            gMe.toolsMgr.workers_postmessage_for_main(gMe.toolsMgr.workers.js, chatid, toolcallid, toolname, `Error:${err}`);
        })
    }
}


/**
 * fetch supported tool calls meta data.
 * NOTE: Currently the logic is setup for the bundled simplemcp.py
 * @param {string} tag
 * @param {string} chatId
 * @param {mToolsMgr.TCSwitch} tcs
 */
async function mcpserver_toolslist(tag, chatId, tcs) {
    tag = `${tag}:${chatId}`
    try {
        let chat = gMe.multiChat.simpleChats[chatId]

        let id = new Date().getTime()
        let ibody = {
            jsonrpc: "2.0",
            id: id,
            method: "tools/list"
        }
        let headers = new Headers();
        headers.append("Content-Type", "application/json")
        let resp = await fetch(`${chat.cfg.tools.proxyUrl}/mcp`, {
            method: "POST",
            headers: headers,
            body: JSON.stringify(ibody),
        });
        if (resp.status != 200) {
            console.log`WARN:${tag}:ToolsList:MCP server says:${resp.status}:${resp.statusText}`
            return
        }
        let obody = await resp.json()
        if ((obody.results) && (obody.results.tools)) {
            for(const tcmeta of obody.results.tools) {
                if (!tcmeta.function) {
                    continue
                }
                console.log`INFO:${tag}:ToolsList:${tcmeta.function.name}`
                tcs[tcmeta.function.name] = {
                    "handler": mcpserver_toolcall,
                    "meta": tcmeta,
                    "result": ""
                }
            }
        }
    } catch (err) {
        console.log(`ERRR:${tag}:ToolsList:MCP server hs failed:${err}\nDont forget to run bundled local.tools/simplemcp.py`)
    }
}


//
// Search Web Text
//


let searchwebtext_meta = {
        "type": "function",
        "function": {
            "name": "search_web_text",
            "description": "search web for given words and return the plain text content after stripping the html tags as well as head, script, style, header, footer, nav blocks from got html result page, in few seconds",
            "parameters": {
                "type": "object",
                "properties": {
                    "words":{
                        "type":"string",
                        "description":"the words to search for on the web"
                    }
                },
                "required": ["words"]
            }
        }
    }


/**
 * Implementation of the search web text logic. Initial go.
 * Builds on htmltext path service of the bundled simpleproxy.py.
 * ALERT: Accesses a seperate/external web proxy/caching server, be aware and careful
 * @param {string} chatid
 * @param {string} toolcallid
 * @param {string} toolname
 * @param {any} obj
 */
function searchwebtext_run(chatid, toolcallid, toolname, obj) {
    let chat = gMe.multiChat.simpleChats[chatid]
    /** @type {string} */
    let searchUrl = chat.cfg.tools.searchUrl;
    searchUrl = searchUrl.replace("SEARCHWORDS", encodeURIComponent(obj.words));
    delete(obj.words)
    obj['url'] = searchUrl
    let headers = { 'htmltext-tag-drops': JSON.stringify(chat.cfg.tools.searchDrops) }
    return proxyserver_get_anyargs(chatid, toolcallid, toolname, obj, 'htmltext', headers);
}


/**
 * Setup search_web_text for tool calling
 * NOTE: Currently the logic is setup for the bundled simpleproxy.py
 * @param {mToolsMgr.TCSwitch} tcs
 * @param {string} chatId
 */
async function searchwebtext_setup(tcs, chatId) {
    return proxyserver_tc_setup('SearchWebText', chatId, 'htmltext', 'search_web_text', {
        "handler": searchwebtext_run,
        "meta": searchwebtext_meta,
        "result": ""
    }, tcs);
}


function fetchpdftext_run(chatid, toolcallid, toolname, obj) {
    return proxyserver_get_anyargs(chatid, toolcallid, toolname, obj, 'pdftext');
}


/**
 * Setup fetchpdftext for tool calling
 * NOTE: Currently the logic is setup for the bundled simpleproxy.py
 * @param {mToolsMgr.TCSwitch} tcs
 * @param {string} chatId
 */
async function fetchpdftext_setup(tcs, chatId) {
    return proxyserver_tc_setup('FetchPdfAsText', chatId, 'pdftext', 'fetch_pdf_as_text', {
        "handler": fetchpdftext_run,
        "meta": fetchpdftext_meta,
        "result": ""
    }, tcs);
}


//
// Entry point
//


/**
 * Used to get hold of the global Me instance, and through it
 * the toolsManager and chat settings ...
 * @param {mChatMagic.Me} me
 */
export async function init(me) {
    gMe = me
}


/**
 * Return the tool call switch with supported / enabled / available tool calls
 * Allows to verify / setup tool calls, which need to cross check things at runtime
 * before getting allowed, like maybe bcas they depend on a config wrt specified
 * chat session or handshake with mcpish server in this case and so...
 * @param {string} chatId
 */
export async function setup(chatId) {
    /**
     * @type {mToolsMgr.TCSwitch} tcs
     */
    let tc_switch = {}
    await mcpserver_toolslist("ToolMCP", chatId, tc_switch)
    return tc_switch
}
