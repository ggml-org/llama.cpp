//@ts-check
// ALERT - Simple minded flow - Using from a discardable VM is better.
// Simple mcpish client to handle tool/function calling provided by bundled simplemcp.py server logic.
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
    let data = `${new Date().getUTCFullYear()}${chat.cfg.tools.mcpServerAuth}`
    const ab = await crypto.subtle.digest('sha-256', new TextEncoder().encode(data));
    return Array.from(new Uint8Array(ab)).map(b => b.toString(16).padStart(2, '0')).join('');
}


/**
 * Implements tool call execution through a mcpish server. Initial go.
 * NOTE: Currently only uses textual contents in the result.
 * NOTE: Currently the logic is setup to work with bundled simplemcp.py
 * ALERT: Accesses a seperate/external mcpish server, be aware and careful
 * @param {string} chatid
 * @param {string} toolcallid
 * @param {string} toolname
 * @param {any} obj
 */
async function mcpserver_toolcall(chatid, toolcallid, toolname, obj) {
    let chat = gMe.multiChat.simpleChats[chatid]
    if (gMe.toolsMgr.workers.js.onmessage == null) {
        return
    }
    try {
        let newUrl = `${chat.cfg.tools.mcpServerUrl}`
        let headers = new Headers();
        let btoken = await bearer_transform(chat)
        headers.append('Authorization', `Bearer ${btoken}`)
        headers.append("Content-Type", "application/json")
        let ibody = {
            jsonrpc: "2.0",
            id: toolcallid,
            method: "tools/call",
            params: {
                name: toolname,
                arguments: obj
            }
        }
        let resp = await fetch(newUrl, {
            method: "POST",
            headers: headers,
            body: JSON.stringify(ibody),
        });
        if (!resp.ok) {
            throw new Error(`${resp.status}:${resp.statusText}`);
        }
        let obody = await resp.json()
        let textResult = ""
        if ((obody.result) && (obody.result.content)) {
            for(const tcr of obody.result.content) {
                if (!tcr.text) {
                    continue
                }
                textResult += `\n\n${tcr.text}`
            }
        }
        gMe.toolsMgr.workers_postmessage_for_main(gMe.toolsMgr.workers.js, chatid, toolcallid, toolname, textResult);
    } catch (err) {
        gMe.toolsMgr.workers_postmessage_for_main(gMe.toolsMgr.workers.js, chatid, toolcallid, toolname, `Error:${err}`);
    }
}


/**
 * Fetch supported tool calls meta data from a mcpish server.
 * NOTE: Currently the logic is setup to work with bundled simplemcp.py
 * ALERT: Accesses a seperate/external mcpish server, be aware and careful
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
        let btoken = await bearer_transform(chat)
        headers.append('Authorization', `Bearer ${btoken}`)
        headers.append("Content-Type", "application/json")
        let resp = await fetch(`${chat.cfg.tools.mcpServerUrl}`, {
            method: "POST",
            headers: headers,
            body: JSON.stringify(ibody),
        });
        if (resp.status != 200) {
            console.log(`WARN:${tag}:ToolsList:MCP server says:${resp.status}:${resp.statusText}`)
            return
        }
        let obody = await resp.json()
        if ((obody.result) && (obody.result.tools)) {
            for(const tcmeta of obody.result.tools) {
                if (!tcmeta.function) {
                    continue
                }
                console.log(`INFO:${tag}:ToolsList:${tcmeta.function.name}`)
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
     * @type {mToolsMgr.TCSwitch}
     */
    let tc_switch = {}
    await mcpserver_toolslist("ToolMCP", chatId, tc_switch)
    return tc_switch
}
