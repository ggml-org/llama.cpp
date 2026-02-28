// @ts-check
// A simple minded GenAi/LLM chat web client implementation.
// Handshakes with
// * ai server's completions and chat/completions endpoints
// * simplemcp tool calls provider
// Helps with basic usage and testing.
// by Humans for All


import * as mChatMagic from './simplechat.js'
import * as du from "./datautils.mjs";


/** @type {mChatMagic.Me} */
let gMe;


function devel_expose() {
    // @ts-ignore
    document["gMe"] = gMe;
    // @ts-ignore
    document["du"] = du;
}


function startme() {
    console.log("INFO:SimpleChat:StartMe:Starting...");
    gMe = new mChatMagic.Me();
    gMe.debug_disable();
    devel_expose()
    gMe.toolsMgr.init(gMe).then(async ()=>{
        let sL = []
        for (let cid of gMe.defaultChatIds) {
            sL.push(gMe.multiChat.new_chat_session(cid));
        }
        await Promise.allSettled(sL)
        gMe.multiChat.simpleChats[mChatMagic.AI_TC_SESSIONNAME].default_isolating()
        gMe.multiChat.setup_ui(gMe.defaultChatIds[0]);
        gMe.multiChat.show_sessions();
        gMe.multiChat.handle_session_switch(gMe.multiChat.curChatId)
    })
}

document.addEventListener("DOMContentLoaded", startme);
