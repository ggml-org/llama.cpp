// @ts-check
// A simple implementation of GenAi/LLM chat web client ui / front end logic.
// It handshake with ai server's completions and chat/completions endpoints
// and helps with basic usage and testing.
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
    gMe.toolsMgr.init(gMe).then(()=>{
        gMe.multiChat.chat_show(gMe.multiChat.curChatId);
    })
    for (let cid of gMe.defaultChatIds) {
        gMe.multiChat.new_chat_session(cid);
    }
    gMe.multiChat.setup_ui(gMe.defaultChatIds[0], true);
    gMe.multiChat.show_sessions();
}

document.addEventListener("DOMContentLoaded", startme);
