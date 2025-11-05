// @ts-check
// A simple completions and chat/completions test related web front end logic
// by Humans for All


import * as mChatMagic from './simplechat.js'
import * as tools from "./tools.mjs"
import * as du from "./datautils.mjs";



/** @type {mChatMagic.Me} */
let gMe;

function startme() {
    console.log("INFO:SimpleChat:StartMe:Starting...");
    gMe = new mChatMagic.Me();
    gMe.debug_disable();
    // @ts-ignore
    document["gMe"] = gMe;
    // @ts-ignore
    document["du"] = du;
    // @ts-ignore
    document["tools"] = tools;
    tools.init().then((toolNames)=>gMe.tools.toolNames=toolNames).then(()=>gMe.multiChat.chat_show(gMe.multiChat.curChatId))
    for (let cid of gMe.defaultChatIds) {
        gMe.multiChat.new_chat_session(cid);
    }
    gMe.multiChat.setup_ui(gMe.defaultChatIds[0], true);
    gMe.multiChat.show_sessions();
}

document.addEventListener("DOMContentLoaded", startme);
