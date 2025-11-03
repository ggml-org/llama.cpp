// @ts-check
// A simple completions and chat/completions test related web front end logic
// by Humans for All

import * as du from "./datautils.mjs";
import * as ui from "./ui.mjs"
import * as tools from "./tools.mjs"


class Roles {
    static System = "system";
    static User = "user";
    static Assistant = "assistant";
    static Tool = "tool";
    static ToolTemp = "TOOL.TEMP";
}

class ApiEP {
    static Type = {
        Chat: "chat",
        Completion: "completion",
    }
    /** @type {Object<string, string>} */
    static UrlSuffix = {
        'chat': `/chat/completions`,
        'completion': `/completions`,
    }

    /**
     * Build the url from given baseUrl and apiEp id.
     * @param {string} baseUrl
     * @param {string} apiEP
     */
    static Url(baseUrl, apiEP) {
        if (baseUrl.endsWith("/")) {
            baseUrl = baseUrl.substring(0, baseUrl.length-1);
        }
        return `${baseUrl}${this.UrlSuffix[apiEP]}`;
    }

}

/**
 * @typedef {{id: string, type: string, function: {name: string, arguments: string}}} NSToolCalls
 */

/**
 * @typedef {{role: string, content: string, reasoning_content: string, tool_calls: Array<NSToolCalls>}} NSChatMessage
 */

class ChatMessageEx {

    /**
     * Represent a Message in the Chat
     * @param {string} role
     * @param {string} content
     * @param {string} reasoning_content
     * @param {Array<any>} tool_calls
     * @param {string} trimmedContent
     */
    constructor(role = "", content="", reasoning_content="", tool_calls=[], trimmedContent="") {
        /** @type {NSChatMessage} */
        this.ns = { role: role, content: content, tool_calls: tool_calls, reasoning_content: reasoning_content }
        this.trimmedContent = trimmedContent;
    }

    /**
     * Create a new instance from an existing instance
     * @param {ChatMessageEx} old
     */
    static newFrom(old) {
        return new ChatMessageEx(old.ns.role, old.ns.content, old.ns.reasoning_content, old.ns.tool_calls, old.trimmedContent)
    }

    clear() {
        this.ns.role = "";
        this.ns.content = "";
        this.ns.reasoning_content = "";
        this.ns.tool_calls = [];
        this.trimmedContent = "";
    }

    /**
     * Create a all in one tool call result string
     * Use browser's dom logic to handle strings in a xml/html safe way by escaping things where needed,
     * so that extracting the same later doesnt create any problems.
     * @param {string} toolCallId
     * @param {string} toolName
     * @param {string} toolResult
     */
    static createToolCallResultAllInOne(toolCallId, toolName, toolResult) {
        let dp = new DOMParser()
        let doc = dp.parseFromString("<tool_response></tool_response>", "text/xml")
        for (const k of [["id", toolCallId], ["name", toolName], ["content", toolResult]]) {
            let el = doc.createElement(k[0])
            el.appendChild(doc.createTextNode(k[1]))
            doc.documentElement.appendChild(el)
        }
        let xmlStr = new XMLSerializer().serializeToString(doc);
        xmlStr = xmlStr.replace(/\/name><content/, '\/name>\n<content');
        return xmlStr;
    }

    /**
     * Extract the elements of the all in one tool call result string
     * @param {string} allInOne
     */
    static extractToolCallResultAllInOneSimpleMinded(allInOne) {
        const regex = /<tool_response>\s*<id>(.*?)<\/id>\s*<name>(.*?)<\/name>\s*<content>([\s\S]*?)<\/content>\s*<\/tool_response>/si;
        const caught = allInOne.match(regex)
        let data = { tool_call_id: "Error", name: "Error", content: "Error" }
        if (caught) {
            data = {
                tool_call_id: caught[1].trim(),
                name: caught[2].trim(),
                content: caught[3].trim()
            }
        }
        return data
    }

    /**
     * Extract the elements of the all in one tool call result string
     * This should potentially account for content tag having xml/html content within to an extent.
     *
     * NOTE: Rather text/html is a more relaxed/tolarent mode for parseFromString than text/xml.
     * NOTE: Maybe better to switch to a json string format or use a more intelligent xml encoder
     * in createToolCallResultAllInOne so that extractor like this dont have to worry about special
     * xml chars like & as is, in the AllInOne content. For now text/html tolarence seems ok enough.
     *
     * @param {string} allInOne
     */
    static extractToolCallResultAllInOne(allInOne) {
        const dParser = new DOMParser();
        const got = dParser.parseFromString(allInOne, 'text/html');
        const parseErrors = got.querySelector('parseerror')
        if (parseErrors) {
            console.debug("WARN:ChatMessageEx:ExtractToolCallResultAllInOne:", parseErrors.textContent.trim())
        }
        const id = got.querySelector('id')?.textContent.trim();
        const name = got.querySelector('name')?.textContent.trim();
        const content = got.querySelector('content')?.textContent.trim();
        let data = {
            tool_call_id: id? id : "Error",
            name: name? name : "Error",
            content: content? content : "Error"
        }
        return data
    }

    /**
     * Set extra members into the ns object
     * @param {string | number} key
     * @param {any} value
     */
    ns_set_extra(key, value) {
        // @ts-ignore
        this.ns[key] = value
    }

    /**
     * Remove specified key and its value from ns object
     * @param {string | number} key
     */
    ns_delete(key) {
        // @ts-ignore
        delete(this.ns[key])
    }

    /**
     * Update based on the drip by drip data got from network in streaming mode.
     * Tries to support both Chat and Completion endpoints
     * @param {any} nwo
     * @param {string} apiEP
     */
    update_stream(nwo, apiEP) {
        console.debug(nwo, apiEP)
        if (apiEP == ApiEP.Type.Chat) {
            if (nwo["choices"][0]["finish_reason"] === null) {
                let content = nwo["choices"][0]["delta"]["content"];
                if (content !== undefined) {
                    if (content !== null) {
                        this.ns.content += content;
                    } else {
                        this.ns.role = nwo["choices"][0]["delta"]["role"];
                    }
                } else {
                    let toolCalls = nwo["choices"][0]["delta"]["tool_calls"];
                    let reasoningContent = nwo["choices"][0]["delta"]["reasoning_content"];
                    if (toolCalls !== undefined) {
                        if (toolCalls[0]["function"]["name"] !== undefined) {
                            this.ns.tool_calls.push(toolCalls[0]);
                            /*
                            this.ns.tool_calls[0].function.name = toolCalls[0]["function"]["name"];
                            this.ns.tool_calls[0].id = toolCalls[0]["id"];
                            this.ns.tool_calls[0].type = toolCalls[0]["type"];
                            this.ns.tool_calls[0].function.arguments = toolCalls[0]["function"]["arguments"]
                            */
                        } else {
                            if (toolCalls[0]["function"]["arguments"] !== undefined) {
                                this.ns.tool_calls[0].function.arguments += toolCalls[0]["function"]["arguments"];
                            }
                        }
                    }
                    if (reasoningContent !== undefined) {
                        this.ns.reasoning_content += reasoningContent
                    }
                }
            }
        } else {
            try {
                this.ns.content += nwo["choices"][0]["text"];
            } catch {
                this.ns.content += nwo["content"];
            }
        }
    }

    /**
     * Update based on the data got from network in oneshot mode
     * @param {any} nwo
     * @param {string} apiEP
     */
    update_oneshot(nwo, apiEP) {
        if (apiEP == ApiEP.Type.Chat) {
            let curContent = nwo["choices"][0]["message"]["content"];
            if (curContent != undefined) {
                if (curContent != null) {
                    this.ns.content = curContent;
                }
            }
            let curRC = nwo["choices"][0]["message"]["reasoning_content"];
            if (curRC != undefined) {
                this.ns.reasoning_content = curRC;
            }
            let curTCs = nwo["choices"][0]["message"]["tool_calls"];
            if (curTCs != undefined) {
                this.ns.tool_calls = curTCs;
            }
        } else {
            try {
                this.ns.content = nwo["choices"][0]["text"];
            } catch {
                this.ns.content = nwo["content"];
            }
        }
    }

    has_toolcall() {
        if (this.ns.tool_calls.length == 0) {
            return false
        }
        return true
    }

    /**
     * Collate all the different parts of a chat message into a single string object.
     *
     * This currently includes reasoning, content and toolcall parts.
     */
    content_equiv() {
        let reasoning = ""
        let content = ""
        let toolcall = ""
        if (this.ns.reasoning_content.trim() !== "") {
            reasoning = `!!!Reasoning: ${this.ns.reasoning_content.trim()} !!!\n\n`;
        }
        if (this.ns.content !== "") {
            content = this.ns.content;
        }
        if (this.has_toolcall()) {
            toolcall = `\n\n<tool_call>\n<tool_name>${this.ns.tool_calls[0].function.name}</tool_name>\n<tool_args>${this.ns.tool_calls[0].function.arguments}</tool_args>\n</tool_call>\n`;
        }
        return `${reasoning} ${content} ${toolcall}`;
    }

}


function usage_note() {
    let sUsageNote = `
    <details>
    <summary id="UsageNote" class="role-system">Usage Note</summary>
    <ul class="ul1">
    <li> System prompt above, helps control ai response characteristics.</li>
        <ul class="ul2">
        <li> Completion mode - no system prompt normally.</li>
        </ul>
    <li> Use shift+enter for inserting enter/newline.</li>
    <li> Enter your query/response to ai assistant in text area provided below.</li>
    <li> settings-tools-enabled should be true to enable tool calling.</li>
        <ul class="ul2">
        <li> If ai assistant requests a tool call, verify same before triggering.</li>
        <li> submit tool response placed into user query/response text area</li>
        </ul>
    <li> ContextWindow = [System, Last[${gMe.chatProps.iRecentUserMsgCnt-1}] User Query/Resp, Cur Query].</li>
        <ul class="ul2">
        <li> ChatHistInCtxt, MaxTokens, ModelCtxt window to expand</li>
        </ul>
    </ul>
    </details>`;
    return sUsageNote;
}


/** @typedef {ChatMessageEx[]} ChatMessages */

/** @typedef {{iLastSys: number, xchat: ChatMessages}} SimpleChatODS */

class SimpleChat {

    /**
     * @param {string} chatId
     */
    constructor(chatId) {
        this.chatId = chatId;
        /**
         * Maintain in a form suitable for common LLM web service chat/completions' messages entry
         * @type {ChatMessages}
         */
        this.xchat = [];
        this.iLastSys = -1;
        this.latestResponse = new ChatMessageEx();
    }

    clear() {
        this.xchat = [];
        this.iLastSys = -1;
    }

    ods_key() {
        return `SimpleChat-${this.chatId}`
    }

    save() {
        /** @type {SimpleChatODS} */
        let ods = {iLastSys: this.iLastSys, xchat: this.xchat};
        localStorage.setItem(this.ods_key(), JSON.stringify(ods));
    }

    load() {
        let sods = localStorage.getItem(this.ods_key());
        if (sods == null) {
            return;
        }
        /** @type {SimpleChatODS} */
        let ods = JSON.parse(sods);
        this.iLastSys = ods.iLastSys;
        this.xchat = [];
        for (const cur of ods.xchat) {
            if (cur.ns == undefined) { // this relates to the old on-disk-structure/format, needs to be removed later
                /** @typedef {{role: string, content: string}} OldChatMessage */
                let tcur = /** @type {OldChatMessage} */(/** @type {unknown} */(cur));
                this.xchat.push(new ChatMessageEx(tcur.role, tcur.content))
            } else {
                this.xchat.push(new ChatMessageEx(cur.ns.role, cur.ns.content, cur.ns.reasoning_content, cur.ns.tool_calls, cur.trimmedContent))
            }
        }
    }

    /**
     * Recent chat messages.
     *
     * If iRecentUserMsgCnt < 0, Then return the full chat history
     *
     * Else Return chat messages from latest going back till the last/latest system prompt.
     * While keeping track that the number of user queries/messages doesnt exceed iRecentUserMsgCnt.
     * @param {number} iRecentUserMsgCnt
     */
    recent_chat(iRecentUserMsgCnt) {
        if (iRecentUserMsgCnt < 0) {
            return this.xchat;
        }
        if (iRecentUserMsgCnt == 0) {
            console.warn("WARN:SimpleChat:SC:RecentChat:iRecentUsermsgCnt of 0 means no user message/query sent");
        }
        /** @type {ChatMessages} */
        let rchat = [];
        let sysMsg = this.get_system_latest();
        if (sysMsg.ns.content.length != 0) {
            rchat.push(sysMsg)
        }
        let iUserCnt = 0;
        let iStart = this.xchat.length;
        for(let i=this.xchat.length-1; i > this.iLastSys; i--) {
            if (iUserCnt >= iRecentUserMsgCnt) {
                break;
            }
            let msg = this.xchat[i];
            if (msg.ns.role == Roles.User) {
                iStart = i;
                iUserCnt += 1;
            }
        }
        for(let i = iStart; i < this.xchat.length; i++) {
            let msg = this.xchat[i];
            if (msg.ns.role == Roles.System) {
                continue;
            }
            rchat.push(msg)
        }
        return rchat;
    }


    /**
     * Return recent chat messages in the format,
     * which can be directly sent to the ai server.
     * @param {number} iRecentUserMsgCnt - look at recent_chat for semantic
     */
    recent_chat_ns(iRecentUserMsgCnt) {
        let xchat = this.recent_chat(iRecentUserMsgCnt);
        let chat = [];
        for (const msg of xchat) {
            if (msg.ns.role == Roles.ToolTemp) {
                // Skip (temp) tool response which has not yet been accepted by user
                // In future need to check that it is the last message
                // and not something in between, which shouldnt occur normally.
                continue
            }
            let tmsg = ChatMessageEx.newFrom(msg);
            if (!tmsg.has_toolcall()) {
                tmsg.ns_delete("tool_calls")
            }
            if (tmsg.ns.reasoning_content.trim() === "") {
                tmsg.ns_delete("reasoning_content")
            }
            if (tmsg.ns.role == Roles.Tool) {
                let res = ChatMessageEx.extractToolCallResultAllInOne(tmsg.ns.content)
                tmsg.ns.content = res.content
                tmsg.ns_set_extra("tool_call_id", res.tool_call_id)
                tmsg.ns_set_extra("name", res.name)
            }
            chat.push(tmsg.ns);
        }
        return chat
    }

    /**
     * Add an entry into xchat.
     * If the last message in chat history is a ToolTemp message, discard it
     * as the runtime logic is asking for adding new message instead of promoting the tooltemp message.
     *
     * NOTE: A new copy is created and added into xchat.
     * Also update iLastSys system prompt index tracker
     * @param {ChatMessageEx} chatMsg
     */
    add(chatMsg) {
        if (this.xchat.length > 0) {
            let lastIndex = this.xchat.length - 1;
            if (this.xchat[lastIndex].ns.role == Roles.ToolTemp) {
                console.debug("DBUG:SimpleChat:Add:Discarding prev ToolTemp message...")
                this.xchat.pop()
            }
        }
        this.xchat.push(ChatMessageEx.newFrom(chatMsg));
        if (chatMsg.ns.role == Roles.System) {
            this.iLastSys = this.xchat.length - 1;
        }
        this.save();
        return true;
    }

    /**
     * Check if the last message in the chat history is a ToolTemp role based one.
     * If so, then
     * * update that to a regular Tool role based message.
     * * also update the content of that message to what is passed.
     * @param {string} content
     */
    promote_tooltemp(content) {
        let lastIndex = this.xchat.length - 1;
        if (lastIndex < 0) {
            console.error("DBUG:SimpleChat:PromoteToolTemp:No chat messages including ToolTemp")
            return
        }
        if (this.xchat[lastIndex].ns.role != Roles.ToolTemp) {
            console.error("DBUG:SimpleChat:PromoteToolTemp:LastChatMsg not ToolTemp")
            return
        }
        this.xchat[lastIndex].ns.role = Roles.Tool;
        this.xchat[lastIndex].ns.content = content;
    }

    /**
     * Setup the fetch headers.
     * It picks the headers from gMe.headers.
     * It inserts Authorization only if its non-empty.
     * @param {string} apiEP
     */
    fetch_headers(apiEP) {
        let headers = new Headers();
        for(let k in gMe.headers) {
            let v = gMe.headers[k];
            if ((k == "Authorization") && (v.trim() == "")) {
                continue;
            }
            headers.append(k, v);
        }
        return headers;
    }

    /**
     * Add needed fields wrt json object to be sent wrt LLM web services completions endpoint.
     * The needed fields/options are picked from a global object.
     * Add optional stream flag, if required.
     * Convert the json into string.
     * @param {Object<string, any>} obj
     */
    request_jsonstr_extend(obj) {
        for(let k in gMe.apiRequestOptions) {
            obj[k] = gMe.apiRequestOptions[k];
        }
        if (gMe.chatProps.stream) {
            obj["stream"] = true;
        }
        if (gMe.tools.enabled) {
            obj["tools"] = tools.meta();
        }
        return JSON.stringify(obj);
    }

    /**
     * Return a string form of json object suitable for chat/completions
     */
    request_messages_jsonstr() {
        let req = {
            messages: this.recent_chat_ns(gMe.chatProps.iRecentUserMsgCnt),
        }
        return this.request_jsonstr_extend(req);
    }

    /**
     * Return a string form of json object suitable for /completions
     * @param {boolean} bInsertStandardRolePrefix Insert "<THE_ROLE>: " as prefix wrt each role's message
     */
    request_prompt_jsonstr(bInsertStandardRolePrefix) {
        let prompt = "";
        let iCnt = 0;
        for(const msg of this.recent_chat(gMe.chatProps.iRecentUserMsgCnt)) {
            iCnt += 1;
            if (iCnt > 1) {
                prompt += "\n";
            }
            if (bInsertStandardRolePrefix) {
                prompt += `${msg.ns.role}: `;
            }
            prompt += `${msg.ns.content}`;
        }
        let req = {
            prompt: prompt,
        }
        return this.request_jsonstr_extend(req);
    }

    /**
     * Return a string form of json object suitable for specified api endpoint.
     * @param {string} apiEP
     */
    request_jsonstr(apiEP) {
        if (apiEP == ApiEP.Type.Chat) {
            return this.request_messages_jsonstr();
        } else {
            return this.request_prompt_jsonstr(gMe.chatProps.bCompletionInsertStandardRolePrefix);
        }
    }


    /**
     * Allow setting of system prompt, at any time.
     * Updates the system prompt, if one was never set or if the newly passed is different from the last set system prompt.
     * @param {string} sysPrompt
     * @param {string} msgTag
     */
    add_system_anytime(sysPrompt, msgTag) {
        if (sysPrompt.length <= 0) {
            return false;
        }

        if (this.iLastSys < 0) {
            return this.add(new ChatMessageEx(Roles.System, sysPrompt));
        }

        let lastSys = this.xchat[this.iLastSys].ns.content;
        if (lastSys !== sysPrompt) {
            return this.add(new ChatMessageEx(Roles.System, sysPrompt));
        }
        return false;
    }

    /**
     * Retrieve the latest system prompt related chat message entry.
     */
    get_system_latest() {
        if (this.iLastSys == -1) {
            return new ChatMessageEx(Roles.System);
        }
        return this.xchat[this.iLastSys];
    }


    /**
     * Handle the multipart response from server/ai-model
     * @param {Response} resp
     * @param {string} apiEP
     * @param {HTMLDivElement} elDiv
     */
    async handle_response_multipart(resp, apiEP, elDiv) {
        let elP = ui.el_create_append_p("", elDiv);
        elP.classList.add("chat-message-content-live")
        if (!resp.body) {
            throw Error("ERRR:SimpleChat:SC:HandleResponseMultiPart:No body...");
        }
        let tdUtf8 = new TextDecoder("utf-8");
        let rr = resp.body.getReader();
        this.latestResponse.clear()
        this.latestResponse.ns.role = Roles.Assistant
        let xLines = new du.NewLines();
        while(true) {
            let { value: cur,  done: done } = await rr.read();
            if (cur) {
                let curBody = tdUtf8.decode(cur, {stream: true});
                console.debug("DBUG:SC:PART:Str:", curBody);
                xLines.add_append(curBody);
            }
            while(true) {
                let curLine = xLines.shift(!done);
                if (curLine == undefined) {
                    break;
                }
                if (curLine.trim() == "") {
                    continue;
                }
                if (curLine.startsWith("data:")) {
                    curLine = curLine.substring(5);
                }
                if (curLine.trim() === "[DONE]") {
                    break;
                }
                let curJson = JSON.parse(curLine);
                console.debug("DBUG:SC:PART:Json:", curJson);
                this.latestResponse.update_stream(curJson, apiEP);
            }
            elP.innerText = this.latestResponse.content_equiv()
            elP.scrollIntoView(false);
            if (done) {
                break;
            }
        }
        console.debug("DBUG:SC:PART:Full:", this.latestResponse.content_equiv());
        return ChatMessageEx.newFrom(this.latestResponse);
    }

    /**
     * Handle the oneshot response from server/ai-model
     * @param {Response} resp
     * @param {string} apiEP
     */
    async handle_response_oneshot(resp, apiEP) {
        let respBody = await resp.json();
        console.debug(`DBUG:SimpleChat:SC:${this.chatId}:HandleUserSubmit:RespBody:${JSON.stringify(respBody)}`);
        let cm = new ChatMessageEx(Roles.Assistant)
        cm.update_oneshot(respBody, apiEP)
        return cm
    }

    /**
     * Handle the response from the server be it in oneshot or multipart/stream mode.
     * Also take care of the optional garbage trimming.
     * TODO: Need to handle tool calling and related flow, including how to show
     * the assistant's request for tool calling and the response from tool.
     * @param {Response} resp
     * @param {string} apiEP
     * @param {HTMLDivElement} elDiv
     */
    async handle_response(resp, apiEP, elDiv) {
        let theResp = null;
        if (gMe.chatProps.stream) {
            try {
                theResp = await this.handle_response_multipart(resp, apiEP, elDiv);
                this.latestResponse.clear();
            } catch (error) {
                theResp = this.latestResponse;
                theResp.ns.role = Roles.Assistant;
                this.add(theResp);
                this.latestResponse.clear();
                throw error;
            }
        } else {
            theResp = await this.handle_response_oneshot(resp, apiEP);
        }
        if (gMe.chatProps.bTrimGarbage) {
            let origMsg = theResp.ns.content;
            theResp.ns.content = du.trim_garbage_at_end(origMsg);
            theResp.trimmedContent = origMsg.substring(theResp.ns.content.length);
        }
        theResp.ns.role = Roles.Assistant;
        this.add(theResp);
        return theResp;
    }

    /**
     * Handle the chat handshake with the ai server
     * @param {string} baseURL
     * @param {string} apiEP
     * @param {HTMLDivElement} elDivChat - used to show chat response as it is being generated/recieved in streaming mode
     */
    async handle_chat_hs(baseURL, apiEP, elDivChat) {
        class ChatHSError extends Error {
            constructor(/** @type {string} */message) {
                super(message);
                this.name = 'ChatHSError'
            }
        }

        let theUrl = ApiEP.Url(baseURL, apiEP);
        let theBody = this.request_jsonstr(apiEP);
        console.debug(`DBUG:SimpleChat:${this.chatId}:HandleChatHS:${theUrl}:ReqBody:${theBody}`);

        let theHeaders = this.fetch_headers(apiEP);
        let resp = await fetch(theUrl, {
            method: "POST",
            headers: theHeaders,
            body: theBody,
        });

        if (resp.status >= 300) {
            throw new ChatHSError(`HandleChatHS:GotResponse:NotOk:${resp.status}:${resp.statusText}`);
        }

        return this.handle_response(resp, apiEP, elDivChat);
    }

    /**
     * Call the requested tool/function.
     * Returns undefined, if the call was placed successfully
     * Else some appropriate error message will be returned.
     * @param {string} toolcallid
     * @param {string} toolname
     * @param {string} toolargs
     */
    async handle_toolcall(toolcallid, toolname, toolargs) {
        if (toolname === "") {
            return "Tool/Function call name not specified"
        }
        try {
            return await tools.tool_call(this.chatId, toolcallid, toolname, toolargs)
        } catch (/** @type {any} */error) {
            return `Tool/Function call raised an exception:${error.name}:${error.message}`
        }
    }

}


class MultiChatUI {

    constructor() {
        /** @type {Object<string, SimpleChat>} */
        this.simpleChats = {};
        /** @type {string} */
        this.curChatId = "";

        this.TimePeriods = {
            ToolCallAutoTimeUnit: 1000
        }

        this.timers = {
            /**
             * Used to identify Delay with getting response from a tool call.
             * @type {number | undefined}
             */
            toolcallResponseTimeout: undefined,
            /**
             * Used to auto trigger tool call, after a set time, if enabled.
             * @type {number | undefined}
             */
            toolcallTriggerClick: undefined,
            /**
             * Used to auto submit tool call response, after a set time, if enabled.
             * @type {number | undefined}
             */
            toolcallResponseSubmitClick: undefined
        }

        /**
         * Used for tracking presence of any chat message in show related logics
         * @type {HTMLElement | null}
         */
        this.elLastChatMessage = null

        // the ui elements
        this.elInSystem = /** @type{HTMLInputElement} */(document.getElementById("system-in"));
        this.elDivChat = /** @type{HTMLDivElement} */(document.getElementById("chat-div"));
        this.elBtnUser = /** @type{HTMLButtonElement} */(document.getElementById("user-btn"));
        this.elInUser = /** @type{HTMLInputElement} */(document.getElementById("user-in"));
        this.elDivHeading = /** @type{HTMLSelectElement} */(document.getElementById("heading"));
        this.elDivSessions = /** @type{HTMLDivElement} */(document.getElementById("sessions-div"));
        this.elBtnSettings = /** @type{HTMLButtonElement} */(document.getElementById("settings"));
        this.elDivTool = /** @type{HTMLDivElement} */(document.getElementById("tool-div"));
        this.elBtnTool = /** @type{HTMLButtonElement} */(document.getElementById("tool-btn"));
        this.elInToolName = /** @type{HTMLInputElement} */(document.getElementById("toolname-in"));
        this.elInToolArgs = /** @type{HTMLInputElement} */(document.getElementById("toolargs-in"));

        this.validate_element(this.elInSystem, "system-in");
        this.validate_element(this.elDivChat, "chat-div");
        this.validate_element(this.elInUser, "user-in");
        this.validate_element(this.elDivHeading, "heading");
        this.validate_element(this.elDivChat, "sessions-div");
        this.validate_element(this.elBtnSettings, "settings");
        this.validate_element(this.elDivTool, "tool-div");
        this.validate_element(this.elInToolName, "toolname-in");
        this.validate_element(this.elInToolArgs, "toolargs-in");
        this.validate_element(this.elBtnTool, "tool-btn");
    }

    /**
     * Check if the element got
     * @param {HTMLElement | null} el
     * @param {string} msgTag
     */
    validate_element(el, msgTag) {
        if (el == null) {
            throw Error(`ERRR:SimpleChat:MCUI:${msgTag} element missing in html...`);
        } else {
            // @ts-ignore
            console.debug(`INFO:SimpleChat:MCUI:${msgTag} Id[${el.id}] Name[${el["name"]}]`);
        }
    }

    /**
     * Reset/Setup Tool Call UI parts as needed
     * @param {ChatMessageEx} ar
     * @param {boolean} bAuto - allows caller to explicitly control whether auto triggering should be setup.
     */
    ui_reset_toolcall_as_needed(ar, bAuto = false) {
        if (ar.has_toolcall()) {
            this.elDivTool.hidden = false
            this.elInToolName.value = ar.ns.tool_calls[0].function.name
            this.elInToolName.dataset.tool_call_id = ar.ns.tool_calls[0].id
            this.elInToolArgs.value = ar.ns.tool_calls[0].function.arguments
            this.elBtnTool.disabled = false
            if ((gMe.tools.auto > 0) && (bAuto)) {
                this.timers.toolcallTriggerClick = setTimeout(()=>{
                    this.elBtnTool.click()
                }, gMe.tools.auto*this.TimePeriods.ToolCallAutoTimeUnit)
            }
        } else {
            this.elDivTool.hidden = true
            this.elInToolName.value = ""
            this.elInToolName.dataset.tool_call_id = ""
            this.elInToolArgs.value = ""
            this.elBtnTool.disabled = true
        }
    }

    /**
     * Reset user input ui.
     * * clear user input (if requested, default true)
     * * enable user input
     * * set focus to user input
     * @param {boolean} [bClearElInUser=true]
     */
    ui_reset_userinput(bClearElInUser=true) {
        if (bClearElInUser) {
            this.elInUser.value = "";
        }
        this.elInUser.disabled = false;
        this.elInUser.focus();
    }

    /**
     * Show the passed function / tool call details in specified parent element.
     * @param {HTMLElement} elParent
     * @param {NSToolCalls} tc
     */
    show_message_toolcall(elParent, tc) {
        let secTC = document.createElement('section')
        secTC.classList.add('chat-message-toolcall')
        elParent.append(secTC)
        let entry = ui.el_create_append_p(`name: ${tc.function.name}`, secTC);
        entry = ui.el_create_append_p(`id: ${tc.id}`, secTC);
        let oArgs = JSON.parse(tc.function.arguments)
        for (const k in oArgs) {
            entry = ui.el_create_append_p(`arg: ${k}`, secTC);
            let secArg = document.createElement('section')
            secArg.classList.add('chat-message-toolcall-arg')
            secTC.append(secArg)
            secArg.innerText = oArgs[k]
        }
    }

    /**
     * Handles showing a chat message in UI.
     *
     * If handling message belonging to role
     * * ToolTemp, updates user query input element, if its the last message.
     * * Assistant which contains a tool req, shows tool call ui if needed. ie
     *   * if it is the last message OR
     *   * if it is the last but one message and there is a ToolTemp message next
     * @param {HTMLElement | undefined} elParent
     * @param {ChatMessageEx} msg
     * @param {number} iFromLast
     * @param {ChatMessageEx | undefined} nextMsg
     */
    show_message(elParent, msg, iFromLast, nextMsg) {
        // Handle ToolTemp
        if (msg.ns.role === Roles.ToolTemp) {
            if (iFromLast == 0) {
                this.elInUser.value = msg.ns.content;
            }
            return
        }
        // Create main section
        let secMain = document.createElement('section')
        secMain.classList.add(`role-${msg.ns.role}`)
        secMain.classList.add('chat-message')
        elParent?.append(secMain)
        this.elLastChatMessage = secMain;
        // Create role para
        let entry = ui.el_create_append_p(`${msg.ns.role}`, secMain);
        entry.className = `chat-message-role`;
        // Create content section
        let secContents = document.createElement('section')
        secContents.classList.add('chat-message-contents')
        secMain.append(secContents)
        // Add the content
        //entry = ui.el_create_append_p(`${msg.content_equiv()}`, secContents);
        let showList = []
        if (msg.ns.reasoning_content.trim().length > 0) {
            showList.push(['reasoning', `!!!Reasoning: ${msg.ns.reasoning_content.trim()} !!!\n\n`])
        }
        if (msg.ns.content.trim().length > 0) {
            showList.push(['content', msg.ns.content.trim()])
        }
        for (const [name, content] of showList) {
            if (content.length > 0) {
                entry = ui.el_create_append_p(`${content}`, secContents);
                entry.classList.add(`chat-message-${name}`)
            }
        }
        // Handle tool call ui, if reqd
        let bTC = false
        let bAuto = false
        if (msg.ns.role === Roles.Assistant) {
            if (iFromLast == 0) {
                bTC = true
                bAuto = true
            } else if ((iFromLast == 1) && (nextMsg != undefined)) {
                if (nextMsg.ns.role == Roles.ToolTemp) {
                    bTC = true
                }
            }
            if (bTC) {
                this.ui_reset_toolcall_as_needed(msg, bAuto);
            }
        }
        // Handle tool call non ui
        if (msg.has_toolcall() && !bTC) {
            for (const i in msg.ns.tool_calls) {
                this.show_message_toolcall(secContents, msg.ns.tool_calls[i])
            }
        }
    }

    /**
     * Refresh UI wrt given chatId, provided it matches the currently selected chatId
     *
     * Show the chat contents in elDivChat.
     * Also update
     * * the user query input box, with ToolTemp role message, if last one.
     * * the tool call trigger ui, with Tool role message, if last one.
     *
     * If requested to clear prev stuff and inturn no chat content then show
     * * usage info
     * * option to load prev saved chat if any
     * * as well as settings/info.
     *
     * @param {string} chatId
     * @param {boolean} bClear
     * @param {boolean} bShowInfoAll
     */
    chat_show(chatId, bClear=true, bShowInfoAll=false) {
        if (chatId != this.curChatId) {
            return false
        }
        let chat = this.simpleChats[this.curChatId];
        if (bClear) {
            this.elDivChat.replaceChildren();
            this.ui_reset_toolcall_as_needed(new ChatMessageEx());
        }
        this.elLastChatMessage = null
        let chatToShow = chat.recent_chat(gMe.chatProps.iRecentUserMsgCnt);
        for(const [i, x] of chatToShow.entries()) {
            let iFromLast = (chatToShow.length - 1)-i
            let nextMsg = undefined
            if (iFromLast == 1) {
                nextMsg = chatToShow[i+1]
            }
            this.show_message(this.elDivChat, x, iFromLast, nextMsg)
        }
        if (this.elLastChatMessage != null) {
            /** @type{HTMLElement} */(this.elLastChatMessage).scrollIntoView(false); // Stupid ts-check js-doc intersection ???
        } else {
            if (bClear) {
                this.elDivChat.innerHTML = usage_note();
                gMe.setup_load(this.elDivChat, chat);
                gMe.show_info(this.elDivChat, bShowInfoAll);
            }
        }
        return true
    }

    /**
     * Setup the needed callbacks wrt UI, curChatId to defaultChatId and
     * optionally switch to specified defaultChatId.
     * @param {string} defaultChatId
     * @param {boolean} bSwitchSession
     */
    setup_ui(defaultChatId, bSwitchSession=false) {

        this.curChatId = defaultChatId;
        if (bSwitchSession) {
            this.handle_session_switch(this.curChatId);
        }

        this.ui_reset_toolcall_as_needed(new ChatMessageEx());

        this.elBtnSettings.addEventListener("click", (ev)=>{
            this.elDivChat.replaceChildren();
            gMe.show_settings(this.elDivChat);
        });

        this.elBtnUser.addEventListener("click", (ev)=>{
            clearTimeout(this.timers.toolcallResponseSubmitClick)
            this.timers.toolcallResponseSubmitClick = undefined
            if (this.elInUser.disabled) {
                return;
            }
            this.handle_user_submit(this.curChatId, gMe.chatProps.apiEP).catch((/** @type{Error} */reason)=>{
                let msg = `ERRR:SimpleChat\nMCUI:HandleUserSubmit:${this.curChatId}\n${reason.name}:${reason.message}`;
                console.error(msg.replace("\n", ":"));
                alert(msg);
            });
        });

        this.elBtnTool.addEventListener("click", (ev)=>{
            clearTimeout(this.timers.toolcallTriggerClick)
            this.timers.toolcallTriggerClick = undefined
            if (this.elDivTool.hidden) {
                return;
            }
            this.handle_tool_run(this.curChatId);
        })

        // Handle messages from Tools web worker
        tools.setup((cid, tcid, name, data)=>{
            clearTimeout(this.timers.toolcallResponseTimeout)
            this.timers.toolcallResponseTimeout = undefined
            let chat = this.simpleChats[cid];
            let limitedData = data
            if (gMe.tools.iResultMaxDataLength > 0) {
                if (data.length > gMe.tools.iResultMaxDataLength) {
                    limitedData = data.slice(0, gMe.tools.iResultMaxDataLength) + `\n\n\nALERT: Data too long, was chopped ....`
                }
            }
            chat.add(new ChatMessageEx(Roles.ToolTemp, ChatMessageEx.createToolCallResultAllInOne(tcid, name, limitedData)))
            if (this.chat_show(cid)) {
                if (gMe.tools.auto > 0) {
                    this.timers.toolcallResponseSubmitClick = setTimeout(()=>{
                        this.elBtnUser.click()
                    }, gMe.tools.auto*this.TimePeriods.ToolCallAutoTimeUnit)
                }
            }
            this.ui_reset_userinput(false)
        })

        this.elInUser.addEventListener("keyup", (ev)=> {
            // allow user to insert enter into their message using shift+enter.
            // while just pressing enter key will lead to submitting.
            if ((ev.key === "Enter") && (!ev.shiftKey)) {
                let value = this.elInUser.value;
                this.elInUser.value = value.substring(0,value.length-1);
                this.elBtnUser.click();
                ev.preventDefault();
            }
        });

        this.elInSystem.addEventListener("keyup", (ev)=> {
            // allow user to insert enter into the system prompt using shift+enter.
            // while just pressing enter key will lead to setting the system prompt.
            if ((ev.key === "Enter") && (!ev.shiftKey)) {
                let value = this.elInSystem.value;
                this.elInSystem.value = value.substring(0,value.length-1);
                let chat = this.simpleChats[this.curChatId];
                chat.add_system_anytime(this.elInSystem.value, this.curChatId);
                this.chat_show(chat.chatId)
                ev.preventDefault();
            }
        });

    }

    /**
     * Setup a new chat session and optionally switch to it.
     * @param {string} chatId
     * @param {boolean} bSwitchSession
     */
    new_chat_session(chatId, bSwitchSession=false) {
        this.simpleChats[chatId] = new SimpleChat(chatId);
        if (bSwitchSession) {
            this.handle_session_switch(chatId);
        }
    }


    /**
     * Handle user query submit request, wrt specified chat session.
     *
     * NOTE: Currently the user query entry area is used for
     * * showing and allowing edits by user wrt tool call results
     *   in a predfined simple xml format,
     *   ie before they submit tool result to ai engine on server
     * * as well as for user to enter their own queries.
     * Based on presence of the predefined xml format data at beginning
     * the logic will treat it has a tool result and if not then as a
     * normal user query.
     *
     * @param {string} chatId
     * @param {string} apiEP
     */
    async handle_user_submit(chatId, apiEP) {

        let chat = this.simpleChats[chatId];

        // In completion mode, if configured, clear any previous chat history.
        // So if user wants to simulate a multi-chat based completion query,
        // they will have to enter the full thing, as a suitable multiline
        // user input/query.
        if ((apiEP == ApiEP.Type.Completion) && (gMe.chatProps.bCompletionFreshChatAlways)) {
            chat.clear();
        }

        this.ui_reset_toolcall_as_needed(new ChatMessageEx());

        chat.add_system_anytime(this.elInSystem.value, chatId);

        let content = this.elInUser.value;
        if (content.trim() == "") {
            console.debug(`WARN:SimpleChat:MCUI:${chatId}:HandleUserSubmit:Ignoring empty user input...`);
            return;
        }
        if (content.startsWith("<tool_response>")) {
            chat.promote_tooltemp(content)
        } else {
            chat.add(new ChatMessageEx(Roles.User, content))
        }
        this.chat_show(chat.chatId);

        this.elInUser.value = "working...";
        this.elInUser.disabled = true;

        try {
            let theResp = await chat.handle_chat_hs(gMe.baseURL, apiEP, this.elDivChat)
            if (chatId == this.curChatId) {
                this.chat_show(chatId);
                if (theResp.trimmedContent.length > 0) {
                    let p = ui.el_create_append_p(`TRIMMED:${theResp.trimmedContent}`, this.elDivChat);
                    p.className="role-trim";
                }
            } else {
                console.debug(`DBUG:SimpleChat:MCUI:HandleUserSubmit:ChatId has changed:[${chatId}] [${this.curChatId}]`);
            }
        } finally {
            this.ui_reset_userinput();
        }
    }

    /**
     * Handle running of specified tool call if any, for the specified chat session.
     * Also sets up a timeout, so that user gets control back to interact with the ai model.
     * @param {string} chatId
     */
    async handle_tool_run(chatId) {
        let chat = this.simpleChats[chatId];
        this.elInUser.value = "toolcall in progress...";
        this.elInUser.disabled = true;
        let toolname = this.elInToolName.value.trim()
        let toolCallId = this.elInToolName.dataset.tool_call_id;
        if (toolCallId === undefined) {
            toolCallId = "??? ToolCallId Missing ???"
        }
        let toolResult = await chat.handle_toolcall(toolCallId, toolname, this.elInToolArgs.value)
        if (toolResult !== undefined) {
            chat.add(new ChatMessageEx(Roles.ToolTemp, ChatMessageEx.createToolCallResultAllInOne(toolCallId, toolname, toolResult)))
            this.chat_show(chat.chatId)
            this.ui_reset_userinput(false)
        } else {
            this.timers.toolcallResponseTimeout = setTimeout(() => {
                chat.add(new ChatMessageEx(Roles.ToolTemp, ChatMessageEx.createToolCallResultAllInOne(toolCallId, toolname, `Tool/Function call ${toolname} taking too much time, aborting...`)))
                this.chat_show(chat.chatId)
                this.ui_reset_userinput(false)
            }, gMe.tools.toolCallResponseTimeoutMS)
        }
    }

    /**
     * Show buttons for NewChat and available chat sessions, in the passed elDiv.
     * If elDiv is undefined/null, then use this.elDivSessions.
     * Take care of highlighting the selected chat-session's btn.
     * @param {HTMLDivElement | undefined} elDiv
     */
    show_sessions(elDiv=undefined) {
        if (!elDiv) {
            elDiv = this.elDivSessions;
        }
        elDiv.replaceChildren();
        // Btn for creating new chat session
        let btnNew = ui.el_create_button("New CHAT", (ev)=> {
            if (this.elInUser.disabled) {
                console.error(`ERRR:SimpleChat:MCUI:NewChat:Current session [${this.curChatId}] awaiting response, ignoring request...`);
                alert("ERRR:SimpleChat\nMCUI:NewChat\nWait for response to pending query, before starting new chat session");
                return;
            }
            let chatId = `Chat${Object.keys(this.simpleChats).length}`;
            let chatIdGot = prompt("INFO:SimpleChat\nMCUI:NewChat\nEnter id for new chat session", chatId);
            if (!chatIdGot) {
                console.error("ERRR:SimpleChat:MCUI:NewChat:Skipping based on user request...");
                return;
            }
            this.new_chat_session(chatIdGot, true);
            this.create_session_btn(elDiv, chatIdGot);
            ui.el_children_config_class(elDiv, chatIdGot, "session-selected", "");
        });
        elDiv.appendChild(btnNew);
        // Btns for existing chat sessions
        let chatIds = Object.keys(this.simpleChats);
        for(let cid of chatIds) {
            let btn = this.create_session_btn(elDiv, cid);
            if (cid == this.curChatId) {
                btn.className = "session-selected";
            }
        }
    }

    /**
     * Create session button and append to specified Div element.
     * @param {HTMLDivElement} elDiv
     * @param {string} cid
     */
    create_session_btn(elDiv, cid) {
        let btn = ui.el_create_button(cid, (ev)=>{
            let target = /** @type{HTMLButtonElement} */(ev.target);
            console.debug(`DBUG:SimpleChat:MCUI:SessionClick:${target.id}`);
            if (this.elInUser.disabled) {
                console.error(`ERRR:SimpleChat:MCUI:SessionClick:${target.id}:Current session [${this.curChatId}] awaiting response, ignoring switch...`);
                alert("ERRR:SimpleChat\nMCUI:SessionClick\nWait for response to pending query, before switching");
                return;
            }
            this.handle_session_switch(target.id);
            ui.el_children_config_class(elDiv, target.id, "session-selected", "");
        });
        elDiv.appendChild(btn);
        return btn;
    }

    /**
     * Switch ui to the specified chatId and set curChatId to same.
     * @param {string} chatId
     */
    async handle_session_switch(chatId) {
        let chat = this.simpleChats[chatId];
        if (chat == undefined) {
            console.error(`ERRR:SimpleChat:MCUI:HandleSessionSwitch:${chatId} missing...`);
            return;
        }
        this.elInSystem.value = chat.get_system_latest().ns.content;
        this.elInUser.value = "";
        this.curChatId = chatId;
        this.chat_show(chatId, true, true);
        this.elInUser.focus();
        console.log(`INFO:SimpleChat:MCUI:HandleSessionSwitch:${chatId} entered...`);
    }

}


/**
 * Few web search engine url template strings.
 * The SEARCHWORDS keyword will get replaced by the actual user specified search words at runtime.
 */
const SearchURLS = {
    duckduckgo: {
        'template': "https://duckduckgo.com/html/?q=SEARCHWORDS",
        'drop': [ { 'tag': 'div', 'id': "header" } ]
    },
    bing: {
        'template': "https://www.bing.com/search?q=SEARCHWORDS", // doesnt seem to like google chrome clients in particular
    },
    brave: {
        'template': "https://search.brave.com/search?q=SEARCHWORDS",
    },
    google: {
        'template': "https://www.google.com/search?q=SEARCHWORDS", // doesnt seem to like any client in general
    },
}


class Me {

    constructor() {
        this.baseURL = "http://127.0.0.1:8080";
        this.defaultChatIds = [ "Default", "Other" ];
        this.multiChat = new MultiChatUI();
        this.tools = {
            enabled: true,
            proxyUrl: "http://127.0.0.1:3128",
            proxyAuthInsecure: "NeverSecure",
            searchUrl: SearchURLS.duckduckgo.template,
            searchDrops: SearchURLS.duckduckgo.drop,
            toolNames: /** @type {Array<string>} */([]),
            /**
             * Control the length of the tool call result data returned to ai after tool call.
             * A value of 0 is treated as unlimited data.
             */
            iResultMaxDataLength: 1024*128,
            /**
             * Control how many milliseconds to wait for tool call to respond, before generating a timed out
             * error response and giving control back to end user.
             */
            toolCallResponseTimeoutMS: 20000,
            /**
             * Control how many seconds to wait before auto triggering tool call or its response submission.
             * A value of 0 is treated as auto triggering disable.
             */
            auto: 0
        };
        this.chatProps = {
            apiEP: ApiEP.Type.Chat,
            stream: true,
            iRecentUserMsgCnt: 5,
            bCompletionFreshChatAlways: true,
            bCompletionInsertStandardRolePrefix: false,
            bTrimGarbage: true,
        };
        /** @type {Object<string, number>} */
        this.sRecentUserMsgCnt = {
            "Full": -1,
            "Last0": 1,
            "Last1": 2,
            "Last2": 3,
            "Last4": 5,
            "Last9": 10,
        };
        /** @type {Object<string, string>} */
        this.headers = {
            "Content-Type": "application/json",
            "Authorization": "", // Authorization: Bearer OPENAI_API_KEY
        }
        /**
         * Add needed fields wrt json object to be sent wrt LLM web services completions endpoint.
         * @type {Object<string, any>}
         */
        this.apiRequestOptions = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 2048,
            "n_predict": 2048,
            "cache_prompt": true,
            //"frequency_penalty": 1.2,
            //"presence_penalty": 1.2,
        };
    }

    /**
     * Disable console.debug by mapping it to a empty function.
     */
    debug_disable() {
        this.console_debug = console.debug;
        console.debug = () => {

        };
    }

    /**
     * Setup the load saved chat ui.
     * @param {HTMLDivElement} div
     * @param {SimpleChat} chat
     */
    setup_load(div, chat) {
        if (!(chat.ods_key() in localStorage)) {
            return;
        }
        div.innerHTML += `<p class="role-system">Restore</p>
        <p>Load previously saved chat session, if available</p>`;
        let btn = ui.el_create_button(chat.ods_key(), (ev)=>{
            console.log("DBUG:SimpleChat:SC:Load", chat);
            chat.load();
            queueMicrotask(()=>{
                this.multiChat.chat_show(chat.chatId, true, true);
                this.multiChat.elInSystem.value = chat.get_system_latest().ns.content;
            });
        });
        div.appendChild(btn);
    }

    /**
     * Show the configurable parameters info in the passed Div element.
     * @param {HTMLDivElement} elDiv
     * @param {boolean} bAll
     */
    show_info(elDiv, bAll=false) {
        let props = ["baseURL", "modelInfo","headers", "tools", "apiRequestOptions", "chatProps"];
        if (!bAll) {
            props = [ "baseURL", "modelInfo", "tools", "chatProps" ];
        }
        fetch(`${this.baseURL}/props`).then(resp=>resp.json()).then(json=>{
            this.modelInfo = {
                modelPath: json["model_path"],
                ctxSize: json["default_generation_settings"]["n_ctx"]
            }
            ui.ui_show_obj_props_info(elDiv, this, props, "Settings/Info (devel-tools-console document[gMe])", "", { legend: 'role-system' })
        }).catch(err=>console.log(`WARN:ShowInfo:${err}`))
    }

    /**
     * Show settings ui for configurable parameters, in the passed Div element.
     * @param {HTMLDivElement} elDiv
     */
    show_settings(elDiv) {
        ui.ui_show_obj_props_edit(elDiv, "", this, ["baseURL", "headers", "tools", "apiRequestOptions", "chatProps"], "Settings", (prop, elProp)=>{
            if (prop == "headers:Authorization") {
                // @ts-ignore
                elProp.placeholder = "Bearer OPENAI_API_KEY";
            }
            if (prop.startsWith("tools:toolName")) {
                /** @type {HTMLInputElement} */(elProp).disabled = true
            }
        }, [":chatProps:apiEP", ":chatProps:iRecentUserMsgCnt"], (propWithPath, prop, elParent)=>{
            if (propWithPath == ":chatProps:apiEP") {
                let sel = ui.el_creatediv_select("SetApiEP", "ApiEndPoint", ApiEP.Type, this.chatProps.apiEP, (val)=>{
                    // @ts-ignore
                    this.chatProps.apiEP = ApiEP.Type[val];
                });
                elParent.appendChild(sel.div);
            }
            if (propWithPath == ":chatProps:iRecentUserMsgCnt") {
                let sel = ui.el_creatediv_select("SetChatHistoryInCtxt", "ChatHistoryInCtxt", this.sRecentUserMsgCnt, this.chatProps.iRecentUserMsgCnt, (val)=>{
                    this.chatProps.iRecentUserMsgCnt = this.sRecentUserMsgCnt[val];
                });
                elParent.appendChild(sel.div);
            }
        })
    }

}


/** @type {Me} */
let gMe;

function startme() {
    console.log("INFO:SimpleChat:StartMe:Starting...");
    gMe = new Me();
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
