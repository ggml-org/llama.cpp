//@ts-check
// ALERT - Simple Stupid flow - Using from a discardable VM is better
// Helpers to handle tools/functions calling related to local/web access, pdf, etal
// which work in sync with the bundled simpleproxy.py server logic.
// Uses the js specific web worker path.
// by Humans for All
//

//
// The simpleproxy.py server is expected to provide the below services
// urlraw - fetch the request url content as is
// htmltext - fetch the requested html content and provide plain text version
//     after stripping it of tag blocks like head, script, style, header, footer, nav, ...
// pdftext - fetch the requested pdf and provide the plain text version
// xmlfiltered - fetch the requested xml content and provide a optionally filtered version of same
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
 * Setup a proxy server dependent tool call
 * NOTE: Currently the logic is setup for the bundled simpleproxy.py
 * @param {string} tag
 * @param {string} chatId
 * @param {string} tcPath
 * @param {string} tcName
 * @param {{ [x: string]: any; }} tcsData
 * @param {mToolsMgr.TCSwitch} tcs
 */
async function proxyserver_tc_setup(tag, chatId, tcPath, tcName, tcsData, tcs) {
    tag = `${tag}:${chatId}`
    let chat = gMe.multiChat.simpleChats[chatId]
    await fetch(`${chat.cfg.tools.proxyUrl}/aum?url=${tcPath}.jambudweepe.akashaganga.multiverse.987654321123456789`).then(resp=>{
        if (resp.statusText != 'bharatavarshe') {
            console.log(`WARN:ToolWeb:${tag}:Dont forget to run the bundled local.tools/simpleproxy.py to enable me`)
            return
        } else {
            console.log(`INFO:ToolWeb:${tag}:Enabling...`)
        }
        tcs[tcName] = tcsData;
    }).catch(err=>console.log(`WARN:ToolWeb:${tag}:ProxyServer missing?:${err}\nDont forget to run the bundled local.tools/simpleproxy.py`))
}


//
// Fetch Url Raw
//


let fetchurlraw_meta = {
        "type": "function",
        "function": {
            "name": "fetch_url_raw",
            "description": "Fetch contents of the requested url (local file path / web based) through a proxy server and return the got content as is, in few seconds. Mainly useful for getting textual non binary contents",
            "parameters": {
                "type": "object",
                "properties": {
                    "url":{
                        "type":"string",
                        "description":"url of the local file / web content to fetch"
                    }
                },
                "required": ["url"]
            }
        }
    }


/**
 * Implementation of the fetch url raw logic.
 * Expects a simple minded proxy server to be running locally
 * * listening on a configured port
 * * expecting http requests
 *   * with a query token named url wrt the path urlraw
 *     which gives the actual url to fetch
 * ALERT: Accesses a seperate/external web proxy/caching server, be aware and careful
 * @param {string} chatid
 * @param {string} toolcallid
 * @param {string} toolname
 * @param {any} obj
 */
function fetchurlraw_run(chatid, toolcallid, toolname, obj) {
    // maybe filter out any key other than 'url' in obj
    return proxyserver_get_anyargs(chatid, toolcallid, toolname, obj, 'urlraw');
}


/**
 * Setup fetch_url_raw for tool calling
 * NOTE: Currently the logic is setup for the bundled simpleproxy.py
 * @param {mToolsMgr.TCSwitch} tcs
 * @param {string} chatId
 */
async function fetchurlraw_setup(tcs, chatId) {
    return proxyserver_tc_setup('FetchUrlRaw', chatId, 'urlraw', 'fetch_url_raw', {
        "handler": fetchurlraw_run,
        "meta": fetchurlraw_meta,
        "result": ""
    }, tcs);
}


//
// Fetch html Text
//


let fetchhtmltext_meta = {
        "type": "function",
        "function": {
            "name": "fetch_html_text",
            "description": "Fetch html content from given url through a proxy server and return its text content after stripping away the html tags as well as head, script, style, header, footer, nav blocks, in few seconds",
            "parameters": {
                "type": "object",
                "properties": {
                    "url":{
                        "type":"string",
                        "description":"url of the html page that needs to be fetched and inturn unwanted stuff stripped from its contents to some extent"
                    }
                },
                "required": ["url"]
            }
        }
    }


/**
 * Implementation of the fetch html text logic.
 * Expects the simple minded simpleproxy server to be running locally,
 * providing service for htmltext path.
 * ALERT: Accesses a seperate/external web proxy/caching server, be aware and careful
 * @param {string} chatid
 * @param {string} toolcallid
 * @param {string} toolname
 * @param {any} obj
 */
function fetchhtmltext_run(chatid, toolcallid, toolname, obj) {
    // maybe filter out any key other than 'url' in obj
    return proxyserver_get_anyargs(chatid, toolcallid, toolname, obj, 'htmltext');
}


/**
 * Setup fetch_html_text for tool calling
 * NOTE: Currently the logic is setup for the bundled simpleproxy.py
 * @param {mToolsMgr.TCSwitch} tcs
 * @param {string} chatId
 */
async function fetchhtmltext_setup(tcs, chatId) {
    return proxyserver_tc_setup('FetchHtmlText', chatId, 'htmltext', 'fetch_html_text', {
        "handler": fetchhtmltext_run,
        "meta": fetchhtmltext_meta,
        "result": ""
    }, tcs);
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


//
// FetchPdfText
//


let fetchpdftext_meta = {
        "type": "function",
        "function": {
            "name": "fetch_pdf_as_text",
            "description": "Fetch pdf from requested local file path / web url through a proxy server and return its text content after converting pdf to text, in few seconds. One is allowed to get a part of the pdf by specifying the starting and ending page numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "url":{
                        "type":"string",
                        "description":"local file path (file://) / web (http/https) based url of the pdf that will be got and inturn converted to text"
                    },
                    "startPageNumber":{
                        "type":"integer",
                        "description":"Specify the starting page number within the pdf, this is optional. If not specified set to first page."
                    },
                    "endPageNumber":{
                        "type":"integer",
                        "description":"Specify the ending page number within the pdf, this is optional. If not specified set to the last page."
                    },
                },
                "required": ["url"]
            }
        }
    }


/**
 * Implementation of the fetch pdf as text logic.
 * Expects a simple minded proxy server to be running locally
 * * listening on a configured port
 * * expecting http requests
 *   * with a query token named url wrt pdftext path,
 *     which gives the actual url to fetch
 * * gets the requested pdf and converts to text, before returning same.
 * ALERT: Accesses a seperate/external web proxy/caching server, be aware and careful
 * @param {string} chatid
 * @param {string} toolcallid
 * @param {string} toolname
 * @param {any} obj
 */
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
// Fetch XML Filtered
//


let gRSSTagDropsDefault = [
    "^rss:channel:item:guid:.*",
    "^rss:channel:item:link:.*",
    "^rss:channel:item:description:.*",
    ".*:image:.*",
    ".*:enclosure:.*"
];

let fetchxmlfiltered_meta = {
        "type": "function",
        "function": {
            "name": "fetch_xml_filtered",
            "description": "Fetch requested xml url through a proxy server that can optionally filter out unwanted tags and their contents. Will take few seconds",
            "parameters": {
                "type": "object",
                "properties": {
                    "url":{
                        "type":"string",
                        "description":"url of the xml file that will be fetched"
                    },
                    "tagDropREs":{
                        "type":"string",
                        "description":`Optionally specify a json stringified list of xml tag heirarchies to drop.
                        For each tag that needs to be dropped, one needs to specify regular expression of the heirarchy of tags involved,
                        where the tag names are always mentioned in lower case along with a : as suffix.
                        For example for rss feeds one could use ${JSON.stringify(gRSSTagDropsDefault)} and so...`
                    }
                },
                "required": ["url"]
            }
        }
    }


/**
 * Implementation of the fetch xml filtered logic.
 * Expects simpleproxy to be running at specified url and providing xmltext service
 * ALERT: Accesses a seperate/external web proxy/caching server, be aware and careful
 * @param {string} chatid
 * @param {string} toolcallid
 * @param {string} toolname
 * @param {any} obj
 */
function fetchxmlfiltered_run(chatid, toolcallid, toolname, obj) {
    let tagDropREs = obj.tagDropREs
    if (tagDropREs == undefined) {
        tagDropREs = JSON.stringify([]) // JSON.stringify(gRSSTagDropsDefault)
    }
    let headers = { 'xmlfiltered-tagdrop-res': tagDropREs }
    return proxyserver_get_anyargs(chatid, toolcallid, toolname, obj, 'xmlfiltered', headers);
}


/**
 * Setup fetch_xml_filtered for tool calling
 * NOTE: Currently the logic is setup for the bundled simpleproxy.py
 * @param {mToolsMgr.TCSwitch} tcs
 * @param {string} chatId
 */
async function fetchxmlfiltered_setup(tcs, chatId) {
    return proxyserver_tc_setup('FetchXmlFiltered', chatId, 'xmlfiltered', 'fetch_xml_filtered', {
        "handler": fetchxmlfiltered_run,
        "meta": fetchxmlfiltered_meta,
        "result": ""
    }, tcs);
}


//
// Entry point
//


/**
 * Used to get hold of the web worker to use for running tool/function call related code.
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
    /**
     * @type {mToolsMgr.TCSwitch} tcs
     */
    let tc_switch = {}
    await fetchurlraw_setup(tc_switch, chatId)
    await fetchhtmltext_setup(tc_switch, chatId)
    await searchwebtext_setup(tc_switch, chatId)
    await fetchpdftext_setup(tc_switch, chatId)
    await fetchxmlfiltered_setup(tc_switch, chatId)
    return tc_switch
}
