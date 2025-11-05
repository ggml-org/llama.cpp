//@ts-check
// ALERT - Simple Stupid flow - Using from a discardable VM is better
// Helpers to handle tools/functions calling related to web access, pdf, etal
// which work in sync with the bundled simpleproxy.py server logic.
// Uses the js specific web worker path.
// by Humans for All
//

import * as mChatMagic from './simplechat.js'


/**
 * @type {mChatMagic.Me}
 */
let gMe = /** @type{mChatMagic.Me} */(/** @type {unknown} */(null));


/**
 * For now hash the shared secret with the year.
 */
async function bearer_transform() {
    let data = `${new Date().getUTCFullYear()}${gMe.tools.proxyAuthInsecure}`
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
    if (gMe.workers.js.onmessage != null) {
        let params = new URLSearchParams(objSearchParams)
        let newUrl = `${gMe.tools.proxyUrl}/${path}?${params}`
        let headers = new Headers(objHeaders)
        let btoken = await bearer_transform()
        headers.append('Authorization', `Bearer ${btoken}`)
        fetch(newUrl, { headers: headers}).then(resp => {
            if (!resp.ok) {
                throw new Error(`${resp.status}:${resp.statusText}`);
            }
            return resp.text()
        }).then(data => {
            gMe.workers_postmessage_for_main(gMe.workers.js, chatid, toolcallid, toolname, data);
        }).catch((err)=>{
            gMe.workers_postmessage_for_main(gMe.workers.js, chatid, toolcallid, toolname, `Error:${err}`);
        })
    }
}


/**
 * Setup a proxy server dependent tool call
 * NOTE: Currently the logic is setup for the bundled simpleproxy.py
 * @param {string} tag
 * @param {string} tcPath
 * @param {string} tcName
 * @param {{ [x: string]: any; }} tcsData
 * @param {Object<string, Object<string, any>>} tcs
 */
async function proxyserver_tc_setup(tag, tcPath, tcName, tcsData, tcs) {
    await fetch(`${gMe.tools.proxyUrl}/aum?url=${tcPath}.jambudweepe.akashaganga.multiverse.987654321123456789`).then(resp=>{
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
// Fetch Web Url Raw
//


let fetchweburlraw_meta = {
        "type": "function",
        "function": {
            "name": "fetch_web_url_raw",
            "description": "Fetch the requested web url through a proxy server and return the got content as is, in few seconds",
            "parameters": {
                "type": "object",
                "properties": {
                    "url":{
                        "type":"string",
                        "description":"url of the web page to fetch from the internet"
                    }
                },
                "required": ["url"]
            }
        }
    }


/**
 * Implementation of the fetch web url raw logic.
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
function fetchweburlraw_run(chatid, toolcallid, toolname, obj) {
    // maybe filter out any key other than 'url' in obj
    return proxyserver_get_anyargs(chatid, toolcallid, toolname, obj, 'urlraw');
}


/**
 * Setup fetch_web_url_raw for tool calling
 * NOTE: Currently the logic is setup for the bundled simpleproxy.py
 * @param {Object<string, Object<string, any>>} tcs
 */
async function fetchweburlraw_setup(tcs) {
    return proxyserver_tc_setup('FetchWebUrlRaw', 'urlraw', 'fetch_web_url_raw', {
        "handler": fetchweburlraw_run,
        "meta": fetchweburlraw_meta,
        "result": ""
    }, tcs);
}


//
// Fetch Web Url Text
//


let fetchweburltext_meta = {
        "type": "function",
        "function": {
            "name": "fetch_web_url_text",
            "description": "Fetch the requested web url through a proxy server and return its text content after stripping away the html tags as well as head, script, style, header, footer, nav blocks, in few seconds",
            "parameters": {
                "type": "object",
                "properties": {
                    "url":{
                        "type":"string",
                        "description":"url of the page that will be fetched from the internet and inturn unwanted stuff stripped from its contents to some extent"
                    }
                },
                "required": ["url"]
            }
        }
    }


/**
 * Implementation of the fetch web url text logic.
 * Expects a simple minded proxy server to be running locally
 * * listening on a configured port
 * * expecting http requests
 *   * with a query token named url wrt urltext path,
 *     which gives the actual url to fetch
 * * strips out head as well as any script, style, header, footer, nav and so blocks in body
 *   before returning remaining body contents.
 * ALERT: Accesses a seperate/external web proxy/caching server, be aware and careful
 * @param {string} chatid
 * @param {string} toolcallid
 * @param {string} toolname
 * @param {any} obj
 */
function fetchweburltext_run(chatid, toolcallid, toolname, obj) {
    // maybe filter out any key other than 'url' in obj
    return proxyserver_get_anyargs(chatid, toolcallid, toolname, obj, 'urltext');
}


/**
 * Setup fetch_web_url_text for tool calling
 * NOTE: Currently the logic is setup for the bundled simpleproxy.py
 * @param {Object<string, Object<string, any>>} tcs
 */
async function fetchweburltext_setup(tcs) {
    return proxyserver_tc_setup('FetchWebUrlText', 'urltext', 'fetch_web_url_text', {
        "handler": fetchweburltext_run,
        "meta": fetchweburltext_meta,
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
 * Builds on urltext path of the bundled simpleproxy.py.
 * Expects simpleproxy.py server to be running locally
 * * listening on a configured port
 * * expecting http requests
 *   * with a query token named url wrt urltext path,
 *     which gives the actual url to fetch
 * * strips out head as well as any script, style, header, footer, nav and so blocks in body
 *   before returning remaining body contents.
 * ALERT: Accesses a seperate/external web proxy/caching server, be aware and careful
 * @param {string} chatid
 * @param {string} toolcallid
 * @param {string} toolname
 * @param {any} obj
 */
function searchwebtext_run(chatid, toolcallid, toolname, obj) {
    /** @type {string} */
    let searchUrl = gMe.tools.searchUrl;
    searchUrl = searchUrl.replace("SEARCHWORDS", encodeURIComponent(obj.words));
    delete(obj.words)
    obj['url'] = searchUrl
    let headers = { 'urltext-tag-drops': JSON.stringify(gMe.tools.searchDrops) }
    return proxyserver_get_anyargs(chatid, toolcallid, toolname, obj, 'urltext', headers);
}


/**
 * Setup search_web_text for tool calling
 * NOTE: Currently the logic is setup for the bundled simpleproxy.py
 * @param {Object<string, Object<string, any>>} tcs
 */
async function searchwebtext_setup(tcs) {
    return proxyserver_tc_setup('SearchWebText', 'urltext', 'search_web_text', {
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
 * @param {Object<string, Object<string, any>>} tcs
 */
async function fetchpdftext_setup(tcs) {
    return proxyserver_tc_setup('FetchPdfAsText', 'pdftext', 'fetch_pdf_as_text', {
        "handler": fetchpdftext_run,
        "meta": fetchpdftext_meta,
        "result": ""
    }, tcs);
}



/**
 * Used to get hold of the web worker to use for running tool/function call related code
 * Also to setup tool calls, which need to cross check things at runtime
 * @param {mChatMagic.Me} me
 */
export async function init(me) {
    /**
     * @type {Object<string, Object<string, any>>} tcs
     */
    let tc_switch = {}
    gMe = me
    await fetchweburlraw_setup(tc_switch)
    await fetchweburltext_setup(tc_switch)
    await searchwebtext_setup(tc_switch)
    await fetchpdftext_setup(tc_switch)
    return tc_switch
}
