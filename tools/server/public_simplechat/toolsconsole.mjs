//@ts-check
// Helpers to handle tools/functions calling wrt console
// by Humans for All
//


/** The redirected console.log's capture-data-space */
export let gConsoleStr = ""
/**
 * Maintain original console.log, when needed
 * @type { {(...data: any[]): void} | null}
 */
let gOrigConsoleLog = null


/**
 * The trapping console.log
 * @param {any[]} args
 */
export function console_trapped(...args) {
    let res = args.map((arg)=>{
        if (typeof arg == 'object') {
            return JSON.stringify(arg);
        } else {
            return String(arg);
        }
    }).join(' ');
    gConsoleStr += `${res}\n`;
}

/**
 * Save the original console.log, if needed.
 * Setup redir of console.log.
 * Clear the redirected console.log's capture-data-space.
 */
export function console_redir() {
    if (gOrigConsoleLog == null) {
        if (console.log == console_trapped) {
            throw new Error("ERRR:ToolsConsole:ReDir:Original Console.Log lost???");
        }
        gOrigConsoleLog = console.log
    }
    console.log = console_trapped
    gConsoleStr = ""
}

/**
 * Revert the redirected console.log to the original console.log, if possible.
 */
export function console_revert() {
    if (gOrigConsoleLog !== null) {
        if (gOrigConsoleLog == console_trapped) {
            throw new Error("ERRR:ToolsConsole:Revert:Original Console.Log lost???");
        }
        console.log = gOrigConsoleLog
    }
}
