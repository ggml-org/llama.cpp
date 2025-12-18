//@ts-check
// Helpers to work with different data types
// by Humans for All
//

/**
 * Given the limited context size of local LLMs and , many a times when context gets filled
 * between the prompt and the response, it can lead to repeating text garbage generation.
 * And many a times setting penalty wrt repeatation leads to over-intelligent garbage
 * repeatation with slight variations. These garbage inturn can lead to overloading of the
 * available model context, leading to less valuable response for subsequent prompts/queries,
 * if chat history is sent to ai model.
 *
 * So two simple minded garbage trimming logics are experimented below.
 * * one based on progressively-larger-substring-based-repeat-matching-with-partial-skip and
 * * another based on char-histogram-driven garbage trimming.
 *   * in future characteristic of histogram over varying lengths could be used to allow for
 *     a more aggressive and adaptive trimming logic.
 */


/**
 * Simple minded logic to help remove repeating garbage at end of the string.
 * The repeatation needs to be perfectly matching.
 *
 * The logic progressively goes on probing for longer and longer substring based
 * repeatation, till there is no longer repeatation. Inturn picks the one with
 * the longest chain.
 *
 * @param {string} sIn
 * @param {number} maxSubL
 * @param {number} maxMatchLenThreshold
 */
export function trim_repeat_garbage_at_end(sIn, maxSubL=10, maxMatchLenThreshold=40) {
    let rCnt = [0];
    let maxMatchLen = maxSubL;
    let iMML = -1;
    for(let subL=1; subL < maxSubL; subL++) {
        rCnt.push(0);
        let i;
        let refS = sIn.substring(sIn.length-subL, sIn.length);
        for(i=sIn.length; i > 0; i -= subL) {
            let curS = sIn.substring(i-subL, i);
            if (refS != curS) {
                let curMatchLen = rCnt[subL]*subL;
                if (maxMatchLen < curMatchLen) {
                    maxMatchLen = curMatchLen;
                    iMML = subL;
                }
                break;
            }
            rCnt[subL] += 1;
        }
    }
    console.debug("DBUG:DU:TrimRepeatGarbage:", rCnt);
    if ((iMML == -1) || (maxMatchLen < maxMatchLenThreshold)) {
        return {trimmed: false, data: sIn};
    }
    console.debug("DBUG:TrimRepeatGarbage:TrimmedCharLen:", maxMatchLen);
    let iEnd = sIn.length - maxMatchLen;
    return { trimmed: true, data: sIn.substring(0, iEnd) };
}


/**
 * Simple minded logic to help remove repeating garbage at end of the string, till it cant.
 * If its not able to trim, then it will try to skip a char at end and then trim, a few times.
 * This ensures that even if there are multiple runs of garbage with different patterns, the
 * logic still tries to munch through them.
 *
 * @param {string} sIn
 * @param {number} maxSubL
 * @param {number | undefined} [maxMatchLenThreshold]
 */
export function trim_repeat_garbage_at_end_loop(sIn, maxSubL, maxMatchLenThreshold, skipMax=16) {
    let sCur = sIn;
    let sSaved = "";
    let iTry = 0;
    while(true) {
        let got = trim_repeat_garbage_at_end(sCur, maxSubL, maxMatchLenThreshold);
        if (got.trimmed != true) {
            if (iTry == 0) {
                sSaved = got.data;
            }
            iTry += 1;
            if (iTry >= skipMax) {
                return sSaved;
            }
            got.data = got.data.substring(0,got.data.length-1);
        } else {
            iTry = 0;
        }
        sCur = got.data;
    }
}


/**
 * A simple minded try trim garbage at end using histogram driven characteristics.
 * There can be variation in the repeatations, as long as no new char props up.
 *
 * This tracks the chars and their frequency in a specified length of substring at the end
 * and inturn checks if moving further into the generated text from the end remains within
 * the same char subset or goes beyond it and based on that either trims the string at the
 * end or not. This allows to filter garbage at the end, including even if there are certain
 * kind of small variations in the repeated text wrt position of seen chars.
 *
 * Allow the garbage to contain upto maxUniq chars, but at the same time ensure that
 * a given type of char ie numerals or alphabets or other types dont cross the specified
 * maxType limit. This allows intermixed text garbage to be identified and trimmed.
 *
 * ALERT: This is not perfect and only provides a rough garbage identification logic.
 * Also it currently only differentiates between character classes wrt english.
 *
 * @param {string} sIn
 * @param {number} maxType
 * @param {number} maxUniq
 * @param {number} maxMatchLenThreshold
 */
export function trim_hist_garbage_at_end(sIn, maxType, maxUniq, maxMatchLenThreshold) {
    if (sIn.length < maxMatchLenThreshold) {
        return { trimmed: false, data: sIn };
    }
    let iAlp = 0;
    let iNum = 0;
    let iOth = 0;
    // Learn
    let hist = {};
    let iUniq = 0;
    for(let i=0; i<maxMatchLenThreshold; i++) {
        let c = sIn[sIn.length-1-i];
        if (c in hist) {
            hist[c] += 1;
        } else {
            if(c.match(/[0-9]/) != null) {
                iNum += 1;
            } else if(c.match(/[A-Za-z]/) != null) {
                iAlp += 1;
            } else {
                iOth += 1;
            }
            iUniq += 1;
            if (iUniq >= maxUniq) {
                break;
            }
            hist[c] = 1;
        }
    }
    console.debug("DBUG:TrimHistGarbage:", hist);
    if ((iAlp > maxType) || (iNum > maxType) || (iOth > maxType)) {
        return { trimmed: false, data: sIn };
    }
    // Catch and Trim
    for(let i=0; i < sIn.length; i++) {
        let c = sIn[sIn.length-1-i];
        if (!(c in hist)) {
            if (i < maxMatchLenThreshold) {
                return { trimmed: false, data: sIn };
            }
            console.debug("DBUG:TrimHistGarbage:TrimmedCharLen:", i);
            return { trimmed: true, data: sIn.substring(0, sIn.length-i+1) };
        }
    }
    console.debug("DBUG:TrimHistGarbage:Trimmed fully");
    return { trimmed: true, data: "" };
}

/**
 * Keep trimming repeatedly using hist_garbage logic, till you no longer can.
 * This ensures that even if there are multiple runs of garbage with different patterns,
 * the logic still tries to munch through them.
 *
 * @param {any} sIn
 * @param {number} maxType
 * @param {number} maxUniq
 * @param {number} maxMatchLenThreshold
 */
export function trim_hist_garbage_at_end_loop(sIn, maxType, maxUniq, maxMatchLenThreshold) {
    let sCur = sIn;
    while (true) {
        let got = trim_hist_garbage_at_end(sCur, maxType, maxUniq, maxMatchLenThreshold);
        if (!got.trimmed) {
            return got.data;
        }
        sCur = got.data;
    }
}

/**
 * Try trim garbage at the end by using both the hist-driven-garbage-trimming as well as
 * skip-a-bit-if-reqd-then-repeat-pattern-based-garbage-trimming, with blind retrying.
 * @param {string} sIn
 */
export function trim_garbage_at_end(sIn) {
    let sCur = sIn;
    for(let i=0; i<2; i++) {
        sCur = trim_hist_garbage_at_end_loop(sCur, 8, 24, 72);
        sCur = trim_repeat_garbage_at_end_loop(sCur, 32, 72, 12);
    }
    return sCur;
}


/**
 * NewLines array helper.
 * Allow for maintaining a list of lines.
 * Allow for a line to be builtup/appended part by part.
 */
export class NewLines {

    constructor() {
        /** @type {string[]} */
        this.lines = [];
    }

    /**
     * Extracts lines from the passed string and inturn either
     * append to a previous partial line or add a new line.
     * @param {string} sLines
     */
    add_append(sLines) {
        let aLines = sLines.split("\n");
        let lCnt = 0;
        for(let line of aLines) {
            lCnt += 1;
            // Add back newline removed if any during split
            if (lCnt < aLines.length) {
                line += "\n";
            } else {
                if (sLines.endsWith("\n")) {
                    line += "\n";
                }
            }
            // Append if required
            if (lCnt == 1) {
                let lastLine = this.lines[this.lines.length-1];
                if (lastLine != undefined) {
                    if (!lastLine.endsWith("\n")) {
                        this.lines[this.lines.length-1] += line;
                        continue;
                    }
                }
            }
            // Add new line
            this.lines.push(line);
        }
    }

    /**
     * Shift the oldest/earliest/0th line in the array. [Old-New|Earliest-Latest]
     * Optionally control whether only full lines (ie those with newline at end) will be returned
     * or will a partial line without a newline at end (can only be the last line) be returned.
     * @param {boolean} bFullWithNewLineOnly
     */
    shift(bFullWithNewLineOnly=true) {
        let line = this.lines[0];
        if (line == undefined) {
            return undefined;
        }
        if ((line[line.length-1] != "\n") && bFullWithNewLineOnly){
            return undefined;
        }
        return this.lines.shift();
    }

}
