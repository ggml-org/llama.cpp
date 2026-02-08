//@ts-check
// simple minded helpers to handle markdown content
// by Humans for All
//


/**
 * A simple minded Markdown to Html convertor, which tries to support
 * basic forms of the below in a simple, stupid and some cases in a semi rigid way.
 * * headings
 * * fenced code blocks / pres
 * * unordered list
 * * tables
 * * horizontal line
 */
export class MarkDown {

    /**
     * Markdown parse and convert to html.
     * @param {boolean} bHtmlSanitize
     */
    constructor(bHtmlSanitize) {
        this.bHtmlSanitize = bHtmlSanitize
        this.in = {
            preFenced: "",
            table: {
                columns: 0,
                rawRow: 0,
            },
            list: {
                /** @type {Array<number>} */
                offsets: [],
                /** @type {Array<string>} */
                endType: [],
            },
            /** @type {Object<string, number>} */
            empty: {
            },
            /** @type {string} */
            blockQuote: "",
        }
        /**
         * @type {Array<*>}
         */
        this.errors = []
        this.raw = ""
        this.html = ""
    }

    /** @typedef {{prev: number, cur: number}} EmptyTrackerResult */

    /**
     * Track how many adjacent empty lines have been seen till now, in the immidate past.
     * as well as whether the current line is empty or otherwise.
     * @param {string} key
     * @param {string} line
     * @returns {EmptyTrackerResult}
     */
    empty_tracker(key, line) {
        if (this.in.empty[key] == undefined) {
            this.in.empty[key] = 0
        }
        let prev = this.in.empty[key]
        if (line.trim().length == 0) {
            this.in.empty[key] += 1
        } else {
            this.in.empty[key] = 0
        }
        return {prev: prev, cur: this.in.empty[key]}
    }

    /**
     * Append a new block to the end of html.
     * @param {string} line
     * @param {string} startMarker
     * @param {string} endMarker
     */
    appendnew(line, startMarker, endMarker) {
        this.html += `${startMarker}${line}${endMarker}`
    }

    /**
     * Extend to the existing last block
     * @param {string} line
     * @param {string} endMarker
     */
    extend(line, endMarker) {
        let html = this.html
        this.html = `${html.slice(0,html.length-endMarker.length)} ${line}${endMarker}`
    }

    /**
     * Extend the existing block, if
     * * there was no immidiate empty lines AND
     * * the existing block corresponds to what is specified.
     * Else
     * * append a new block
     *
     * @param {string} line
     * @param {string} endMarker
     * @param {string} startMarker
     * @param {EmptyTrackerResult} emptyTracker
     */
    extend_else_appendnew(line, endMarker, startMarker, emptyTracker) {
        if ((emptyTracker.prev != 0) || (!this.html.endsWith(endMarker))) {
            this.appendnew(line, startMarker, endMarker)
        } else {
            this.extend(line, endMarker)
        }
    }

    /**
     * Unwind till the specified offset level.
     * @param {number} unwindTillOffset
     */
    unwind_list(unwindTillOffset=-1) {
        if (this.in.list.offsets.length == 0) {
            return { done: true, remaining: 0 }
        }
        while (this.in.list.offsets[this.in.list.offsets.length-1] > unwindTillOffset) {
            this.in.list.offsets.pop()
            let popped = this.in.list.endType.pop()
            this.html += popped;
            if (this.in.list.offsets.length == 0) {
                break
            }
        }
        return { done: true, remaining: this.in.list.offsets.length }
    }

    /**
     * Process list one line at a time.
     *
     * Account for ordered lists as well as unordered lists, including intermixing of the lists.
     * * inturn at different list hierarchy levels.
     *
     * Allow a list item line to be split into multiple lines provided the split lines retain
     * the same or more line offset compared to the starting line of the item to which they belong.
     * * these following split lines wont have the list marker in front of them.
     *
     * Allows for empty lines inbetween items (ie lines with list marker)
     * * currently there is no limit on the number of empty lines, but may bring in a limit later.
     *
     * If empty line between a list item and new line with some content, but without a list marker
     * * if content offset less than last list item, then unwind the lists before such a line.
     * * if content offset larger than last list item, then line will be added as new list item
     *   at the same level as the last list item.
     * * if content offset same as last list item, then unwind list by one level and insert line
     *   as a new list item at this new unwound level.
     *
     * @param {string} line
     */
    process_list(line) {
        let emptyTracker = this.empty_tracker("list", line)
        // spaces followed by - or + or * followed by a space and actual list item
        let matchList = line.match(/^([ ]*)([-+*]|[0-9]+\.)[ ](.*)$/);
        if (matchList != null) {
            let listLvl = 0
            let curOffset = matchList[1].length
            let lastOffset = this.in.list.offsets[this.in.list.offsets.length-1];
            if (lastOffset == undefined) {
                lastOffset = -1
            }
            if (lastOffset < curOffset){
                this.in.list.offsets.push(curOffset)
                listLvl = this.in.list.offsets.length
                if (matchList[2][matchList[2].length-1] == '.') {
                    this.html += "<ol>\n"
                    this.in.list.endType.push("</ol>\n")
                } else {
                    this.html += "<ul>\n"
                    this.in.list.endType.push("</ul>\n")
                }
            } else if (lastOffset > curOffset){
                this.unwind_list(curOffset)
            }
            this.html += `<li>${matchList[3]}</li>\n`
            return true
        } else {
            if (this.in.list.offsets.length > 0) {

                if (emptyTracker.cur > 0) {
                    // skip empty line
                    return true
                }
                let matchOffset = line.match(/^([ ]*)(.*)$/);
                if (matchOffset == null) {
                    return false
                }
                let lastOffset = this.in.list.offsets[this.in.list.offsets.length-1];
                if (matchOffset[1].length < lastOffset) {
                    return false
                }

                if (emptyTracker.prev == 0) {
                    if (this.html.endsWith("</li>\n")) {
                        this.extend(matchOffset[2], "</li>\n")
                        return true
                    }
                } else {
                    if (matchOffset[1].length > lastOffset) {
                        this.appendnew(matchOffset[2], "<li>", "</li>\n")
                        return true
                    }
                    let uw = this.unwind_list(lastOffset-1)
                    if (uw.remaining > 0) {
                        this.appendnew(matchOffset[2], "<li>", "</li>\n")
                        return true
                    }
                }
                return false
            }
        }
        return false
    }

    /**
     * Try extract a table from markdown content, one line at a time.
     * This is a imperfect logic, but should give a rough semblance of a table many a times.
     * Purposefully allows for any text beyond table row end | marker to be shown.
     * @param {string} line
     */
    process_table_line(line) {
        if (!line.startsWith("|")) {
            if (this.in.table.columns > 0) {
                this.html += "</tbody>\n"
                this.html += "</table>\n"
                this.in.table.columns = 0
            }
            return false
        }
        let lineA = line.split('|')
        if (lineA.length > 2) {
            if (this.in.table.columns == 0) {
                // table heading
                this.html += "<table>\n<thead>\n<tr>\n"
                for(let i=1; i<lineA.length; i++) {
                    this.html += `<th>${lineA[i]}</th>\n`
                }
                this.html += "</tr>\n</thead>\n"
                this.in.table.columns = lineA.length-2;
                this.in.table.rawRow = 0
                return true
            }
            if (this.in.table.columns > 0) {
                if (this.in.table.columns != lineA.length-2) {
                    console.log("DBUG:TypeMD:Table:NonHead columns mismatch")
                }
                this.in.table.rawRow += 1
                if (this.in.table.rawRow == 1) {
                    // skip the table head vs body seperator
                    // rather skipping blindly without even checking if seperator or not.
                    this.html += "<tbody>\n"
                    return true
                }
                this.html += "<tr>\n"
                for(let i=1; i<lineA.length; i++) {
                    this.html += `<td>${lineA[i]}</td>\n`
                }
                this.html += "</tr>\n"
                return true
            }
            console.warn("DBUG:TypeMD:Table:Thrisanku???")
        } else {
            if (this.in.table.columns > 0) {
                this.html += "</tbody>\n"
                this.html += "</table>\n"
                this.in.table.columns = 0
            }
            return false
        }
    }

    /**
     * Process Pre Fenced block one line at a time.
     * @param {string} line
     */
    process_pre_fenced(line) {
        if (this.in.preFenced.length > 0) {
            if (line == this.in.preFenced) {
                this.in.preFenced = ""
                this.html += "</pre>\n"
            } else {
                this.html += `${line}\n`
            }
            return true
        }
        // same number of space followed by ``` or ~~~
        // some samples with spaces at beginning seen, so accepting spaces at begin
        let matchPreFenced = line.match(/^(\s*```|\s*~~~)([a-zA-Z0-9]*)(.*)/);
        if ( matchPreFenced != null) {
            this.unwind_list()
            this.in.preFenced = matchPreFenced[1]
            this.html += `<pre class="${matchPreFenced[2]}">\n`
            return true
        }
        return false
    }

    unwind_blockquote() {
        for(let i=0; i<this.in.blockQuote.length; i++) {
            this.html += `</blockquote>\n`
        }
        this.in.blockQuote = ""
    }

    /**
     * Handle blockquote block one line at a time.
     * This expects all lines in the block quote to have the marker at the begining.
     *
     * @param {string} lineRaw
     * @param {string} lineSani
     */
    process_blockquote(lineRaw, lineSani) {
        if (!lineRaw.startsWith(">")) {
            this.unwind_blockquote()
            return false
        }
        let startTok = lineRaw.split(' ', 1)[0]
        if (startTok.match(/^>+$/) == null) {
            this.unwind_blockquote()
            return false
        }
        this.unwind_list()
        if (startTok.length > this.in.blockQuote.length) {
            this.html += `<blockquote>\n`
        } else if (startTok.length < this.in.blockQuote.length) {
            this.html += `</blockquote>\n`
        }
        this.in.blockQuote = startTok
        this.html += `<p>${lineSani}</p>\n`
        return true
    }

    /**
     * Process headline.
     * @param {string} line
     */
    process_headline(line) {
        if (line.startsWith ("#")) {
            this.unwind_list()
            let startTok = line.split(' ', 1)[0]
            let hLevel = startTok.length
            this.html += `<h${hLevel}>${line.slice(hLevel)}</h${hLevel}>\n`
            return true
        }
        return false
    }

    /**
     * Process horizontal line.
     * @param {string} line
     */
    process_horizline(line) {
        // 3 or more of --- or ___ or *** followed by space
        // some online notes seemed to indicate spaces at end, so accepting same
        if (line.match(/^[-]{3,}|[*]{3,}|[_]{3,}\s*$/) != null) {
            this.unwind_list()
            this.html += "<hr>\n"
            return true
        }
        return false
    }

    /**
     * Process a line from markdown content
     * @param {string} lineRaw
     */
    process_line(lineRaw) {
        let line = ""
        if (this.bHtmlSanitize) {
            let elSanitize = document.createElement('div')
            elSanitize.textContent = lineRaw
            line = elSanitize.innerHTML
        } else {
            line = lineRaw
        }
        if (this.process_pre_fenced(line)) {
            return
        }
        if (this.process_table_line(line)) {
            return
        }
        if (this.process_horizline(line)) {
            return
        }
        if (this.process_headline(line)) {
            return
        }
        if (this.process_blockquote(lineRaw, line)) {
            return
        }
        if (this.process_list(line)) {
            return
        }
        this.unwind_list()
        let emptyTrackerPara = this.empty_tracker("para", line)
        this.extend_else_appendnew(line, "</p>\n", "<p>", emptyTrackerPara)
    }

    /**
     * Process a bunch of lines in markdown format.
     * @param {string} lines
     */
    process(lines) {
        this.raw = lines
        let linesA = lines.split('\n')
        for(const line of linesA) {
            try {
                this.process_line(line)
            } catch (err) {
                this.errors.push(err)
            }
        }
    }

}
