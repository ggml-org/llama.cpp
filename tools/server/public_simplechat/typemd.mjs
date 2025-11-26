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

    constructor() {
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
            }
        }
        /**
         * @type {Array<*>}
         */
        this.errors = []
        this.raw = ""
        this.html = ""
    }

    /**
     * @param {string} key
     * @param {string} line
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

    unwind_list() {
        while (true) {
            let popped = this.in.list.endType.pop()
            if (popped == undefined) {
                break
            }
            this.html += popped
        }
        this.in.list.offsets.length = 0
    }

    /**
     * Process list one line at a time.
     * * Account for ordered lists as well as unordered lists, including intermixing of the lists.
     *   at different list hierarchy levels.
     * * Allow a list item line to be split into multiple lines provided the split lines retain
     *   the same or more line offset compared to the starting line of the item to which they belong.
     *   * if there is a empty line in between, then the new line will be treated as a new item.
     * * allows for empty lines inbetween items.
     *   * currently there is no limit on the number of empty lines.
     *     but may bring in a limit later.
     * @param {string} line
     */
    process_list(line) {
        let emptyTracker = this.empty_tracker("list", line)
        // spaces followed by - or + or * followed by a space and actual list item
        let matchList = line.match(/^([ ]*)([-+*]|[a-zA-Z0-9]\.)[ ](.*)$/);
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
                while (this.in.list.offsets[this.in.list.offsets.length-1] > curOffset) {
                    this.in.list.offsets.pop()
                    let popped = this.in.list.endType.pop()
                    this.html += popped;
                    if (this.in.list.offsets.length == 0) {
                        break
                    }
                }
            }
            this.html += `<li>${matchList[3]}</li>\n`
            return true
        } else {
            if (this.in.list.offsets.length > 0) {
                if (emptyTracker.cur > 0) {
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
                if ((emptyTracker.prev != 0) || (!this.html.endsWith("</li>\n"))) {
                    this.html += `<li>${matchOffset[2]}</li>\n`
                } else {
                    let html = this.html
                    this.html = `${html.slice(0,html.length-"</li>\n".length)} ${matchOffset[2]}</li>\n`
                }
                return true
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
     * Process a line from markdown content
     * @param {string} line
     */
    process_line(line) {
        let elSanitize = document.createElement('div')
        elSanitize.textContent = line
        line = elSanitize.innerHTML
        let lineA = line.split(' ')
        if (this.in.preFenced.length > 0) {
            if (line == this.in.preFenced) {
                this.in.preFenced = ""
                this.html += "</pre>\n"
            } else {
                this.html += `${line}\n`
            }
            return
        }
        if (this.process_table_line(line)) {
            return
        }
        // 3 or more of --- or ___ or *** followed by space
        // some online notes seemed to indicate spaces at end, so accepting same
        if (line.match(/^[-]{3,}|[*]{3,}|[_]{3,}\s*$/) != null) {
            this.unwind_list()
            this.html += "<hr>\n"
            return
        }
        if (line.startsWith ("#")) {
            this.unwind_list()
            let hLevel = lineA[0].length
            this.html += `<h${hLevel}>${line.slice(hLevel)}</h${hLevel}>\n`
            return
        }
        // same number of space followed by ``` or ~~~
        // some samples with spaces at beginning seen, so accepting spaces at begin
        let matchPreFenced = line.match(/^(\s*```|\s*~~~)([a-zA-Z0-9]*)(.*)/);
        if ( matchPreFenced != null) {
            this.unwind_list()
            this.in.preFenced = matchPreFenced[1]
            this.html += `<pre class="${matchPreFenced[2]}">\n`
            return
        }
        if (this.process_list(line)) {
            return
        }
        this.unwind_list()
        this.html += `<p>${line}</p>`
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
