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
            }
        }
        /**
         * @type {Array<*>}
         */
        this.errors = []
        this.raw = ""
        this.html = ""
    }

    unwind_list_unordered() {
        while (true) {
            let popped = this.in.list.endType.pop()
            if (popped == undefined) {
                break
            }
            this.html += popped
        }
        this.in.list.offsets.length = 0
    }

    unwind_list() {
        this.unwind_list_unordered()
    }

    /**
     * Process a unordered list one line at a time
     * @param {string} line
     */
    process_list_unordered(line) {
        // spaces followed by - or + or * followed by a space and actual list item
        let matchUnOrdered = line.match(/^([ ]*)([-+*]|[a-zA-Z0-9]\.)[ ](.*)$/);
        if (matchUnOrdered != null) {
            let listLvl = 0
            let curOffset = matchUnOrdered[1].length
            let lastOffset = this.in.list.offsets[this.in.list.offsets.length-1];
            if (lastOffset == undefined) {
                lastOffset = -1
            }

            if (lastOffset < curOffset){
                this.in.list.offsets.push(curOffset)
                listLvl = this.in.list.offsets.length
                this.html += "<ul>\n"
                this.in.list.endType.push("</ul>\n")
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

            this.html += `<li>${matchUnOrdered[3]}</li>\n`
            return true
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
        if (this.process_list_unordered(line)) {
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
