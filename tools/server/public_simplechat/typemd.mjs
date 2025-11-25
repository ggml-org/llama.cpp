//@ts-check
// simple minded helpers to handle markdown content
// by Humans for All
//


export class MarkDown {

    constructor() {
        this.in = {
            preFenced: "",
            table: false,
            /** @type {Array<number>} */
            listUnordered: []
        }
        this.md = ""
        this.html = ""
    }

    unwind_list_unordered() {
        for(const i in this.in.listUnordered) {
            this.html += "</ul>\n"
        }
        this.in.listUnordered.length = 0
    }

    unwind_list() {
        this.unwind_list_unordered()
    }

    /**
     * Process a line from markdown content
     * @param {string} line
     */
    process_line(line) {
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
        // spaces followed by - or + or * followed by a space and actual list item
        let matchUnOrdered = line.match(/^([ ]*)[-+*][ ](.*)$/);
        if ( matchUnOrdered != null) {
            let sList = 'none'
            let listLvl = 0
            if (this.in.listUnordered.length == 0) {
                sList = 'same'
                this.in.listUnordered.push(matchUnOrdered[1].length)
                listLvl = this.in.listUnordered.length // ie 1
                this.html += "<ul>\n"
            } else {
                if (this.in.listUnordered[this.in.listUnordered.length-1] < matchUnOrdered[1].length){
                    sList = 'same'
                    this.in.listUnordered.push(matchUnOrdered[1].length)
                    listLvl = this.in.listUnordered.length
                    this.html += "<ul>\n"
                } else if (this.in.listUnordered[this.in.listUnordered.length-1] == matchUnOrdered[1].length){
                    sList = 'same'
                } else {
                    sList = 'same'
                    while (this.in.listUnordered[this.in.listUnordered.length-1] > matchUnOrdered[1].length) {
                        this.in.listUnordered.pop()
                        this.html += `</ul>\n`
                        if (this.in.listUnordered.length == 0) {
                            break
                        }
                    }
                }
            }
            if (sList == 'same') {
                this.html += `<li>${matchUnOrdered[2]}</li>\n`
            }
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
        let linesA = lines.split('\n')
        for(const line of linesA) {
            this.process_line(line)
        }
    }

}
