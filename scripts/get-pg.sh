#!/usr/bin/env bash

function usage {
    echo "usage: <n>$0"
    echo "note: n is the number of essays to download"
    echo "for specific n, the resulting pg.txt file will have the following number of tokens:"
    echo "n   | tokens"
    echo "--- | ---"
    echo "1   | 6230"
    echo "2   | 23619"
    echo "5   | 25859"
    echo "10  | 36888"
    echo "15  | 50188"
    echo "20  | 59094"
    echo "25  | 88764"
    echo "30  | 103121"
    echo "32  | 108338"
    echo "35  | 113403"
    echo "40  | 127699"
    echo "45  | 135896"
    exit 1
}

function has_cmd {
    if ! [ -x "$(command -v $1)" ]; then
        echo "error: $1 is not available" >&2
        exit 1
    fi
}

# check for: curl, html2text, tail, sed, fmt
has_cmd curl
has_cmd html2text
has_cmd tail
has_cmd sed

if [ $# -ne 1 ]; then
    usage
fi

n=$1

# get urls
urls="$(curl http://www.aaronsw.com/2002/feeds/pgessays.rss | grep html | sed -e "s/.*http/http/" | sed -e "s/html.*/html/" | head -n $n)"

printf "urls:\n%s\n" "$urls"

if [ -f pg.txt ]; then
    rm pg.txt
fi

c=1
for url in $urls; do
    echo "processing $url"

    cc=$(printf "%03d" $c)

    curl -L $url | html2text | tail -n +4 | sed -E "s/^[[:space:]]+//g" | fmt -w 80 >> pg-$cc-one.txt
    cat pg-$cc-one.txt >> pg.txt

    cp -v pg.txt pg-$cc-all.txt
    c=$((c+1))

    # don't flood the server
    sleep 1
done

echo "done. data in pg.txt"

exit 0
