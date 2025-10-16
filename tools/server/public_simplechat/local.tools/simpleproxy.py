# A simple proxy server
# by Humans for All
#
# Listens on the specified port (defaults to squids 3128)
# * if a url query is got (http://localhost:3128/?url=http://site.of.interest/path/of/interest)
#   fetches the contents of the specified url and returns the same to the requester
#


import sys
import http.server


gMe = {
    '--port': 3128,
    'server': None
}


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        print(self.path)



def process_args(args: list[str]):
    global gMe
    gMe['INTERNAL.ProcessArgs.Malformed'] = []
    gMe['INTERNAL.ProcessArgs.Unknown'] = []
    iArg = 1
    while iArg < len(args):
        cArg = args[iArg]
        if (not cArg.startswith("--")):
            gMe['INTERNAL.ProcessArgs.Malformed'].append(cArg)
            print(f"WARN:ProcessArgs:{iArg}:IgnoringMalformedCommandOr???:{cArg}")
            iArg += 1
            continue
        match cArg:
            case '--port':
                iArg += 1
                gMe[cArg] = int(args[iArg])
                iArg += 1
            case _:
                gMe['INTERNAL.ProcessArgs.Unknown'].append(cArg)
                print(f"WARN:ProcessArgs:{iArg}:IgnoringUnknownCommand:{cArg}")
                iArg += 1


def run():
    try:
        gMe['server'] = http.server.HTTPServer(('',gMe['--port']), ProxyHandler)
        gMe['server'].serve_forever()
    except KeyboardInterrupt:
        print("INFO:Run:Shuting down...")
        if (gMe['server']):
            gMe['server'].server_close()
        sys.exit(0)
    except Exception as exc:
        print(f"ERRR:Run:Exception:{exc}")
        if (gMe['server']):
            gMe['server'].server_close()
        sys.exit(1)


if __name__ == "__main__":
    process_args(sys.argv)
    run()
