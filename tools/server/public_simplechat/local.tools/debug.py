# Helpers for debugging
# by Humans for All


import time

gMe = { '--debug' : False }


def setup(bEnable):
    global gMe
    gMe['--debug'] = bEnable


def dump(meta: dict, data: dict):
    if not gMe['--debug']:
        return
    timeTag = f"{time.time():0.12f}"
    with open(f"/tmp/simplemcp.{timeTag}.meta", '+w') as f:
        for k in meta:
            f.write(f"\n\n\n\n{k}:{meta[k]}\n\n\n\n")
    with open(f"/tmp/simplemcp.{timeTag}.data", '+w') as f:
        for k in data:
            f.write(f"\n\n\n\n{k}:{data[k]}\n\n\n\n")
