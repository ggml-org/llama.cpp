## MTMD Hang Test

MTMD video hang test in windows related for issue [#24429](https://github.com/ggml-org/llama.cpp/issues/24429).

### What's the problem?
1. `mtmd_helper_video` - `subprocess_handle` - `stop` checks `alive`, and if `alive=false`, it skips terminating and destroying process.
2. `feeder` sends video buffer to ffmpeg stdin.
3. video reader reads video frames from ffmpeg stdout.
4. ffmpeg process can still be alive after video reader reads EOF from stdout, pipe handles may remain opened.
5. `feeder.join()` can block because `feeder` is blocked in `fwrite()`. 

## Testing
1. Build llama.cpp CLI
2. Run test via
```sh
node ./mtmd-hang-test.ts <CLI exe path>
```
3. See result like
```
...

Exiting...
Process exited with 0

##########################
Test result: success!
##########################
```

------
* Failed case before fix:

```
Cli: <workspace>\build-before-fix\bin\llama-cli.exe
Check: model is valid.
Check: mmproj is valid.


Executing llama-cli...


Llama-cli timeout for 120 sec.
```

* `build-before-fix` commit: `5fff128451d7603857597ee1fc18ac1dfb90f148`

### Test Environment
* ffmpeg, ffprobe: `9.0 full`
* nodejs: `v26.1.0`
* windows: Windows 11 26H2 (26300.9168)

## `circle.mp4` source
 * Video: MS PowerPoint.
 * Audio: generated with ffmpeg. No third-party audio is used.
 ```sh
 ffmpeg -f lavfi -i "aevalsrc=(0.5+0.5*sin(2*PI*2*t))*(0.20*sin(2*PI*261.63*t)+0.16*sin(2*PI*329.63*t)+0.12*sin(2*PI*392*t)+0.08*sin(2*PI*523.25*t)):s=44100:d=15" -ac 2 -c:a aac -b:a 192k buz.m4a
 ffmpeg -i circle-noaudio.mp4 -i buz.m4a -shortest -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac circle.mp4
 ```