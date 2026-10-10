# VAAPI encode on the A770 under xe (2026-09-29, 11:15 GMT)

xe.ko on 7.3.0-rc1 has no DG2 HuC firmware entry (only `i915/dg2_guc_70.bin` and
`dg2_dmc_ver2_08.bin`); under i915 HuC 7.10.16 was RUNNING. Bitrate control (BRC)
runs on the HuC.

`vainfo` failed on both drivers because `/etc/environment` set
`LIBVA_DRIVER_NAME=radeonsi` globally (removed 11:48; libva then auto-selects iHD for
`0000:03:00.0` and radeonsi for the Raphael iGPU). With `LIBVA_DRIVER_NAME=iHD` forced,
iHD 26.2.4 advertises 18 encode entrypoints incl. AV1 `EncSliceLP`.

Real test, 90 frames of `testsrc2` 1920x1080@30, ffmpeg VAAPI on
`/dev/dri/by-path/pci-0000:03:00.0-render`:

| encoder | rc mode | result |
|---|---|---|
| h264_vaapi | CQP qp 24 | OK, 0.4 s, 2.96 MB |
| h264_vaapi | VBR 6M | FAIL `Terminating thread with return code -5 (Input/output error)` |
| hevc_vaapi | CQP qp 24 | OK |
| av1_vaapi | CQP qp 60 | OK, 0.3 s, 5.97 MB |
| av1_vaapi | VBR 4M | FAIL, same error |
| av1_vaapi | CBR 4M | FAIL, same error |

CQP works for all three codecs; VBR/CBR die at submit. `*.err` are the ffmpeg stderr
files. Media outputs not kept. QSV paths untested.
