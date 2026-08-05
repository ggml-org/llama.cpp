# Packaging

`build-and-notarize.sh` is the canonical "build, sign, notarize, and package"
pipeline for the macOS app. It runs the whole chain end-to-end:

1. Build the C++ engine xcframework (if `TesseraStudio/artifacts/` is empty)
2. `xcodebuild Release` for `TesseraStudioMac` with Developer ID signing +
   hardened runtime
3. `xcrun notarytool submit` + `xcrun stapler staple` for the `.app`
4. `pkgbuild` to wrap the `.app` at `/Applications/TesseraStudio.app`
5. `productbuild --sign` with the Developer ID Installer identity
6. `xcrun notarytool submit` + `xcrun stapler staple` for the `.pkg`

Output lands at `dist/TesseraStudio-<version>.pkg`.

## One-time setup

```sh
# 1. Import your Developer ID certificates (Xcode does this automatically
#    once you sign in with your Apple ID and have a team).
#
# 2. Create a notarytool credential profile. Generate the API key at
#    https://appstoreconnect.apple.com/access/api, then:
xcrun notarytool store-credentials tessera-notary \
    --key <path-to-AuthKey_XXXXXXXXXX.p8> \
    --key-id <KEY_ID> \
    --issuer <ISSUER_ID>
```

## Run it

```sh
./packaging/build-and-notarize.sh              # version 1.0.0, build 1
./packaging/build-and-notarize.sh 1.2.0 42     # version 1.2.0, build 42
SKIP_NOTARIZATION=1 ./packaging/build-and-notarize.sh   # skip the notary tickets
```

## Environment overrides

| Var | Default | Notes |
|---|---|---|
| `DEVELOPER_ID_APP` | auto-detected from keychain | "Developer ID Application: … (TEAMID)" |
| `DEVELOPER_ID_INSTALLER` | auto-detected | "Developer ID Installer: … (TEAMID)" |
| `DEVELOPMENT_TEAM` | inferred from the app identity's team ID | pass-through to xcodebuild |
| `NOTARY_PROFILE` | `tessera-notary` | the keychain profile name |
| `BUNDLE_ID` | `com.tessera.studio.mac` | must match the Xcode target |
| `SKIP_NOTARIZATION` | unset | set to `1` for local builds |
| `OUTPUT_DIR` | `dist/` | relative to repo root |

## Why `.pkg` not `.dmg`

- **Per-machine install** into `/Applications` — no "drag to Applications"
  friction, the user's `~/Applications` situation doesn't matter
- **CLI-installable** with `installer -pkg TesseraStudio-1.0.0.pkg -target /`
  for headless / fleet installs
- **Distribution-signed separately** with the Developer ID Installer
  identity, so the .pkg's seal stays valid even if the .app inside is
  later moved or replaced
- **Right feel** for a developer tool — matches Homebrew/CLI conventions
  better than a drag-install .dmg
