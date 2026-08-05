# Tessera Studio: "Plead the Fifth" Design Specification

**Status:** Draft v1 — 2026-08-05
**Author:** Tessera Architecture
**Applies to:** Tessera Studio for macOS 1.0.0+ (post-Materials slice)

---

## 1. Executive Summary

Tessera Studio ships a **"Plead the Fifth"** command that, on user invocation, renders all of Tessera's data unrecoverable to anyone — including the user, the operating system, the SSD firmware, the law enforcement forensic lab, the SSD manufacturer, or the cloud. The command is named for the Fifth Amendment right against compelled self-incrimination and is the architectural expression of the Fourth Amendment right to be secure in one's papers against unreasonable search and seizure.

It is technically implemented as **crypto-shred**: a key destruction that makes all of Tessera's encrypted data cryptographically unrecoverable. The wipe is permanent, repeatable, and survives forensic recovery tools (PhotoRec, TestDisk, EnCase, Cellebrite, Cellebrite UFED) because the underlying technical premise — that you cannot reliably overwrite individual files on an SSD due to the flash translation layer — is the very reason crypto-shred is the only sound answer on modern storage.

The feature ships in three surfaces:

1. **Hot-key** (`Cmd+Shift+Backspace`) for direct, instant destruction.
2. **Menu bar item** with a typed-phrase confirmation for the standard UX.
3. **Covert trigger** — a user-chosen phrase typed anywhere in the app — for coercion scenarios where the user is being watched.

External security audit is non-negotiable; the marketing claim "unrecoverable by anyone under any circumstances" is audited before public release. A feature whose entire pitch is "we promise it works" that ships without an external audit is theater.

---

## 2. Goals and Non-Goals

### 2.1 Goals

1. **Make Tessera's data cryptographically unrecoverable** on user demand, within a time budget of **≤ 2 seconds** for a hot-key wipe and **≤ 10 seconds** for a full encrypted-volume overwrite.
2. **Survive forensic recovery** by any tool available today or in the foreseeable future (NIST SP 800-88 r1 "Purge" and "Destroy" methods).
3. **Be triggerable under coercion** without the adversary's awareness, via the covert trigger phrase.
4. **Be auditable** — the feature is reviewed by an external security firm before public release; the audit result is published.
5. **Coexist with the rest of the privacy story** — the local-first architecture, no cloud, no telemetry, no third-party data custody.
6. **Be reversible for legitimate recovery** — the user can re-enter the volume password and re-create the data, so a panic wipe isn't a self-DDOS for the user. The user is in control of the password; if they have it, they can re-mount the volume; if they don't, the data is gone.

### 2.2 Non-Goals

1. **Anti-forensics against nation-state attackers with physical access to the running Mac.** If the attacker has root on the running machine, they can read the plaintext data from RAM before, during, or after the wipe. The threat model assumes the Mac is not currently compromised. (For that threat, see [Section 3.4 — out of scope].)
2. **A wipe that survives backup systems the user didn't consent to.** If the user has Time Machine or iCloud Drive backing up `~/Library/Containers/com.tessera.studio.mac/`, the encrypted data is in those backups. The wipe covers the live data on the user's Mac only. Time Machine and iCloud Drive behavior is documented but is the user's responsibility to manage.
3. **Wiping the user's GGUF model files.** Models are the user's property, not Tessera's data. The model file does not reveal what the user did with it. The wipe targets Tessera's data (Postgres, Valkey, DuckDB, SwiftData, materials cache, screenshots) — not the user's model collection.
4. **Networked destruction across devices.** The wipe is local. A future v2 might add a "destroy on all devices" signal for Mesh devices, but that's a separate design.
5. **A "kill switch" that the user can't see.** The hot-key and menu bar trigger are visible. The covert trigger is opt-in (user sets a phrase in Settings) and documented in the user-visible privacy page. We do not ship hidden triggers.
6. **Defense against rubber-hose cryptanalysis.** If the user is physically compelled to reveal the volume password, they can choose to reveal it. "Plead the Fifth" doesn't prevent compulsion; it provides a destruction path that the user can use *before* compulsion becomes irresistible. The covert trigger is the design's response to this, but no software is a complete answer to physical coercion.

---

## 3. Threat Model

### 3.1 Adversary Profiles

| Adversary | Capability | Goal |
|---|---|---|
| **Curious employer / school IT** | Administrator access to the user's work machine; can install monitoring software | Read Tessera materials to learn what the user is working on |
| **Domestic abuser** | Physical access to the user's unlocked Mac; emotional leverage to compel unlocking | Find private notes, conversations, materials to use against the user |
| **Civil litigant** | Subpoenas, court orders | Discover materials to use in litigation |
| **Law enforcement** | Search warrants, pen-register orders, compelled decryption (with varying legal thresholds across jurisdictions) | Access Tessera materials as evidence |
| **National security agency** | NSL (national security letter), FISA orders, sometimes gag-attached | Bulk access to materials |
| **Corporate competitor** | Industrial espionage, supply-chain compromise | Steal proprietary code or materials |
| **Foreign state actor** | Sophisticated, well-resourced | Long-term access, exfiltration |

### 3.2 Attack Scenarios

**Scenario A — Compelled disclosure at a border crossing.**
A user crosses a US border with a MacBook containing Tessera materials related to confidential journalistic work. CBP agents demand the user unlock the Mac. The user types the covert trigger phrase into the agent's text input while pretending to comply. The wipe begins silently. The agent sees an app with no materials, no chat history, no library. The user "cooperates" with the unlock. The agent finds nothing Tessera-related and either lets the user through or seizes the hardware. If the hardware is seized, the Tessera data is unrecoverable from the SSD via forensic tools; what's left is the encrypted volume file plus the SSD firmware, both of which require the volume password to decrypt.

**Scenario B — Targeted device seizure at the user's home.**
Police execute a search warrant. The user, anticipating this, types the hot-key `Cmd+Shift+Backspace` while the agents are at the door. The wipe completes in under 2 seconds. By the time the agents take the Mac, Tessera's data is gone. The agents may seize the hardware, but the encrypted volume is unrecoverable without the password.

**Scenario C — Subpoena to the user (not to Tessera).**
A civil litigant subpoenas the user for documents. The user, who has not been compelled to produce the password, chooses to destroy the documents before the court date. The user triggers the menu bar → "Plead the Fifth" with the typed-phrase confirmation. The wipe completes. The user appears in court with the hardware; the hardware contains no Tessera data; the user is in compliance (they're not withholding documents, they have nothing to produce). Whether the court accepts this is a separate legal question; the technical capability is the user's right.

**Scenario D — Employer / IT admin reads the disk while the user is logged out.**
FileVault (or per-volume APFS encryption) means the data is encrypted at rest when the volume is unmounted. When the user is logged out, the volume is unmounted. The data is cryptographically protected. This is "free" defense; it comes with the architecture.

**Scenario E — Forensic lab obtains the SSD after the user has been compelled to surrender it.**
The lab uses PhotoRec, EnCase, or similar tools to attempt recovery. The data is encrypted. Without the volume password, the encrypted blobs are unrecoverable. The user, if compelled, can surrender the password or not; either way, the *crypto-shred* path (where the user destroyed the key before surrender) makes the data cryptographically unrecoverable. This is the key design property.

### 3.3 What This Design Defends Against

- **Logical forensic recovery from the live SSD after a wipe.** Defense in depth: crypto-shred (key destroyed) + ciphertext overwrite (random data) + fsync. Survives PhotoRec, TestDisk, EnCase, Cellebrite UFED.
- **Compelled disclosure via the application's visible state.** The covert trigger lets the user destroy data without the adversary seeing the trigger. The menu bar item can be hidden in "coercion mode" (see [Section 9.5](#95-coercion-mode-display)).
- **Application-layer obfuscation.** The data on disk is encrypted; even if the file is intact, it's not readable without the key.
- **Backup-system leakage (Time Machine, iCloud Drive).** Out of scope (see [Section 2.2](#22-non-goals) #2). Documented to the user; the user can manage.

### 3.4 What This Design Does NOT Defend Against

- **Root on the running Mac.** If the Mac is compromised at runtime, the attacker reads the volume's plaintext from RAM. "Plead the Fifth" runs in user space; it cannot protect against a kernel-level attacker. Users concerned about this threat should pair Tessera with a secure-boot, kernel-integrity-verified macOS install and never leave the Mac unattended while the volume is mounted.
- **Compulsion to reveal the volume password.** The user can be legally compelled to reveal the password (jurisdiction-dependent). The covert trigger is the design's response — destroy before compulsion becomes irresistible — but software cannot prevent a determined coercer with physical access.
- **Memory forensics after the wipe.** RAM contents are not addressed by the design; RAM is volatile and disappears at power loss, but a hibernation file on disk (`/private/var/vm/sleepimage`) may contain pre-wipe plaintext. FileVault covers the hibernation file when the Mac is off. While the Mac is on, the volume is mounted and the hibernation file is encrypted at the FileVault level.
- **A future breakthrough in AES-256 cryptanalysis.** Practically zero risk on the 10-20 year horizon, but documented as a residual. The ciphertext-overwrite step (defense in depth) addresses this concern.
- **SSD firmware vulnerabilities that leak data after cryptographic erase.** These have been demonstrated in research papers (e.g., "Self-encrypting deception: weaknesses in the encryption of solid-state drives" by Cai et al., 2018). Mitigated by the per-volume APFS encryption (Apple's implementation, not SSD firmware), which uses the Secure Enclave's key storage rather than relying on the SSD controller.

---

## 4. Legal & Regulatory Context

This section is for the design's reference. It is not legal advice; the user should consult counsel for their jurisdiction. The references are real and verifiable.

### 4.1 United States

#### Constitutional

- **Fourth Amendment.** "The right of the people to be secure in their papers and effects, against unreasonable searches and seizures, shall not be violated." Crypto-shred is the architectural expression of this right in software: the user's "papers" are protected by key destruction, not by hoping the search is "reasonable."
- **Fifth Amendment.** "No person ... shall be compelled in any criminal case to be a witness against himself." The "Plead the Fifth" name is intentional. The feature protects the user's right against compelled self-incrimination by providing a destruction path the user can invoke before compulsion becomes irresistible. The legal status of destroying evidence is a separate question (see [Section 4.1.4](#414-compelled-decryption-and-the-foregone-conclusion-doctrine)).

#### Attorney-Client Privilege

- **ABA Model Rule 1.6 (Confidentiality of Information).** Requires lawyers to "make reasonable efforts to prevent the inadvertent or unauthorized disclosure of, or unauthorized access to, information relating to the representation of a client." Local-first architecture satisfies this by design: there is no third party to disclose to.
- **ABA Formal Opinion 477R (2017) — Ethical Obligations Related to Internet of Things and Other Networked Devices.** Addresses lawyer obligations when using third-party cloud services. While it permits cloud use with reasonable precautions, it recognizes that **local-only software eliminates the third-party risk entirely**. Tessera's architecture maps directly to the "no third party" path.
- **State bar opinions on AI.** Multiple state bars have issued opinions on lawyer use of AI, all converging on: (a) local processing is preferred for privileged material; (b) cloud processing without a BAA or equivalent is risky; (c) the lawyer is responsible for the technical safeguards.

#### HIPAA

- **45 CFR Part 160 (General Administrative Requirements).** Requires covered entities to safeguard PHI.
- **45 CFR Part 164 (Security Rule).** §164.312(a)(2)(iv) requires "encryption and decryption" of PHI at rest as an addressable implementation specification. APFS encryption (FileVault or per-volume) satisfies this.
- **No BAA required for local processing.** When PHI never leaves the user's device, no business associate relationship exists, and no BAA is required. (See Elephas' analysis of healthcare attorney use cases for the parallel argument.)
- **HIPAA Safe Harbor (§164.514(b)(2)).** De-identification of PHI to the Safe Harbor standard satisfies the disclosure rule. Local-first AI doesn't transmit PHI in the first place, but if any data is exfiltrated, it's encrypted.

#### FERPA

- **20 U.S.C. § 1232g; 34 CFR Part 99.** Protects student education records. Cloud AI services must contractually agree to FERPA compliance; local-first services do not transmit the data and have no FERPA exposure.

#### Compelled Decryption and the "Foregone Conclusion" Doctrine

- **United States v. Boucher, 2009.** The "foregone conclusion" doctrine: the government can compel decryption if it can prove it knows the contents of the encrypted data. A user can be compelled to produce a password if the government already has evidence of what's on the drive.
- **United States v. Kirschner, 2010.** Confirmed that the Fifth Amendment can be invoked against compelled decryption in some cases.
- **In re Grand Jury Subpoena (11th Circuit, 2012).** The court found the act of decryption itself is testimonial, but the "foregone conclusion" exception applies if the government already knows.
- **Practical effect:** the user should not rely on the Fifth Amendment alone to resist compelled decryption. The "Plead the Fifth" architecture protects the user by making the data unrecoverable *before* compulsion. The legal exposure is then "did you destroy evidence?" not "what's on the drive?" — a different and stronger legal posture.

### 4.2 European Union

- **GDPR Article 32 (Security of Processing).** Requires "pseudonymisation and encryption" of personal data. APFS encryption satisfies the encryption requirement. Crypto-shred is the "right to be forgotten" (Article 17) at the technical level.
- **Schrems II (CJEU, 2020).** Invalidated the US-EU Privacy Shield; requires Standard Contractual Clauses or other legal bases for transatlantic data transfer. Local-first processing sidesteps the entire transatlantic data transfer question.
- **GDPR Article 25 (Data Protection by Design and by Default).** Requires privacy to be baked into the design. Crypto-shred and local-first are textbook examples of "data protection by design."

### 4.3 Industry-Specific

- **Finance** — GLBA Safeguards Rule (16 CFR Part 314), SOX, SEC Regulation S-P. All require "reasonable safeguards" for nonpublic personal information; local-first + APFS encryption satisfies.
- **Government contractors** — FedRAMP, FIPS 140-2, NIST 800-171 (for Controlled Unclassified Information). Local-first is a credible FedRAMP alternative for AI inference layers; data never leaves the controlled environment.
- **Journalism** — Source protection under state shield laws and the First Amendment. The "Plead the Fifth" command directly addresses the compelled disclosure of sources scenario.

### 4.4 The Legal Asymmetry

Encryption is legal in all US jurisdictions and in the EU. Crypto-shred is the destruction of a key the user controls, not the destruction of another's property. The user has the right to delete their own data.

The gray area is **spoliation** — the destruction of evidence after a litigation hold or in anticipation of litigation. This is a separate legal question that turns on intent, timing, and the user's specific circumstances. The "Plead the Fifth" command is a tool the user can choose to use; it is not a license to destroy evidence relevant to active litigation. The user is responsible for understanding their own legal obligations. Tessera's design does not provide a destruction-without-consequence guarantee; it provides a destruction capability.

---

## 5. Technical Background

### 5.1 Why SSD Overwrite Doesn't Work

Modern SSDs use a **flash translation layer (FTL)** that maps logical block addresses (LBAs) to physical NAND locations. When the OS writes to LBA 100, the FTL may write to physical block 5,000 instead — the original block 100's data may or may not be erased, depending on the SSD's garbage collection, wear leveling, and over-provisioning strategy.

Implications for data destruction:
- `srm` (secure remove), `shred`, `dd if=/dev/urandom of=file`, and similar overwrite utilities **cannot guarantee** the original data is gone.
- The SSD controller may keep copies for wear leveling; those copies are inaccessible to the OS but recoverable by the manufacturer or a forensic lab with firmware access.
- Apple's documentation is explicit: **"Per Apple, Secure Delete does not work on SSD devices, and accordingly has been removed."** The "Security Options" slider in Disk Utility is HDD-only.

This is a **technical fact**, not a vendor limitation. It applies to all consumer and enterprise SSDs as of 2025.

### 5.2 NIST SP 800-88 r1 — The Standard of Care

NIST Special Publication 800-88 r1 ("Guidelines for Media Sanitization") defines three sanitization methods:

1. **Clear.** Logical overwrite, suitable for HDDs in non-sensitive contexts. Not sufficient for SSDs.
2. **Purge.** Physical or logical techniques that render data recovery infeasible using state-of-the-art laboratory techniques. **Cryptographic erase is the recommended Purge method for SSDs.** The drive is encrypted with a strong key; destroying the key renders the data unrecoverable.
3. **Destroy.** Physical destruction (shredding, pulverization, degaussing for magnetic media). NAID AAA certified providers for SSDs.

For software, **Purge via cryptographic erase is the only practical method** on SSDs. "Plead the Fifth" implements NIST SP 800-88 r1 "Purge" for the application's data.

### 5.3 APFS Encryption on macOS

Apple File System (APFS) supports per-volume encryption. Each encrypted APFS volume has its own encryption key, wrapped by a key derived from the user's password. On Apple Silicon (M1+), the volume key is stored in the Secure Enclave, a hardware security module that performs cryptographic operations and key derivation without exposing the key to user space.

Key properties:
- **Hardware-backed key storage.** The volume key never leaves the Secure Enclave in plaintext. The OS reads the key for I/O but cannot export it.
- **Fast cryptographic erase.** Unmounting the volume and destroying the volume key is the equivalent of `diskutil apfs deleteVolume` followed by deleting the encryption metadata. Effectively zero-time.
- **Defense against SSD firmware vulnerabilities.** Because the key is in the Secure Enclave, not the SSD controller, the SSD firmware attack class (Cai et al., 2018) does not apply.
- **APFS is the standard filesystem on modern Macs.** The architecture is supported on every Mac shipped with macOS 10.13+ (High Sierra, 2017).

### 5.4 macOS Keychain as a Key Vault

The macOS Keychain is the system-wide credential store. For the "Plead the Fifth" architecture, the Keychain stores the volume password (or the key that wraps the volume key). The Keychain itself is protected by the user's login password and, on Apple Silicon, by the Secure Enclave.

Properties relevant to "Plea the Fifth":
- **`kSecAttrAccessibleWhenUnlockedThisDeviceOnly`** ensures the item is only available when the user is logged in, and is not synced to iCloud. This is the correct accessibility class for sensitive credentials.
- **`SecItemDelete`** is the API for key destruction. It is atomic and synchronous. After `SecItemDelete` returns success, the Keychain no longer holds the item; subsequent reads return `errSecItemNotFound`.
- **No key escrow.** Apple does not have access to the user's Keychain. The user's data is not recoverable by Apple, by law enforcement via Apple, or by anyone other than the user.

### 5.5 Reference Architectures

The crypto-shred pattern is established. Implementations across the privacy-first software world:

- **1Password.** Encrypted vaults stored as `.agilekeychain` (legacy) or `.opvault` (current) files. The vault's master key is derived from the user's master password. A user who forgets the master password cannot recover their vault. This is crypto-shred by accident (the user destroyed the key by forgetting the password).
- **Signal Desktop.** Stores the encrypted message database in an encrypted SQLite file. The encryption key is derived from a random key stored in the OS keychain. Wiping the keychain entry makes the database unrecoverable.
- **Tails OS.** The "Encrypted Persistent Storage" feature uses LUKS-encrypted volumes. Wiping the LUKS header destroys the key, making the volume unrecoverable.
- **Mullvad VPN.** The "panic mode" / "always-on" features include a kill switch. The threat model is different (network compromise, not local device seizure) but the wipe pattern is the same.

The Tessera architecture is in this lineage. The novelty is the integration with the "Plead the Fifth" trigger surfaces (hot-key, menu bar, covert trigger) and the constitutional framing.

---

## 6. Architecture

### 6.1 Overview

```
┌─────────────────────────────────────────────────────┐
│  macOS user session                                 │
│                                                     │
│  ┌──────────────┐    mount on launch    ┌─────────┐  │
│  │ Tessera      │ ──────────────────▶ │ APFS    │  │
│  │ app process  │                      │ volume  │  │
│  │              │                      │ (encr.) │  │
│  └──────────────┘                      └─────────┘  │
│         │                                    │      │
│         │ writes/reads                      │      │
│         ▼                                    ▼      │
│  ┌──────────────┐                      ┌─────────┐  │
│  │ Postgres     │ ◀────────────────────│  data   │  │
│  │ Valkey       │   on the volume      │  dir    │  │
│  │ DuckDB       │                      │  (encr.) │  │
│  │ SwiftData    │                      │         │  │
│  └──────────────┘                      └─────────┘  │
│                                                ▲    │
│                                                │    │
│                                  ┌─────────────┴─┐  │
│                                  │ macOS Keychain │  │
│                                  │                │  │
│                                  │ volume password│  │
│                                  │ + DAK (opt)    │  │
│                                  │ + WK (opt)     │  │
│                                  └────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### 6.2 Data Inventory — What Lives in the Encrypted Volume

| Data | Volume location | Encrypted at rest | Notes |
|---|---|---|---|
| SwiftData store (conversations, settings) | `~/Library/Containers/com.tessera.studio.mac/Data/Library/Application Support/TesseraStudio/*.store` | Yes (volume-level APFS encryption) | Migrated from current location on next launch |
| Postgres data dir (post-Materials slice) | `<volume>/postgres/data` | Yes | Initdb-time encryption via init script |
| Valkey data dir | `<volume>/valkey/data` | Yes | RDB + AOF inside the volume |
| DuckDB file | `<volume>/duckdb/tessera.duckdb` | Yes | Single file, encrypted at the volume level |
| Materials cache, screenshots, captured snippets | `<volume>/cache/...` | Yes | |
| PrivacyInfo.xcprivacy | App bundle (`TesseraStudio.app/Contents/Resources/`) | No (it's a public manifest) | Already not sensitive |
| tessera.xcframework | App bundle | No (public binary) | Already not sensitive |

### 6.3 What Stays Outside the Volume

| Data | Location | Why outside |
|---|---|---|
| User's GGUF model files | `~/Models/tessera/*.gguf` (user-configured) | The user's property, not Tessera's data. The model file does not reveal what the user did with it. The wipe targets Tessera's data, not the user's collection. |
| The app binary itself | `TesseraStudio.app` | The app is public. |
| The xcframework | Same | The C++ engine is public. |
| App entitlements | Same | Public. |
| Settings, audience mode | Inside the volume (encrypted) | Tells the app which mode to be in. |

### 6.4 Key Hierarchy

Three layers, each with a clear role:

| Layer | Storage | Purpose | Lifetime |
|---|---|---|---|
| **VP (Volume Password)** | macOS Keychain (`kSecAttrAccessibleWhenUnlockedThisDeviceOnly`) | The password that decrypts the APFS volume on mount | Per user, per device. Created on first launch, optionally rotated |
| **DAK (Data Access Key)** (optional) | macOS Keychain | A key that encrypts per-file data inside the volume. Provides per-data-type key isolation and faster per-file rotation | Per user, per device. Auto-generated, auto-rotated |
| **WK (Wrapping Key)** (optional) | macOS Keychain | A key that wraps the DAK. Allows DAK rotation without rewriting all data | Per user, per device |

**Minimum viable:** only the VP is required. The volume-level APFS encryption protects all data inside the volume. The DAK and WK are layered on top for additional defense in depth and for per-file key isolation (e.g., rotating the DAK for materials without rotating the constitutional-receipts key).

**Recommended:** VP + DAK. The DAK adds the ability to rotate keys for individual data types (materials vs. receipts vs. cache) without losing access to other data. The WK is overkill for v1; defer to v2.

### 6.5 Mount Lifecycle

```
First launch:
  1. Generate VP (32 random bytes, base64-encoded; or user-chosen with entropy check)
  2. Store VP in Keychain with kSecAttrAccessibleWhenUnlockedThisDeviceOnly
  3. Create encrypted APFS volume at <user-configured path> using VP
  4. Mount the volume
  5. Initialize Postgres, Valkey, DuckDB, SwiftData inside the volume
  6. App is fully operational

Subsequent launch:
  1. Read VP from Keychain
  2. Mount the volume (APFS prompts for password if not in Keychain; since we have it, mount is automatic)
  3. Postgres, Valkey, DuckDB, SwiftData come online pointing at the volume
  4. App is fully operational

Quit:
  1. Postgres: SIGTERM, wait, SIGKILL after 5s grace
  2. Valkey: same
  3. DuckDB: close file handle
  4. SwiftData: close container
  5. Unmount the volume
  6. App exits
  7. The encrypted volume file remains on disk, encrypted with VP
  8. Without VP (Keychain entry), the volume is unmountable and the data is cryptographically unrecoverable

Plead the Fifth:
  1. SIGTERM Postgres, Valkey (5s grace, then SIGKILL)
  2. Close DuckDB, SwiftData
  3. Destroy VP in Keychain (SecItemDelete)
  4. (If DAK present) Destroy DAK in Keychain
  5. Unmount the volume (will fail since the key is gone, but that's the point — the data is now unrecoverable)
  6. Overwrite the volume file with random data, 3 passes (defense in depth)
  7. fsync
  8. Delete the volume file (rm)
  9. fsync
  10. Exit the app
```

### 6.6 Fail-Closed Design

The architecture is fail-closed:

- **Volume mount failure on launch:** the app cannot start. It displays a clear error: "Tessera's encrypted data could not be loaded. Either the volume is corrupted or the key is missing. If you triggered 'Plea the Fifth' or restored from a backup that did not include the Keychain entry, this is expected. Reinstall Tessera to start fresh."
- **VP destruction failure:** the wipe continues. The data is already cryptographically unrecoverable once the VP is gone; a failure to delete the encrypted blobs is a defense-in-depth concern, not a correctness concern.
- **Volume overwrite failure (e.g., disk full):** log the failure, continue to step 8 (delete). The crypto-shred property holds regardless of the overwrite.
- **Postgres / Valkey hang on shutdown:** SIGKILL after 5 seconds. The wipe continues.

### 6.7 The User's Recovery Path

The user is the only person who can recover their data. The recovery path:

- The user has (or has lost) the volume password. If they have it, they can re-mount the volume and re-create the data structures inside.
- The user can recover the volume password from a password manager. If the password is only in Tessera's Keychain, and Tessera's Keychain is gone, the password is gone.
- **Recommendation (in-app, in the user-visible privacy page):** "Tessera recommends that you also store your volume password in 1Password / Bitwarden / Apple Passwords. If you lose the volume password AND the Keychain entry, your data is unrecoverable. This is intentional."

---

## 7. The Wipe Procedure

### 7.1 The Steps

The wipe is implemented as an ordered sequence of steps. Each step reports success/failure. The wipe continues even on partial failure, because the crypto-shred property is achieved at step 3 (destroying the Keychain entry); subsequent steps are defense in depth.

```swift
public actor PleadTheFifthExecutor {
    public struct WipeReport: Sendable {
        public let startedAt: Date
        public let completedAt: Date
        public let steps: [WipeStep]
        public var succeeded: Bool { steps.allSatisfy { $0.outcome == .success } }
    }

    public enum WipeOutcome: Sendable {
        case success
        case partialFailure(reason: String)
        case aborted(reason: String)
    }

    public struct WipeStep: Sendable {
        public let name: String
        public let outcome: WipeOutcome
        public let durationMs: Int
    }

    public func destroyAll() async throws -> WipeReport {
        let startedAt = Date()
        var steps: [WipeStep] = []

        // Step 1: stop the Postgres process.
        steps.append(time("stop_postgres") {
            try await ProcessRunner.terminate(name: "tessera-postgres", timeout: .seconds(5))
        })

        // Step 2: stop the Valkey process.
        steps.append(time("stop_valkey") {
            try await ProcessRunner.terminate(name: "tessera-valkey", timeout: .seconds(5))
        })

        // Step 3: destroy the volume password in Keychain.
        // This is the crypto-shred event. After this step, the data
        // on disk is cryptographically unrecoverable, regardless of
        // what happens to steps 4-10.
        steps.append(time("destroy_volume_password") {
            try await TesseraKeychain.deleteVolumePassword()
        })

        // Step 4: destroy the DAK (if present).
        steps.append(time("destroy_dak") {
            try? await TesseraKeychain.deleteDAK()
        })

        // Step 5: unmount the volume.
        // Will fail (the key is gone) but we attempt it anyway. The
        // unmount call uses the cached volume handle; the OS will
        // refuse the operation.
        steps.append(time("unmount_volume") {
            try? await APFSVolume.unmount(handle: cachedVolumeHandle)
        })

        // Step 6: overwrite the encrypted blobs with random data.
        // Defense in depth: even if a future attack breaks AES-256
        // (not realistic on the 10-20 year horizon, but documented),
        // the original ciphertext is gone.
        steps.append(time("overwrite_ciphertext") {
            try await SecureOverwrite.randomPasses(
                paths: encryptedVolumePaths,
                passes: 3
            )
        })

        // Step 7: delete the volume files.
        steps.append(time("delete_volume_files") {
            try await SecureDelete.delete(paths: encryptedVolumePaths)
        })

        // Step 8: fsync the parent directory.
        steps.append(time("fsync") {
            try await fsyncParentDirectory(of: encryptedVolumePaths.first!)
        })

        // Step 9: exit the app.
        // Process exit is the only way to ensure no plaintext state
        // remains in memory. The OS reclaims RAM at process exit.
        steps.append(time("exit") {
            // Defer the actual exit so the report is logged.
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                exit(0)
            }
        })

        return WipeReport(
            startedAt: startedAt,
            completedAt: Date(),
            steps: steps
        )
    }
}
```

### 7.2 Time Budget

| Step | Expected time | Hard limit |
|---|---|---|
| stop_postgres | 200ms | 5s (SIGKILL) |
| stop_valkey | 200ms | 5s (SIGKILL) |
| destroy_volume_password | 50ms | 1s |
| destroy_dak | 50ms | 1s |
| unmount_volume | 100ms | 1s |
| overwrite_ciphertext | 800ms-2s (3 passes, ~50MB volume) | 10s |
| delete_volume_files | 50ms | 1s |
| fsync | 50ms | 1s |
| exit | 100ms (deferred) | 1s |
| **Total** | **~1.5-3s** | **30s** |

The hot-key wipe target is ≤ 2 seconds for the cryptographically-relevant steps (1-3). Defense-in-depth steps (4-10) can run in the background while the app exits.

### 7.3 What Survives (and Why)

| What | Survives? | Why |
|---|---|---|
| The encrypted volume file on disk | **Until step 7** | Crypto-shred makes it unrecoverable. Steps 6-7 destroy it for defense in depth. |
| The unencrypted data on disk | **No** | The unencrypted data was never on disk; it was only in RAM and on the live filesystem (in memory-mapped pages). When the app exits, RAM is reclaimed. |
| Plaintext in RAM | **No (after process exit)** | Process exit reclaims RAM. The OS may have swapped some pages to disk; those pages were encrypted at the FileVault level (the swap partition is FileVault-encrypted on modern macOS). |
| Plaintext in the hibernation file | **No (when Mac is off)** | The hibernation file is on the FileVault-encrypted root volume. When the Mac is off, it's encrypted. When the Mac is on, the volume is mounted — see next row. |
| Plaintext on the live filesystem (with the volume mounted) | **Yes, while the Mac is on and the user is logged in** | This is the threat model that "Plead the Fifth" addresses. The user invokes the wipe before the threat materializes, or in the moment. |
| Backup copies (Time Machine, iCloud Drive) | **Yes, if the user has backups** | Out of scope (see [Section 2.2](#22-non-goals) #2). The user is responsible for managing their own backup strategy. |
| The volume's APFS metadata | **Yes, until step 7** | Same as the encrypted volume file. |

### 7.4 The Report

The wipe produces a `WipeReport` with a step-by-step record. The report is written to a fixed location in the user's home (e.g., `~/.tessera/last-wipe.json`) **before** the app exits. This is for the user's audit trail.

```json
{
  "startedAt": "2026-08-05T12:34:56Z",
  "completedAt": "2026-08-05T12:34:58Z",
  "triggerSource": "hotkey",
  "steps": [
    { "name": "stop_postgres", "outcome": "success", "durationMs": 220 },
    { "name": "stop_valkey", "outcome": "success", "durationMs": 180 },
    { "name": "destroy_volume_password", "outcome": "success", "durationMs": 45 },
    { "name": "destroy_dak", "outcome": "success", "durationMs": 38 },
    { "name": "unmount_volume", "outcome": "partialFailure", "durationMs": 110, "reason": "expected: key destroyed" },
    { "name": "overwrite_ciphertext", "outcome": "success", "durationMs": 1840 },
    { "name": "delete_volume_files", "outcome": "success", "durationMs": 47 },
    { "name": "fsync", "outcome": "success", "durationMs": 52 },
    { "name": "exit", "outcome": "success", "durationMs": 100 }
  ],
  "succeeded": false,
  "note": "Step 5 partial failure is expected and does not affect recoverability"
}
```

The report is JSON, plain text, no secrets. The user can post it to a court, an auditor, or a journalist as evidence of the wipe.

---

## 8. The Three Trigger Surfaces

### 8.1 Hot-Key: `Cmd+Shift+Backspace`

The default hot-key for direct, instant destruction. The choice of `Cmd+Shift+Backspace` is deliberate:
- `Cmd+Shift+?` (Help) and `Cmd+Shift+,` (Settings) are common app hot-keys; `Cmd+Shift+Backspace` is unclaimed and unlikely to conflict.
- Backspace is unambiguous: nobody presses it accidentally as a chord.

The hot-key is global — it works whether the app is foreground, background, or the user is in another app. This is implemented via `NSEvent.addGlobalMonitorForEvents(matching:handler:)` registered on launch.

Behavior:
- No confirmation. The user pressed the key. The wipe begins.
- The wipe runs to completion regardless of UI state.
- The app exits within 3 seconds.

### 8.2 Menu Bar: "Plead the Fifth…"

A `NSStatusItem` in the macOS menu bar. Click → submenu:

- **Plead the Fifth…** — opens a confirmation dialog requiring the typed phrase `destroy everything` (case-insensitive, no copy-paste, with a 5-second rate limit on failed attempts to defeat brute force).
- **Plead the Fifth (covert)** — visible only if the user has set a covert trigger phrase in Settings. Submenu shows the current phrase and a "Test" button.
- **Last wipe report…** — opens the last wipe report JSON for the user's audit trail.

The menu bar item is **visible by default**. There is a Settings toggle to hide it ("Coercion mode: hide the Plead the Fifth menu item"). When hidden, only the hot-key and the covert trigger are available.

### 8.3 Covert Trigger Phrase

The covert trigger is a user-chosen phrase typed anywhere in the app. Examples (DO NOT use these — they're in the documentation):
- `the weather is nice today`
- `i plead the fifth`
- `paris in the spring`
- `<some random string the user invents>`

Settings UX:
- "Covert trigger phrase (advanced):"
- Text field, 8+ characters, with a hint: "Choose something you can type naturally and that an adversary wouldn't think to look for. Don't choose a famous quote or a phrase from a movie."
- A "Test" button: pressing it simulates the trigger without executing the wipe. Shows "OK, the trigger would have fired. Make sure the phrase isn't likely to come up in your normal use."
- A 5-second cooldown after a successful trigger to prevent accidental double-wipes.

The trigger is checked in every text input in the app:
- Playground chat input
- Library search
- Materials search
- Settings text fields
- The capture floating window
- Any other text-input view

The check is case-insensitive and substring-based (the phrase can appear as part of a longer string). This is intentional: the user typing a long message that happens to contain the trigger phrase is the expected UX, and a false-positive wipe is no worse than an intentional one (the user can re-set up the app from scratch either way; see [Section 7.4](#74-the-report) — the wipe report is preserved as a JSON file).

### 8.4 The Coercion Mode

A Settings toggle, off by default. When on:
- The "Plead the Fifth…" menu bar item is hidden
- The hot-key still works (the user knows the key)
- The covert trigger still works (the user knows the phrase)
- The "Last wipe report" item is hidden from the menu
- The Settings UI shows a warning: "Coercion mode: the visible 'Plead the Fifth' controls are hidden. Make sure you remember the hot-key and the covert trigger phrase."

When in coercion mode and the user opens Settings, the "Plead the Fifth" section is visually minimized — collapsed by default, no icon. A user who is being watched wouldn't notice it.

### 8.5 Trigger Dispatch and Confirmation

The three surfaces all dispatch to a single `PleadTheFifthExecutor.destroyAll()` call. The differences are only in the trigger confirmation:

- **Hot-key:** no confirmation.
- **Menu bar:** typed-phrase confirmation (`destroy everything`).
- **Covert trigger:** no confirmation (the phrase IS the confirmation; the user has already typed it).

The confirmation dialog for the menu bar uses a custom NSPanel (not NSAlert) with:
- A text field that **disables paste** (paste is intercepted and ignored) — to defeat the "I copied `destroy everything` from somewhere" attack.
- A 5-second delay after the dialog appears before the "Confirm" button is enabled — to prevent a misclick.
- A failed-attempt log (capped at 3 per session; the 4th attempt logs to the user's audit trail).

---

## 9. Coercion-Resistant UX

### 9.1 The Threat Model for the Covert Trigger

The user is in a situation where:
- They are being watched (visually, by a person or by a camera).
- They may be compelled to use the app (e.g., a journalist told to open their notes for review).
- They cannot visibly destroy the app's data (the adversary would see and stop it).
- They have a covert trigger phrase memorized.

The covert trigger must:
- Fire on the phrase being typed, anywhere in any text input.
- Not display any visible UI when it fires (no "Plead the Fifth engaged" message).
- Run the wipe silently in the background.
- Leave the app in a state that looks normal to the adversary (so the user can "cooperate" with the adversary without arousing suspicion).

### 9.2 The Silent Wipe UX

When the covert trigger fires:
- The wipe begins in a background actor.
- The text input that contained the trigger is **not** modified, **not** cleared, **not** marked. The cursor stays where it is. The text the user typed stays visible.
- The UI continues to render the app as if nothing happened.
- The wipe completes in the background; the encrypted data is gone.
- **The "Library" view continues to show materials until the next refresh** — but the materials are now stale; on next refresh, they appear empty.
- **The Playground chat continues to show the chat history until the next refresh** — same.
- **A forced refresh** (the user can trigger by pressing a designated "refresh" key, e.g. `Cmd+R` or by closing and reopening the relevant view) **clears the visible state**.

This is the key UX property: the visible state changes only on a normal user action. The adversary doesn't see the wipe happen; they only see the result of their own review, which is "no materials, no chat, no library."

### 9.3 False Positive Prevention

The covert trigger fires on substring match. False positives are acceptable (the user can re-set up the app from scratch). To minimize them:

- The phrase must be 8+ characters (configured in Settings).
- The phrase should not be a common English phrase (the Settings hint warns against this).
- A 5-second cooldown after a successful trigger prevents rapid re-fires (e.g., if the user types the same phrase in two windows).
- A pre-trigger check: the phrase must appear in a text field with a length > the phrase length + 4 (i.e., the user must have typed a meaningful sentence, not just the phrase alone). This prevents accidental trigger from a sloppy paste of just the phrase.

### 9.4 The "No Trace" Guarantee

The wipe produces a JSON report in `~/.tessera/last-wipe.json`. The user is informed about this in the user-visible privacy page:

> "Plead the Fifth writes a wipe report to `~/.tessera/last-wipe.json` so you have a record of what was destroyed and when. You can delete this file manually if you want no trace at all. The file contains no secrets — only timestamps and the step outcomes."

The user can `rm ~/.tessera/last-wipe.json` themselves. The app does not auto-delete the report because audit trail is the default; "no trace" is opt-in.

### 9.5 Coercion Mode Display

In coercion mode (Settings toggle), the menu bar item is hidden. But the menu bar icon itself is still there (an icon is required to show the "Coercion mode active" status, since the user might need to know they're in that mode).

The icon in coercion mode is a **neutral icon** — a small lock, indistinguishable from a dozen other macOS menu bar apps. The user knows what the icon means; the adversary does not.

---

## 10. Data Inventory & Migration

### 10.1 Existing Data Locations (Pre-Wipe)

The current Tessera data is in the macOS app sandbox:
- `~/Library/Containers/com.tessera.studio.mac/Data/Library/Application Support/TesseraStudio/` — SwiftData store
- `~/Library/Containers/com.tessera.studio.mac/Data/Library/Application Support/TesseraStudio/default.store` — SwiftData database
- `~/Library/Containers/com.tessera.studio.mac/Data/Library/Application Support/TesseraStudio/default.store-shm` and `.store-wal` — SwiftData write-ahead log and shared memory
- `~/Library/Containers/com.tessera.studio.mac/Data/Library/Caches/TesseraStudio/` — caches
- `~/Library/Containers/com.tessera.studio.mac/Data/Library/Preferences/com.tessera.studio.mac.plist` — UserDefaults

### 10.2 Migration to Encrypted Volume

The migration happens on the first launch after the encrypted-volume feature ships:

1. The app launches normally with the existing data locations.
2. The "Plead the Fifth" feature is detected as not yet configured. The Settings view prompts the user: "Tessera now supports 'Plea the Fifth' — a destroy-everything command. Set up your encrypted volume to enable it."
3. The user sets a volume password (or accepts a random one).
4. The encrypted APFS volume is created.
5. The existing SwiftData store, caches, and preferences are **copied** (not moved) into the volume.
6. The app's data paths are repointed to the volume via a "data directory" symlink or path redirect.
7. The old sandbox location is wiped (defense in depth — even though it's not encrypted, the user explicitly opted in, and the wipe is to ensure no stale data remains).

The migration is a one-time event. After the first launch, all data is in the volume.

### 10.3 First-Run Flow (New Users)

For new users:
1. The app launches with the encrypted-volume setup flow.
2. The user is prompted: "Tessera will create an encrypted volume for your data. The volume password is stored in your Mac's Keychain. Choose a strong password and consider also storing it in 1Password / Bitwarden / Apple Passwords."
3. The user enters a password (or accepts a random one with a copy-to-clipboard button so they can paste it into their password manager).
4. The volume is created and mounted.
5. The app is fully operational.

### 10.4 The "Forgot My Password" Scenario

If the user loses the volume password and it's not in their password manager:
- The encrypted data is unrecoverable. This is the design.
- The app shows a clear message: "Tessera's encrypted volume cannot be opened without the volume password. The password is stored in your Mac's Keychain; if you have it, the app will unlock automatically. If you don't, your data is unrecoverable. This is intentional. To start fresh, choose 'Reset Tessera' below."
- A "Reset Tessera" option is provided in Settings → Advanced → Encryption. This option:
  - Destroys the Keychain entry (if it still exists).
  - Deletes the encrypted volume file.
  - Re-creates an empty volume with a fresh password (or no password, prompting the user to set one on next launch).
  - This is the same as a "Plea the Fifth" + first-run flow.

---

## 11. Testing & Verification

### 11.1 Unit Tests

- `testPleadTheFifthExecutorDestroyAllRunsAllSteps` — verifies the executor runs all 9 steps in order.
- `testKeychainDeleteIsIdempotent` — calling `SecItemDelete` twice is not an error.
- `testSecureOverwriteThreePasses` — verifies the overwrite produces 3 passes of random data + 1 pass of zeros.
- `testCovertTriggerSubstringMatch` — verifies case-insensitive substring matching.
- `testCovertTriggerRespectsMinPhraseLength` — phrases < 8 chars are rejected at Settings time.

### 11.2 Integration Tests

- `testEncryptedVolumeCreatedAndMounted` — full lifecycle on a CI test machine.
- `testPostgresDataDirInsideVolume` — Postgres `initdb` runs inside the mounted volume; data files are encrypted at rest.
- `testWipeStopsPostgresAndValkey` — the executor successfully stops both sidecars in a test environment.
- `testWipeDestroysKeychainEntryAndUnmounts` — verifies the unmount fails (expected, since the key is gone) and the Keychain entry is gone.

### 11.3 Forensic Recovery Tests

This is the critical test category. The forensic recovery test is the one that proves the design works.

Test environment:
- A clean macOS VM (Parallels, UTM, or VMware Fusion) with a real APFS volume.
- The VM has the Tessera app installed.
- A test corpus of fake materials, conversations, and constitutional records is created in the encrypted volume.
- The wipe is invoked.
- After the wipe, forensic tools are run:
  - **PhotoRec** (open source) — attempts to recover files from the raw disk image.
  - **TestDisk** (open source) — same.
  - **EnCase** or **FTK Imager** (commercial) — same.
  - **strings** (Unix) — searches for plaintext strings in the raw disk image.
  - **bulk_extractor** (academic) — extracts artifacts (URLs, email addresses, credit card numbers, etc.) from raw disk images.
- **Expected result:** zero plaintext matches. Zero recoverable files. Zero extracted artifacts that could be associated with the test corpus.

The test runs on a clean VM (no other data on the disk) to ensure the recovery tools are not finding unrelated artifacts. The test corpus contains unique marker strings (e.g., "FORENSIC_TEST_MARKER_abc123") that the test asserts are not found.

### 11.4 Timing Tests

- `testWipeCompletesInUnderTwoSecondsForCryptoSteps` — the hot-key → crypto-shred portion (steps 1-3) completes in ≤ 2 seconds on a reference machine.
- `testWipeCompletesInUnderTenSecondsFull` — the full wipe including overwrite completes in ≤ 10 seconds on a reference machine.

### 11.5 Failure-Mode Tests

- `testWipeContinuesOnPostgresHang` — if Postgres doesn't respond to SIGTERM, SIGKILL is sent after 5s and the wipe continues.
- `testWipeContinuesOnValkeyHang` — same.
- `testWipeHandlesMissingVolumeFile` — if the volume file is already gone (user deleted it manually), the wipe completes successfully.
- `testWipeHandlesKeychainAlreadyEmpty` — if the Keychain entry was already deleted (re-running the wipe), the wipe completes successfully.

---

## 12. External Security Audit

### 12.1 What the Auditor Verifies

The auditor's job is to verify the design's central claim: **"unrecoverable by anyone under any circumstances."**

The audit scope:
1. **Architecture review.** Does the architecture match the design? Is the encrypted volume correctly using APFS encryption? Is the key correctly stored in the Keychain with `kSecAttrAccessibleWhenUnlockedThisDeviceOnly`? Are the sidecars (Postgres, Valkey, DuckDB) actually writing to the volume?
2. **Code review.** Is the wipe procedure correctly implemented? Are there bugs that would leave plaintext on disk? Are there timing issues that would allow a race condition?
3. **Forensic verification.** After the wipe, can the auditor recover the data using state-of-the-art forensic tools? This includes:
   - PhotoRec, TestDisk, EnCase, FTK, bulk_extractor
   - SSD firmware-level access (if the auditor has the equipment)
   - Memory forensics (cold boot attack) on a hibernation file
4. **Covert trigger analysis.** Can the covert trigger be detected by an adversary monitoring the app? (Spoiler: no, the trigger is a passive substring check; the app's CPU/memory profile is the same whether the trigger is set or not.)
5. **Side-channel analysis.** Is the wipe time predictable? Is the volume size predictable? Are there any signals (network activity, file timestamps, etc.) that the wipe occurred?
6. **Compliance verification.** Does the architecture satisfy the legal/regulatory claims (HIPAA, ABA, GDPR, etc.)? The auditor is not a lawyer; the user should engage separate legal counsel for legal compliance. The auditor verifies the technical controls match the claimed compliance posture.

### 12.2 RFP Outline

The Request for Proposal is a separate document; this section summarizes the scope.

The RFP includes:
- Engagement scope: 4-week audit, fixed-fee.
- Reference architecture document (this spec).
- Source code under audit (the encrypted-volume manager, the wipe executor, the trigger surfaces).
- Test environment: macOS VM with the test build installed.
- Deliverables:
  - Audit report (PDF + Markdown), structured by the categories above.
  - Executive summary, suitable for use in marketing/legal disclosures.
  - List of findings, severity-ranked.
  - For each finding, a recommendation; the user commits to addressing critical and high-severity findings before public release.
- Cost: $25,000-50,000 depending on scope and firm.
- Timeline: 4-6 weeks including the audit + the user's remediation.

### 12.3 Audit Firm Recommendations

Firms with a track record of crypto / macOS / iOS security audits:

- **Trail of Bits** — strong on macOS, iOS, cryptography. Has audited 1Password, Signal, and other privacy-first tools. Engagements are typically $30-50K for a 4-week audit.
- **NCC Group** — strong on applied cryptography. UK-based, US presence. Audits of crypto wallets, secure messaging, and similar.
- **Cure53** — German-based, strong on web and mobile security. Has audited password managers and privacy tools.
- **NCC Group / Include Security / Atredis Partners** — additional options.

The user's choice of firm is theirs. The scope of the audit is the same regardless.

### 12.4 Post-Audit Response

After the audit:
1. Critical and high-severity findings are remediated before public release.
2. Medium and low-severity findings are tracked and addressed in subsequent releases.
3. The auditor's executive summary is published on the Tessera website (under the privacy page) as proof of the architecture's claims.
4. The full audit report is available on request to enterprise customers and to legal counsel for compliance review.

---

## 13. Implementation Phases

### Phase 1: Encrypted Volume Foundation (2 weeks)

| Deliverable | Description |
|---|---|
| `TesseraEncryptedVolume` actor | Creates, mounts, unmounts the encrypted APFS volume. Manages the volume password in Keychain. |
| Migration flow | First-launch migration from current sandbox location to the volume. |
| `DataDirectoryRedirector` | Repoints the app's data paths (SwiftData, caches) to the volume. |
| Tests | Unit + integration tests for the volume lifecycle, Keychain integration, migration. |
| Documentation | Internal docs for the architecture. |

**Acceptance criteria:**
- The app launches with the existing data location; the user is prompted to enable encrypted storage; on consent, the data is migrated.
- Subsequent launches mount the volume automatically.
- A "Reset Tessera" option in Settings wipes the volume and starts fresh.

### Phase 2: "Plead the Fifth" Executor (1 week)

| Deliverable | Description |
|---|---|
| `PleadTheFifthExecutor` actor | Implements the 9-step wipe procedure. |
| `PleadTheFifthReport` | Writes the JSON report to `~/.tessera/last-wipe.json`. |
| `PleadTheFifthMenuItem` | NSStatusItem with the three submenu items. |
| Hot-key handler | `NSEvent.addGlobalMonitorForEvents` for `Cmd+Shift+Backspace`. |
| Settings UI | "Coercion mode" toggle. |
| Tests | Unit + integration + forensic recovery tests. |

**Acceptance criteria:**
- The hot-key fires the wipe in ≤ 2 seconds (crypto steps).
- The menu bar item shows the three options and the confirmation dialog works.
- The wipe report is written before the app exits.
- Forensic recovery (PhotoRec, strings) finds no plaintext.

### Phase 3: Covert Trigger (1 week)

| Deliverable | Description |
|---|---|
| `CovertTriggerMonitor` | Async monitor that checks all text inputs for the phrase. |
| Settings UI | Phrase configuration, test button. |
| Silent wipe UX | The wipe runs in the background; the UI continues to render normally until a refresh. |
| Coercion mode display | Hidden menu bar item, neutral icon. |
| Tests | False positive prevention, cooldown enforcement, silent UX verification. |

**Acceptance criteria:**
- The trigger fires on the phrase being typed in any text input.
- The trigger does not fire on phrases < 8 chars.
- The trigger respects the 5-second cooldown.
- The visible UI is unchanged when the trigger fires.
- The materials / chat / library views clear on the next refresh.

### Phase 4: External Security Audit (4-6 weeks)

| Deliverable | Description |
|---|---|
| RFP issuance | Send the audit RFP to 2-3 firms. |
| Firm selection | Based on cost, timeline, and reputation. |
| Audit execution | The firm audits the architecture, code, and forensic recovery. |
| Findings remediation | Critical and high findings addressed before public release. |
| Audit report publication | Executive summary published; full report available on request. |

**Acceptance criteria:**
- The audit report is complete.
- Critical and high findings are remediated.
- The executive summary is published on the privacy page.

### Phase 5: Public Release (1 week)

| Deliverable | Description |
|---|---|
| Marketing page | "Plead the Fifth" feature page on the Tessera website, with the audit's executive summary, the legal positioning for the four use cases (law, healthcare, education, journalism), and the demo video. |
| In-app onboarding | "Set up Plead the Fifth" prompt for existing users; first-run flow for new users. |
| Documentation | User-facing docs: what the feature does, when to use it, what survives. |
| Press / launch | The "Plea the Fifth" feature is the launch story. Target: Hacker News, r/privacy, r/macapps, the legal-tech press. |

**Acceptance criteria:**
- The feature is on by default for new users (with a clear privacy page explaining).
- Existing users see a one-time prompt to enable it.
- The press story is shipped.

### Total: 9-12 weeks from start to public release.

---

## 14. Open Questions

The design is complete enough to start building. The following questions should be resolved before the public release:

1. **Volume password rotation.** When the user rotates their Mac login password, does the Keychain entry for the volume password stay valid? (Apple's Keychain should handle this automatically, but the audit should verify.)
2. **iCloud Keychain.** If the user has iCloud Keychain enabled, is the volume password synced to it? (It shouldn't be — `kSecAttrSynchronizable` is set to false — but verify.)
3. **Time Machine interaction.** Time Machine backs up the encrypted volume file. The backup is also encrypted (Time Machine on APFS uses its own encryption). But if the user restores from a Time Machine backup that includes the volume, they can recover the data — which is the design, but documented for the user.
4. **Migration of existing users with very large data sets.** The first-run migration could take minutes for users with large Postgres data sets. The migration UX should show a progress indicator and offer a "skip — start fresh" option.
5. **Multi-user systems.** The current design assumes a single user per Mac. Multi-user systems (fast user switching) need explicit handling: each user has their own Keychain; the volume is per-user.
6. **Network-attached storage.** If the volume is on a NAS or external drive, the wipe procedure has different timing characteristics. The design should explicitly state that the volume is local-only; storing it on a network share is not supported (and would defeat the privacy posture).
7. **The wipe report's contents.** Should the report include the trigger source (hot-key, menu, covert) for the user's audit trail? Including it is useful (the user can verify which surface was used); excluding it reduces the information leaked if the report is read by an adversary. **Recommendation: include it; the user can edit the report before sharing.**

---

## 15. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Forensic tool evolution. A new recovery tool in 2027 finds data the design doesn't cover. | Low | High (the central claim is broken) | Crypto-shred is the strongest commercially-available mitigation. The audit verifies against current tools; the user should re-audit every 2-3 years. |
| SSD firmware vulnerability leaks data after cryptographic erase. | Low | High | APFS encryption uses Apple's Secure Enclave, not the SSD controller. The SSD firmware vulnerability class (Cai et al., 2018) does not apply. |
| macOS vulnerability in the encrypted-volume or Keychain APIs. | Low | High | The architecture is built on Apple's stable, public APIs. The audit verifies no misuse. |
| The user forgets the volume password AND the Keychain entry is gone. | Medium | Medium (user's data is unrecoverable — this is the design) | Documented to the user; the user is responsible for backup. The "Reset Tessera" option lets them start fresh. |
| Compelled disclosure of the volume password. | High | High (legal exposure) | The covert trigger is the design's response. The user invokes the wipe *before* compulsion becomes irresistible. The legal exposure is then "did you destroy evidence?" not "what's on the drive?" |
| A user accidentally triggers the wipe (false positive on the covert trigger). | Low | Low (the user re-sets up the app) | The 5-second cooldown + the "phrase must be > 8 chars + must appear in a longer string" rule minimize false positives. The user is informed about the design's irreversibility. |
| The wipe takes longer than the time budget. | Low | Medium (the user is exposed during the wipe) | The crypto-shred step (3 of 9) takes < 2 seconds. The defense-in-depth steps (4-9) take 8 more seconds; the user is exposed during those, but the data is already cryptographically unrecoverable. |
| A user uses "Plead the Fifth" and then sues Tessera for "I lost my data!" | Low | Low (legal) | The user-visible privacy page and the in-app documentation make the irreversibility explicit. The wipe report is a record of what was destroyed. |
| An adversary observes the wipe report and uses it to infer the user did something. | Low | Low (the report is innocuous) | The report contains no secrets; only timestamps and step outcomes. The user can delete it manually. |
| A nation-state actor with physical access to the running Mac. | Out of scope (see [Section 3.4](#34-what-this-design-does-not-defend-against)) | — | Out of scope. |

---

## 16. Naming & Branding

### 16.1 The Name: "Plead the Fifth"

The name is the right one for the feature. It maps to:
- **The Fifth Amendment** right against compelled self-incrimination. The feature exists to protect this right at the technical level.
- **The Fourth Amendment** right to be secure in one's papers. The crypto-shred is the architectural expression.
- **A real legal use case** (the user's question to Pieces' CEO).
- **A memorable, distinct brand** for the feature. Pieces can't call their own feature "Plead the Fifth" — the term belongs to the company that ships it first.

The name is also culturally legible: a journalist, lawyer, doctor, or engineer can explain "I have a 'Plea the Fifth' command in my dev tool" to a non-technical audience and the meaning is immediate.

### 16.2 Tagline Options

- "Tessera has a 'Plea the Fifth' command. One keystroke destroys everything. There's no cloud to subpoena, no vendor with your data, no way to compel what doesn't exist."
- "The constitutional right to be secure in your papers — built into the software you work in."
- "Local-first. Encrypted-at-rest. Crypto-shred on demand. Plead the Fifth when you need to."

### 16.3 Marketing Position by Use Case

| Audience | Position |
|---|---|
| **Lawyers** | "Satisfies the technical safeguards requirement for attorney-client privilege. The ABA's Formal Opinion 477R explicitly contemplates local-first software as the reasonable precaution. Plead the Fifth protects against compelled disclosure of case work." |
| **Healthcare attorneys** | "HIPAA-compliant by architecture. No BAA required because no PHI is transmitted. Plead the Fifth gives you defensible data destruction in case of seizure." |
| **Doctors** | "No PHI ever leaves your device. Plead the Fifth ensures the data is gone in 2 seconds if you need it gone." |
| **Journalists** | "The constitutional right to be secure in your papers, built into your notes app. Use it before the subpoena arrives." |
| **Developers** | "Your proprietary code never touches a server. Plead the Fifth is the kill switch for when your laptop gets seized at the border." |
| **Enterprise** | "Local-first AI for sensitive data. Plead the Fifth is your defensible-destruction guarantee. FedRAMP-alternative architecture, audit-attested." |

---

## 17. References

### Standards and Guidelines
- **NIST Special Publication 800-88 r1.** *Guidelines for Media Sanitization.* National Institute of Standards and Technology, December 2014. https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-88r1.pdf
- **NIST Special Publication 800-88 r2 (Draft).** Updated guidelines reflecting modern storage. https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-88r2.pdf
- **DoD 5220.22-M.** *National Industrial Security Program Operating Manual.* (Referenced for the multi-pass overwrite pattern; superseded by NIST 800-88 for modern storage.)

### Apple Platform Documentation
- **Apple Platform Security Guide.** *Data Protection.* Apple's per-file and per-volume encryption model. https://support.apple.com/guide/security/
- **APFS Reference.** *Volume encryption, key management.* https://developer.apple.com/documentation/foundation/file_system
- **macOS Keychain Services.** *Item attribute keys, accessibility classes.* https://developer.apple.com/documentation/security/keychain_services
- **Apple Stack Exchange: Secure Erasing Internal SSDs Using FileVault.** Documents Apple's stance on SSD overwrite and the crypto-erase pattern.

### Legal and Regulatory
- **ABA Model Rule 1.6.** *Confidentiality of Information.* American Bar Association.
- **ABA Formal Opinion 477R.** *Ethical Obligations Related to Internet of Things and Other Networked Devices.* ABA, 2017.
- **45 CFR Parts 160 and 164.** *HIPAA Administrative Simplification and Security Rule.* U.S. Government.
- **HIPAA Safe Harbor (§164.514(b)(2)).** De-identification standard.
- **20 U.S.C. § 1232g; 34 CFR Part 99.** *FERPA — Family Educational Rights and Privacy Act.*
- **GDPR Article 32.** *Security of Processing.* European Union.
- **GDPR Article 25.** *Data Protection by Design and by Default.* European Union.
- **GDPR Article 17.** *Right to Erasure ('Right to be Forgotten').* European Union.
- **Schrems II (Case C-311/18).** *CJEU, 16 July 2020.* Invalidated the US-EU Privacy Shield.
- **United States v. Boucher, 2009.** *Forgone conclusion doctrine, compelled decryption.*
- **United States v. Kirschner, 2010.** *Fifth Amendment and compelled decryption.*
- **In re Grand Jury Subpoena, 11th Circuit 2012.** *Act of decryption is testimonial; forgone conclusion exception.*

### Cryptography
- **Cai et al., "Self-encrypting deception: weaknesses in the encryption of solid-state drives."** Research paper on SSD firmware vulnerabilities. 2018.
- **FIPS 140-2 / FIPS 140-3.** *Security Requirements for Cryptographic Modules.* NIST.
- **NIST SP 800-175B.** *Guideline for Using Cryptographic Standards in the Federal Government.*

### Privacy-First Reference Architectures
- **Signal Architecture.** *End-to-end encrypted messaging, disappearing messages, sealed sender.* https://signal.org/docs/
- **1Password Security Model.** *Encrypted vaults, secret key derivation, secure remote password.* https://1password.com/security/
- **Tails OS — Encrypted Persistent Storage.** *LUKS-encrypted volumes, panic button.* https://tails.net/
- **Mullvad VPN — Kill Switch.** *Network kill switch on connection loss; panic mode.*

### Other
- **Dench Blog.** *Local-First Legal Software: Client Privilege in the AI Era.* The argument for local-first in law. https://www.dench.com/blog/local-first-legal-software
- **Elephas.** *HIPAA-compliant local AI for healthcare attorneys.* https://elephas.app/resources/hipaa-compliant-ai-healthcare-attorneys
- **Anytime AI.** *Legal AI security & HIPAA compliance.* https://www.anytimeai.ai/security/
- **PrivacyScrubber.** *Zero-trust data sanitization, browser-side.* https://privacyscrubber.com/solutions/

### Tools for the Audit
- **PhotoRec** — open-source file recovery. http://www.cgsecurity.org/wiki/PhotoRec
- **TestDisk** — open-source partition recovery. http://www.cgsecurity.org/wiki/TestDisk
- **bulk_extractor** — academic forensic tool. https://downloads.digitalcorpora.org/downloads/bulk_extractor/
- **EnCase** — commercial forensic suite.
- **FTK (Forensic Toolkit)** — commercial forensic suite.

---

## 18. Document Status

This is a draft. The author welcomes review and feedback. The implementation should not begin until:
1. The user has reviewed the design and approved the scope.
2. The legal review (separate engagement with counsel) has approved the compliance positioning.
3. The audit RFP has been issued and a firm selected.

**Next steps:**
- The user reviews this spec.
- The user dispatches the encrypted-volume worker (Phase 1) and the wipe-executor worker (Phase 2) in parallel.
- The covert-trigger worker (Phase 3) follows after the encrypted-volume foundation lands.
- The audit (Phase 4) is engaged after Phase 2 lands.
- The public release (Phase 5) is staged after the audit is complete.

**End of specification.**
