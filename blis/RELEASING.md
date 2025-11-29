## Contents

* **[BLIS version numbering scheme and branching strategy](RELEASING.md#blis-version-numbering-scheme-and-branching-strategy)**
* **[Instructions for creating a new release candidate or point release of BLIS
](RELEASING.md#instructions-for-creating-a-new-release-candidate-or-point-release-of-blis
)**
  * **[Creating a new release lineage branch
](RELEASING.md#creating-a-new-release-lineage-branch)**
  * **[Creating a new release candidate  (e.g. `1.x` -> `2.0-rc0` or `2.0-rc0` -> `2.0-rc1`)](RELEASING.md#creating-a-new-release-candidate-eg-1x---20-rc0-or-20-rc0---20-rc1)**
  * **[Creating a new major release (e.g. `2.0-rc<n>` -> `2.0`)](RELEASING.md#creating-a-new-major-release-eg-20-rcn---20)**
  * **[Back-porting fixes from `master` to releases](RELEASING.md#back-porting-fixes-from-master-to-releases)**
  * **[Creating a new point release (e.g. `1.1` -> `1.2` or `2.0` -> `2.1`)](RELEASING.md#creating-a-new-point-release-eg-11---12-or-20---21)**

## BLIS version numbering scheme and branching strategy

BLIS uses a major.minor version numbering scheme. An increase in the
major version number (a "major release" or simple "new version")
indicates new (usually significant) functionality, and possible
incompatibility with previous major releases, although the ABI
version can be used to check for compatibility across major version
in principle.

Major releases have one or more "release candidates" which are
preliminary versions of the next release, publicly distributed for
comment and/or bug discovery. Subsequent release candidates (rcs)
correct problems found in the previous rc. Once a reasonable level
of stability is achieved, the full release is distributed.

An increase in the minor version number (a "point release") indicates
the incorportation of one or more bugfixes or other minor changes since
the initial major version release or last point release.

Essentially, point releases extend the rc cadence beyond the official
release by correcting additional problems discovered after release.

All rcs, initial major release, and point releases are created along a
linear git branch, named for the major release lineage, e.g. `r1.x`.
Commits indicating rcs and releases are tagged (e.g. `1.0-rc0`, `1.0`,
`1.1`) and also have an associated non-tip branch (e.g. `r1.0-rc0`,
`r1.0`, `r1.1`). Using both tags and branches increases visibility of
important commits, but new commits should only be made on the `r1.x`
lineage branch.

Release lineage branches diverge from `master` starting with the first
rc. Any new commits on the release lineage (except version maintenance
commits such as updating the version file, CHANGELOG, and release notes)
are cherry-picked from `master`. Exceptions may be made if, for example,
a backported bugfix cannot be cherry-picked and requires a more targeted
fix directly on a release branch.

Here is an example illustration of the release branch structure:
```
_________________________________________________________master
   \                    \
    \                    \__r2.0-rc0_____r2.0-rc1_____r2.0,r2.x
     \                      (2.0-rc0)    (2.0-rc1)    (2.0)
      \
       \__r1.0-rc0_____r1.0-rc1_____r1.0_____r1.1_____r1.2,r1.x
          (1.0-rc0)    (1.0-rc1)    (1.0)    (1.1)    (1.2)
                                     /\
      <- release candidates -- major release -- point releases ->
```

In each case, the version number (as encoded in the `version` file)
indicates the `x.y` prefix of the most recent tagged commit. The
exception is `master`, where the `version` file indicates `z.0-dev`,
where `z` is the major version number one higher than the latest major
release (e.g. `3.0-dev` in the example above).

## Instructions for creating a new release candidate or point release of BLIS

### Creating a new release lineage branch

1. Consider whether the so_version should be updated (via the `build/so_version`
   file) due to any ABI changes since the previous version. If so, commit that
   change on `master` now.

2. Create the new release lineage branch.

   ```
   $ git checkout master
   $ git pull
   $ git branch r2.x
   ```

   Note that the new release lineage branch should not be check out at this point.

3. Update the version on the `master` branch to reflect the next release in development.

   ```
   $ ./build/do-release.sh -b "3.0-dev"
   $ git push
   ```

   Note the extra option `-b`.

4. Check out the new release lineage branch.

   ```
   $ git checkout r2.x
   ```

### Creating a new release candidate (e.g. `1.x` -> `2.0-rc0` or `2.0-rc0` -> `2.0-rc1`)

1. Make sure that the release lineage branch is checked out and up-to-date.

   ```
   $ git checkout r2.x
   $ git pull
   ```

2. Draft a new announcement to the blis-devel mailing list, crediting those who
   contributed towards this version by browsing `git log`.

3. Update the CREDITS file if `git log` reveals any new contributors.
   NOTE: This should have already been done prior to the rc cycle.

4. Commit the updated CREDITS file if changed.

5. Update `docs/ReleaseNotes.md` with the body of finalized announcement
   and the date of the release. Developers are encouraged to update
   the release notes on `master` as new changes are made, which simplifies
   preparation of rc0.

6. Commit the updated `docs/ReleaseNotes.md` file.

7. Use the `build/do-release.sh` script to create a new rc branch and tag.

   ```
   $ ./build/do-release.sh "2.0-rc<n>"
   ```

   Where `<n>` is `0` for the first rc, or one higher than the last rc on this release
   lineage branch.

8. Make sure the `do-release` script and other commits did what they were
   supposed to do by inspecting the output of `git log`. If everything looks good,
   you can push the changes via:

   ```
   $ git push
   $ git push --tags
   $ git push -u <origin> 2.0-rc<n>
   ```

   Where `<origin>` is the name of the appropiate upstream git remote.

   At this point, the new release candidate branch is live at `<origin>`.

9. Announce the rc release on blis-devel, Discord, and/or other appropriate
   venues.

10. Wait for bug reports. Typically an rc should stay live for at least a month
    in order to give users time to try it out.

11. After the trial period, cherry-pick any bugfixes or other updates:

    $ git cherry-pick [-nx] <commit>

    Be sure to include lines in the commit
    log entry for each cherry-picked commit that note the commit hash
    of the *original* commit that is being cherry-picked from. Example:

    ```
    Fixed a bug in blahblahblah. (#777)

    Details:
     - Fixed a bug in blahblahblah that manifested as blahblahblah. This
       bug was introduced in commit abc12345. Thanks to John Smith for
       reporting this bug.
     - (cherry picked from commit abc0123456789abc0123456789abc0123456789a)
    ```

    Note the final line, which was *not* present in the original commit
    log entry (on `master`) but *should be* present in the commit log entry for the
    cherry-picked commit (on the release lineage branch).

 12. If no bugs are reported/found, or if the updated rc is otherwise ready
     for promotion to full release, continue with the instructions below.
     Otherwise, return to step 2, incrementing `<n>`.

### Creating a new major release (e.g. `2.0-rc<n>` -> `2.0`)

1. Make sure that the release lineage branch is checked out and up-to-date.

   ```
   $ git checkout r2.x
   $ git pull
   ```

2. Draft a new announcement to the blis-devel mailing list, crediting those who
   contributed towards this version by browsing `git log`.

3. Update the CREDITS file if `git log` reveals any new contributors.
   NOTE: This should have already been done prior to the release cycle.

4. Commit the updated CREDITS file if changed.

5. Update `docs/ReleaseNotes.md` with the body of finalized announcement
   and the date of the release. Developers are encouraged to update
   the release notes on `master` as new changes are made, which simplifies
   preparation of the release.

6. Commit the updated `docs/ReleaseNotes.md` file.

7. Use the `build/do-release.sh` script to create a new release branch and tag.

   ```
   $ ./build/do-release.sh "2.0"
   ```

8. Make sure the `do-release` script and other commits did what they were
   supposed to do by inspecting the output of `git log`. If everything looks good,
   you can push the changes via:

   ```
   $ git push
   $ git push --tags
   $ git push -u <origin> 2.0
   ```

   Where `<origin>` is the name of the appropiate upstream git remote.

   At this point, the new release branch is live at `<origin>`.

9. Publish a new release via GitHub (https://github.com/flame/blis/releases).
   Identify the new version by the tag you just created and pushed. You can
   also identify the previous release.

   Try to use formatting consistent with the prior release. (You can start to
   edit the previous release, inspect/copy some of the markdown syntax, and
   then abort the edit.)

10. Announce the rc release on blis-devel, Discord, and/or other appropriate
    venues.

11. Update the Wikipedia entry for BLIS to reflect the new latest version.

### Back-porting fixes from `master` to releases

1. When a bug fix is developed on `master` which is applicable to a supported release,
   and corrects a significant problem with correctness, usability, or performance
   (e.g. not new functionality or cosmetic changes), it should be back-ported.
   Bug fixes should be individually back-ported to all supported releases.

2. Check out the relevant release lineage branch, e.g.:

   ```
   $ git checkout r2.x
   $ git pull
   ```

3. Verify that the bug affects this release lineage. If not, skip this release lineage.

4. If possible, cherry-pick the bugfix commit from `master`:

   $ git cherry-pick [-nx] <commit>

   Be sure to include lines in the commit
   log entry for each cherry-picked commit that note the commit hash
   of the *original* commit that is being cherry-picked from. Example:

   ```
     Fixed a bug in blahblahblah. (#777)

     Details:
     - Fixed a bug in blahblahblah that manifested as blahblahblah. This
       bug was introduced in commit abc12345. Thanks to John Smith for
       reporting this bug.
     - (cherry picked from commit abc0123456789abc0123456789abc0123456789a)
   ```

   Note the final line, which was *not* present in the original commit
   log entry (on `master`) but *should be* present in the commit log entry
   for the cherry-picked commit (on the release lineage branch).

5. If cherry-picking is not possible (e.g. the commit does not merge cleanly,
   underlying implementation details or internal APIs have changed, etc.,
   then craft a new bugfix on the release lineage branch. Make sure to test
   the new bugfix against the reported bug, as well as the full BLIS testsuite!

7. Push the changes via `git push`. Do not update any other release branches or tags
   at this time.

### Creating a new point release (e.g. `1.1` -> `1.2` or `2.0` -> `2.1`)

1. Once enough bug fixes have accumulated, a bug fix of high enough urgency, or a
   pre-determined period of time has elapsed, all bug fix commits since the last release
   (major or point release) will be included in a new point release.

   Point releases can be made on either the most recent release lineage branch or on
   a "historical" but still supported release lineage.

2. Check out the relevant release lineage branch (which may not be the most recent)

   ```
   $ git checkout r2.x
   $ git pull
   ```

3. Draft a new announcement to the blis-devel mailing list, crediting those who
   contributed towards this version by browsing `git log`.

4. Update the CREDITS file if `git log` reveals any new contributors.
   NOTE: This should have already been done prior to the release cycle.

5. Commit the updated CREDITS file if changed.

6. Update `docs/ReleaseNotes.md` with the body of finalized announcement
   and the date of the release.

7. Commit the updated `docs/ReleaseNotes.md` file.

8. Use the `build/do-release.sh` script to create a new release branch and tag.

   ```
   $ ./build/do-release.sh "2.1"
   ```

9. Make sure the `do-release` script and other commits did what they were
   supposed to do by inspecting the output of `git log`. If everything looks good,
   you can push the changes via:

   ```
   $ git push
   $ git push --tags
   $ git push -u <origin> 2.1
   ```

   Where `<origin>` is the name of the appropiate upstream git remote.

   At this point, the new release branch is live at `<origin>`.

10. Update the release target branch via GitHub (https://github.com/flame/blis/releases).
    Identify the new version by the tag you just created and pushed. You can
    also identify the previous release.

    Try to use formatting consistent with the prior release. (You can start to
    edit the previous release, inspect/copy some of the markdown syntax, and
    then abort the edit.)

11. Announce the rc release on blis-devel, Discord, and/or other appropriate
    venues.

12. If this point release is for the most recent major release lineage,
    update the Wikipedia entry for BLIS to reflect the new latest version.
