#!/bin/bash

set -e
set -x

SDE_VERSION=sde-external-8.69.1-2021-07-18-lin
SDE_TARBALL=$SDE_VERSION.tar.bz2
SDE=$SDE_VERSION/sde64

#
# This doesn't seem to be necessary anymore
#
#curl --verbose --form accept_license=1 --form form_id=intel_licensed_dls_step_1 \
#     --output /dev/null --cookie-jar jar.txt \
#     --location https://software.intel.com/protected-download/267266/144917
#curl --verbose --cookie jar.txt --output $SDE_TARBALL \
#     https://software.intel.com/system/files/managed/2a/1a/$SDE_TARBALL

#curl --verbose --output $SDE_TARBALL \
#     https://software.intel.com/content/dam/develop/external/us/en/documents/downloads/$SDE_TARBALL

CI_UTILS=ci-utils
CI_UTILS_URL=https://github.com/flame/${CI_UTILS}.git
CI_UTILS_SDE_DIR=sde
SDE_DIRPATH=$CI_UTILS/$CI_UTILS_SDE_DIR

git clone $CI_UTILS_URL
mv $SDE_DIRPATH/$SDE_TARBALL .

tar xvf $SDE_TARBALL

make -j2 testsuite-bin blastest-bin

for ARCH in penryn sandybridge haswell skx knl piledriver steamroller excavator zen generic; do
    # The leading space is for stress testing
    export BLIS_ARCH_TYPE=" -1"

    if [ "$ARCH" = "knl" ]; then
        TESTSUITE_WRAPPER="$SDE -knl --"
    elif [ "$ARCH" = "sandybridge" ]; then
        # The sandybridge.def file causes a segfault in SDE on some systems.
        # Instead, use the CPUID values for haswell, but force BLIS to use the
        # sandybridge configuration.
        TESTSUITE_WRAPPER="$SDE -cpuid_in $DIST_PATH/ci/cpuid/haswell.def --"
        export BLIS_ARCH_TYPE="sandybridge"
    elif [ "$ARCH" = "piledriver" ]; then
        # We used to "patch" ld.so and libm to remove CPUID checks so that glibc
        # wouldn't try to use instructions not supported by SDE (FMA4). That no
        # longer works, so test Piledriver/Steamroller/Excavator as haswell
        # but with the configuration forced via environment variable.
        TESTSUITE_WRAPPER="$SDE -cpuid_in $DIST_PATH/ci/cpuid/haswell.def --"
        export BLIS_ARCH_TYPE="piledriver"
    elif [ "$ARCH" = "steamroller" ]; then
        TESTSUITE_WRAPPER="$SDE -cpuid_in $DIST_PATH/ci/cpuid/haswell.def --"
        export BLIS_ARCH_TYPE="steamroller"
    elif [ "$ARCH" = "excavator" ]; then
        TESTSUITE_WRAPPER="$SDE -cpuid_in $DIST_PATH/ci/cpuid/haswell.def --"
        export BLIS_ARCH_TYPE="excavator"
    elif [ "$ARCH" = "generic" ]; then
        TESTSUITE_WRAPPER="$SDE -cpuid_in $DIST_PATH/ci/cpuid/haswell.def --"
        export BLIS_ARCH_TYPE="generic"
    else
        TESTSUITE_WRAPPER="$SDE -cpuid_in $DIST_PATH/ci/cpuid/$ARCH.def --"
    fi

    make TESTSUITE_WRAPPER="$TESTSUITE_WRAPPER" check

    TMP=`grep "active sub-configuration" output.testsuite`
    CONFIG=${TMP##* }
    if [ "$CONFIG" != "$ARCH" ]; then
        echo "Wrong configuration chosen:"
        echo "    Expected: $ARCH"
        echo "    Got: $CONFIG"
        exit 1
    fi
done

