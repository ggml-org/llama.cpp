#!/bin/bash

set -e
set -x

TAG=2024.08.03

# The prebuilt toolchains only support hardfloat, so we only
# test these for now.
case $1 in
	"rv32iv")
	TARBALL=riscv32-glibc-ubuntu-20.04-gcc-nightly-${TAG}-nightly.tar.gz
	;;
	"rv64iv")
	TARBALL=riscv64-glibc-ubuntu-20.04-gcc-nightly-${TAG}-nightly.tar.gz
	;;
	"sifive_x280")
	TARBALL=riscv64-glibc-ubuntu-20.04-llvm-nightly-${TAG}-nightly.tar.gz
	;;
	*)
	exit 1
	;;
esac

TOOLCHAIN_PATH=$DIST_PATH/../toolchain
TOOLCHAIN_URL=https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/${TAG}/${TARBALL}

mkdir -p $TOOLCHAIN_PATH
cd $TOOLCHAIN_PATH

wget $TOOLCHAIN_URL
tar -xf $TARBALL

# Once CI upgrades to jammy, the next three lines can be removed.
# The qemu version installed via packages (qemu-user qemu-user-binfmt)
# is sufficient.
TARBALL_QEMU=qemu-riscv-2023.02.25-ubuntu-20.04.tar.gz
wget https://github.com/flame/ci-utils/raw/master/riscv/${TARBALL_QEMU}
tar -xf $TARBALL_QEMU
