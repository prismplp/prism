# Building PRISM with MinGW

Use Git Bash and a native MinGW-w64 toolchain. The tested toolchain is
[w64devkit 2.10.0 x64 multilib](https://github.com/skeeto/w64devkit/releases/tag/v2.10.0),
which supports both `-m64` and `-m32`. These commands require the MinGW-enabled
PRISM and bp4prism sources, including their `Makefile.mingw` files and B-Prolog
LLP64 fixes. They do not use any scripts from `tools/mingw`.

## Build

Assume PRISM is in `~/prism` and the prepared B-Prolog sources are in
`~/bp4prism/Emulator.prism`. Run in Git Bash; adjust the toolchain path.

```sh
set -e
export PATH="/c/path/to/w64devkit/bin:$PATH"
export PRISM_MINGW=1
export PRISM_MINGW_BITS=${PRISM_MINGW_BITS:-64}  # Set 32 before running for x86.
export PRISM_MINGW_MPI=0
PRISM_ROOT=$(cd ~/prism && pwd)
BP_ROOT=$(cd ~/bp4prism && pwd)
bits=$PRISM_MINGW_BITS

make -C "$BP_ROOT/Emulator.prism" -f Makefile.mingw -j4

# Use the checked-in common headers; install only the matching archive.
bp_dest="$PRISM_ROOT/src/c/bp4prism"
mkdir -p "$bp_dest/lib"
cp "$BP_ROOT/Emulator.prism/bp4prism-mingw$bits.a" "$bp_dest/lib/"

make -C "$PRISM_ROOT/src/c" -f Makefile.mingw -j4 install
make -C "$PRISM_ROOT/src/prolog" install
"$PRISM_ROOT/bin/prism" -g 'print_version,halt'
```

The result is `bin/prism_up_mingw64.exe` (or `prism_up_mingw32.exe`). Run the
repository launchers directly:

```sh
~/prism/bin/prism
~/prism/bin/upprism model.psm
PRISM_MINGW_BITS=32 ~/prism/bin/upprism model.psm
```

Repeat the build with `PRISM_MINGW_BITS=32` to produce x86 binaries. The default
is **64**. Both builds use the checked-in headers in `src/c/bp4prism/include`.
The archives coexist as `lib/bp4prism-mingw32.a` and `lib/bp4prism-mingw64.a`
under `src/c/bp4prism`; C/C++ object directories remain separate. The Prolog
bytecode is shared. Do not regenerate the common headers during a normal build
or reuse an archive from the other architecture.

## Optional MPI build

Prepare an MS-MPI runtime and a matching MinGW SDK containing `include/mpi.h`
and `lib/libmsmpi.dll.a`; see the [detailed installation skill](../install_mingw/SKILL.md).
Set both locations explicitly when `tools/mingw` is not distributed:

```sh
export PRISM_MSMPI_SDK="$(cygpath -m /c/path/to/msmpi-sdk-64)"
export PRISM_MSMPI_BIN="$(cygpath -m '/c/Program Files/Microsoft MPI/Bin')"
PRISM_MINGW_MPI=1 make -C ~/prism/src/c -f Makefile.mingw -j4 install
NPROCS=4 ~/prism/bin/mpprism model.psm
```

Use a 32bit SDK and `PRISM_MINGW_BITS=32` for `prism_mp_mingw32.exe`. `NPROCS`
defaults to 4 and must be at least 2. Keep the appropriate runtime DLL available
to each worker. `PRISM_MINGW_MPI` defaults to 0; MPI objects use separate
`.build-mingw32-mp` / `.build-mingw64-mp` directories. No global command
installation is required.

For source preparation, dependency installation, numerical checks, and common
failures, see [install_mingw/SKILL.md](../install_mingw/SKILL.md).
