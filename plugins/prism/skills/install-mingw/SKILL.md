---
name: install-mingw
description: Build and verify PRISM and bp4prism on Windows with native MinGW-w64, including optional 32bit and 64bit MS-MPI builds, without tools/mingw helper scripts.
---

# Install PRISM with MinGW-w64

Use this workflow for the MinGW-enabled PRISM and bp4prism sources. Produce
working `prism`, `upprism`, and, when requested, `mpprism` launchers inside the
PRISM checkout. Use 64bit by default; retain 32bit support through
`PRISM_MINGW_BITS=32`. Keep dependencies outside the repository and do not
depend on files under `tools/mingw`.

## 1. Check sources and prerequisites

- Use a Windows host with Git Bash or an MSYS MinGW shell. The reference
  toolchain is [w64devkit 2.10.0 x64 multilib](https://github.com/skeeto/w64devkit/releases/tag/v2.10.0)
  (GCC 16.2); extract it to a user-selected directory. It supplies native
  `gcc`, `g++`, `ar`, `make`, `sh`, and `dlltool`. Cygwin executables are a
  different build target.
- Locate the existing PRISM and bp4prism checkouts; examples use `~/prism` and
  `~/bp4prism`. If absent, obtain the project's MinGW-enabled revisions.
  Upstream projects are [PRISM](https://github.com/prismplp/prism) and
  [bp4prism](https://github.com/prismplp/bp4prism). A plain upstream checkout
  may lack these local MinGW changes.
- Require `prism/src/c/Makefile.mingw`, MinGW-aware launchers and
  `src/prolog/Compile.sh`, plus bp4prism's `Makefiles/Makefile.mingw` and updated
  `bp4prism.patch`. B-Prolog needs the provided `Emulator` sources and compatible
  `prism/bin/bp.out`; cloning bp4prism alone does not supply proprietary sources.
- For x64, confirm the B-Prolog sources include the `BP_MINGW64`/LLP64 changes.
  Windows `long` is 32bit even with `-m64`; compiler flags alone cannot fix the
  word/pointer ABI. Keep these changes conditional on `PRISM_MINGW=1`.

Run the following in Git Bash, adjusting the paths. Keep using this shell for
the subsequent commands. Use a writable work directory outside the checkout.

```sh
set -e
export PATH="/c/path/to/w64devkit/bin:$PATH"
export PRISM_MINGW=1
export PRISM_MINGW_BITS=${PRISM_MINGW_BITS:-64}
export PRISM_MINGW_MPI=0
case $PRISM_MINGW_BITS in 32|64) ;; *) exit 1 ;; esac
PRISM_ROOT=$(cd ~/prism && pwd)
BP_ROOT=$(cd ~/bp4prism && pwd)
bits=$PRISM_MINGW_BITS
PRISM_BUILD_WORK=${PRISM_BUILD_WORK:-$HOME/prism-build}
mkdir -p "$PRISM_BUILD_WORK/tmp"
PRISM_BUILD_WORK=$(cd "$PRISM_BUILD_WORK" && pwd)
export TMP="$(cygpath -m "$PRISM_BUILD_WORK/tmp")"
export TEMP="$TMP" TMPDIR="$TMP"
command -v gcc g++ ar make patch sh dlltool
gcc --version
printf 'int main(void){return 0;}\n' > "$PRISM_BUILD_WORK/compiler.c"
gcc -m"$bits" "$PRISM_BUILD_WORK/compiler.c" -o "$PRISM_BUILD_WORK/compiler.exe"
"$PRISM_BUILD_WORK/compiler.exe"
```

Native tools receive Windows paths on their command line through Git Bash.
Paths exported for a native makefile, notably `PRISM_MSMPI_SDK`, should be
converted with `cygpath -m` because environment variable values are not
automatically converted.

## 2. Prepare B-Prolog sources when necessary

Reuse a valid, already patched `Emulator.prism`. Do not rerun the upstream
regeneration script over local edits: it removes that directory.

If `Emulator.prism` is absent, create it from the supplied `Emulator` sources.
Use Git for Windows' GNU patch with `--binary` for the original CRLF sources;
adjust the GNU patch location for another installation. Stop on a rejected
patch and resolve the source/version mismatch before building.

```sh
test ! -e "$BP_ROOT/Emulator.prism"
mkdir "$BP_ROOT/Emulator.prism"
cp "$BP_ROOT"/Emulator/*.c "$BP_ROOT"/Emulator/*.h "$BP_ROOT/Emulator.prism/"
"/c/Program Files/Git/usr/bin/patch.exe" --binary \
    -d "$BP_ROOT/Emulator.prism" -p1 < "$BP_ROOT/bp4prism.patch"
cp "$BP_ROOT"/Makefiles/* "$BP_ROOT/Emulator.prism/"
```

## 3. Build B-Prolog and install the matching archive

```sh
make -C "$BP_ROOT/Emulator.prism" -f Makefile.mingw -j4
bp_dest="$PRISM_ROOT/src/c/bp4prism"
mkdir -p "$bp_dest/lib"
cp "$BP_ROOT/Emulator.prism/bp4prism-mingw$bits.a" "$bp_dest/lib/"
```

Both architectures use the version-controlled public headers in
`src/c/bp4prism/include`. The MinGW word types and macros are conditional on
`PRISM_MINGW=1`; the existing platform branches remain available. Keep the two
archives side by side in `src/c/bp4prism/lib`, with their architecture suffixes.
Do not copy emulator headers over the common headers during ordinary builds.

When upgrading B-Prolog to a different source version, export fresh headers
listed in `bp4prism.headers` to a temporary directory and apply the matching
`bp4prism/bp4prism-release.patch` there. Do not use the older patch under
`prism/tools/bp4prism` or patch the emulator headers in place. The release patch
removes the emulator's `exit` redirection and provides the public `TERM`
definition. Review and merge the resulting public API changes into the common
headers, preserving the MinGW guards and existing platform branches. Recheck
both architectures and non-MinGW preprocessing before replacing those headers.

Before linking PRISM, check that a Prolog word has the same width as a pointer:

```sh
cat > "$PRISM_BUILD_WORK/abi.c" <<'C'
#include "bprolog.h"
_Static_assert(sizeof(BPLONG) == sizeof(void *), "word/pointer width mismatch");
_Static_assert(sizeof(BPULONG) * 8 == NBITS_IN_LONG, "word mask width mismatch");
int main(void) { return 0; }
C
gcc -m"$bits" -std=gnu11 -DPRISM_MINGW=1 -DWIN32 -DNT \
    -I"$bp_dest/include" "$PRISM_BUILD_WORK/abi.c" -o "$PRISM_BUILD_WORK/abi.exe"
"$PRISM_BUILD_WORK/abi.exe"
```

## 4. Build and run the single-process version

```sh
make -C "$PRISM_ROOT/src/c" -f Makefile.mingw -j4 install
make -C "$PRISM_ROOT/src/prolog" install
"$PRISM_ROOT/bin/prism" -g 'print_version,halt'
objdump -f "$PRISM_ROOT/bin/prism_up_mingw$bits.exe"
```

Build C/C++ before compiling Prolog: `Compile.sh` uses the newly built native
executable. Confirm `bp.out`, `prism.out`, `foc.out`, `batch.out`, and
`mpprism.out` exist in `bin`. The latter four are built by the Prolog makefile.
Prolog bytecode is shared by both architectures.

Run `~/prism/bin/prism` interactively and `~/prism/bin/upprism model.psm` in batch
mode. Keep these commands in the checkout; a global `bin` installation is not
needed. To switch target in the same shell, run
`export PRISM_MINGW_BITS=32; bits=$PRISM_MINGW_BITS` (or use `64`), then repeat
sections 3–4. Ordinary objects
use `.build-mingw32` / `.build-mingw64` and executables have distinct names.

## 5. Optional MS-MPI setup

Skip this section for a single-process installation. Reuse an available MS-MPI
runtime, or install `msmpisetup.exe` from the official
[Microsoft MPI 10.1.3 download](https://www.microsoft.com/en-us/download/details.aspx?id=105289).
For a normal Windows installation, the launcher directory is typically
`C:/Program Files/Microsoft MPI/Bin`; the runtime installer supplies the DLLs
through Windows. Microsoft documents [both x86 and x64 application builds](https://github.com/microsoft/Microsoft-MPI/blob/master/examples/helloworld/Run_MPIHelloWorld.md).
For an existing portable runtime, use its directory containing `mpiexec.exe`,
`smpd.exe`, `msmpi.dll`, and `msmpires.dll`; keep x86 and x64 DLLs separate.

The MinGW makefile expects a GNU import archive, `lib/libmsmpi.dll.a`.
Create a C/C++ SDK outside the checkout from the
[MSYS2 MS-MPI definitions](https://github.com/msys2/MINGW-packages/tree/master/mingw-w64-msmpi).
The following pinned hashes identify the headers and definitions used for this
port; a mismatch requires reviewing the upstream change, not bypassing the check.

```sh
sdk="$PRISM_BUILD_WORK/msmpi-sdk-$bits"
mkdir -p "$sdk/include" "$sdk/lib"
base=https://raw.githubusercontent.com/msys2/MINGW-packages/master/mingw-w64-msmpi
case $bits in
    32) arch=i686; machine=i386
        def_hash=90c90a29ed4084dccf227920f1600ebde7d33c02f308551c764bd19d36b0d7fc ;;
    64) arch=x86_64; machine=i386:x86-64
        def_hash=7e8be35bd1286d671dd3d47ffa1c3eaca09bf0cb6286f2474dc0eb71ea976d75 ;;
esac
curl -fL "$base/mpi.h" -o "$sdk/include/mpi.h"
curl -fL "$base/msmpi.def.$arch" -o "$sdk/lib/msmpi.def"
printf '%s  %s\n' \
    baee3f18f38650e7182956baa0d3d8f8e5c26d8603ccee4871a4dbd160c13660 "$sdk/include/mpi.h" \
    "$def_hash" "$sdk/lib/msmpi.def" | sha256sum -c -
dlltool -m "$machine" --as-flags="--$bits" -k \
    -d "$sdk/lib/msmpi.def" -l "$sdk/lib/libmsmpi.dll.a"
export PRISM_MSMPI_SDK="$(cygpath -m "$sdk")"
export PRISM_MSMPI_BIN="$(cygpath -m '/c/Program Files/Microsoft MPI/Bin')"
```

For a portable runtime, replace the last path with its actual location. The
`--as-flags=--32` option is essential with multilib w64devkit: `-m i386` alone
can leave 64bit head/tail objects inside an otherwise 32bit import archive.
These SDK commands cover C/C++, not Fortran modules.

```sh
PRISM_MINGW_MPI=1 make -C "$PRISM_ROOT/src/c" -f Makefile.mingw -j4 install
objdump -f "$PRISM_ROOT/bin/prism_mp_mingw$bits.exe"
```

This compiles all PRISM C/C++ code with `-DMPI` into separate
`.build-mingw$bits-mp` objects and reuses the matching B-Prolog archive.
Repeat SDK preparation and the MPI build after switching bitness. When only an
external SDK archive changes, use `make -B` with the same flags to force relinking.
The launchers' fallback paths are local conveniences under `tools/mingw`; always
set `PRISM_MSMPI_SDK` and `PRISM_MSMPI_BIN` explicitly for a checkout without that
directory.

## 6. Verify behavior

Create a small numerical check outside the checkout:

```sh
cat > "$PRISM_BUILD_WORK/coin.psm" <<'PROLOG'
values(coin,[head,tail]).
toss(Face) :- msw(coin,Face).
prism_main :-
    set_sw(coin,[0.7,0.3]),
    prob(toss(head),P0), abs(P0-0.7) < 0.000000001,
    learn([toss(head),toss(head),toss(head),toss(tail)]),
    prob(toss(head),P1), abs(P1-0.75) < 0.000000001,
    learn([toss(head),toss(tail),toss(tail),toss(tail)]),
    prob(toss(head),P2), abs(P2-0.25) < 0.000000001,
    format("MINGW_OK ~w ~w ~w~n",[P0,P1,P2]).
PROLOG
"$PRISM_ROOT/bin/upprism" "$PRISM_BUILD_WORK/coin.psm"
# Run these two commands only if the MPI version was built:
NPROCS=2 PRISM_MPIRUN_OPTS='-timeout 60' \
    "$PRISM_ROOT/bin/mpprism" "$PRISM_BUILD_WORK/coin.psm"
NPROCS=4 PRISM_MPIRUN_OPTS='-timeout 60' \
    "$PRISM_ROOT/bin/mpprism" "$PRISM_BUILD_WORK/coin.psm"
```

Require `MINGW_OK 0.7 0.75 0.25` and a successful process exit. A failed Prolog
goal may print `no` while returning exit code zero. Four MPI processes also
exercise workers with no assigned goals because there are only two unique
observations. `NPROCS` includes the master, defaults to 4, and must be at least 2.

For x64 also verify wide integers:

```sh
"$PRISM_ROOT/bin/prism" -g 'X is 1099511627899,format("~d~n",[X]),halt'
```

Expect exactly `1099511627899`. Record executable machine types (`pei-i386` for
x86, `pei-x86-64` for x64), compiler version, selected bitness, successful
numerical checks, and any remaining limitations. Verification of x86 binaries
on Windows x64/WOW64 does not establish support for a 32bit Windows OS or a
multi-host MPI cluster.

## Troubleshooting and boundaries

- **Pointer/word mismatch or truncated x64 integers:** verify the MinGW-aware
  B-Prolog patch, checked-in common headers, and matching archive in
  `src/c/bp4prism/lib`; do not suppress the ABI check or globally redefine `long`.
- **Patch fails or says it was already applied:** check the source version and
  line endings. Ordinary builds do not apply the release patch. For a header
  upgrade, apply it only to fresh exported copies in a temporary directory.
- **Missing `uname`, `dirname`, or `cygpath`:** use a configured Git Bash/MSYS
  shell. `bash.exe --login` is appropriate when starting it from PowerShell.
- **Cannot create a compiler temporary file:** point `TMP`, `TEMP`, and `TMPDIR`
  to an existing writable Windows path, as in section 1.
- **MPI link/DLL architecture errors:** inspect both the import archive and
  runtime DLLs; regenerate the SDK for the selected architecture. Use the
  matching portable runtime directory when overriding the system installation.
- **MPI status error or a crash with idle workers:** require this port's guarded
  `MPI_STATUS_IGNORE`, `NUL`, and empty-graph fixes; do not replace an existing
  MPI binary with an unpatched upstream build.
- **MPI batch arguments:** keep `-c` in the launcher. Do not add `-l` to its
  model arguments. `PRISM_MPIRUN_OPTS` is whitespace-separated; for a machine
  file use `MACHINES`. Multi-host operation needs separate validation.

Keep changes opt-in through `PRISM_MINGW` and `PRISM_MINGW_MPI`. Do not alter
Linux, macOS, or Cygwin configurations. This workflow covers native PRISM and
its existing MPI parameter-learning features; it does not install the Python,
HDF5, or Protocol Buffers components of T-PRISM.
