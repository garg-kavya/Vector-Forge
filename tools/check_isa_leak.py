#!/usr/bin/env python3
"""Instruction-set leak check for runtime SIMD dispatch (docs/simd.md, docs/DESIGN.md §10.4).

Only the AVX2 kernel object may contain AVX instructions: everything else in the library and CLI
must run on baseline x86-64, or runtime dispatch is pointless (illegal-instruction crashes on
older CPUs). The script disassembles the object files of an optimised build and fails if

  1. a function in a generic object (vectorforge, vf_cli_commands, vectorforge_cli) contains a
     VEX/EVEX-encoded SIMD instruction (any ymm/zmm register, or a v-prefixed mnemonic on xmm
     registers), unless the function is runtime-dispatched by the toolchain itself: MSVC's
     auto-vectorizer guards AVX2 loops with `__isa_available` checks and the UCRT's inline wmemcmp
     tests `_Avx2WmemEnabled`;
  2. the AVX2 object (vf_simd_avx2) contains no ymm instruction (flags did not apply);
  3. the AVX2 object defines an externally visible function outside vf::detail::avx2 -- an inline
     function or template instantiated there could be picked by the linker for everyone (ODR/ISA
     leakage). Data symbols (e.g. MSVC's merged `__ymm@` constants) carry no instructions.

Usage: tools/check_isa_leak.py <build-dir>
Needs objdump + nm (GCC/Clang/MinGW builds) or dumpbin (MSVC builds, run from a developer shell).
Do not run on sanitizer or fuzzer builds: their runtimes add instrumentation symbols.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

GENERIC_TARGETS = ("vectorforge.dir", "vf_cli_commands.dir", "vectorforge_cli.dir")
AVX2_TARGET = "vf_simd_avx2.dir"
ALLOWED_NAMESPACE = "vf::detail::avx2::"

# "vaddps ymm0, ..." / "vfmadd231ps %ymm1,..." ; plain SSE on xmm has no 'v' prefix.
WIDE_REGISTER = re.compile(r"\b[yz]mm\d+\b")
VEX_ON_XMM = re.compile(r"\bv[a-z0-9]+\s+.*\bxmm\d+\b")
# Legitimate non-SIMD mnemonics starting with 'v'.
NON_SIMD_V = re.compile(r"\b(verr|verw|vmcall|vmlaunch|vmresume|vmxoff|vmxon|vmfunc)\b")
# Toolchain runtime-dispatch guards (see module docstring).
RUNTIME_GUARD = re.compile(r"__isa_available|_Avx2WmemEnabled")
# Function headers: dumpbin "name:" at column 0; objdump "0000000000000000 <name>:".
FUNCTION_HEADER = re.compile(r"^(?:[0-9a-f]+ <.*>:|\S.*:)$")


def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace", check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{result.stderr}")
    return result.stdout


def objects(build_dir: Path, target: str) -> list[Path]:
    found = [p for p in build_dir.rglob("*") if p.suffix in (".o", ".obj") and target in p.parts]
    return sorted(found)


def disassemble(obj: Path) -> str:
    if obj.suffix == ".obj" and shutil.which("dumpbin"):
        return run(["dumpbin", "/nologo", "/disasm:nobytes", str(obj)])
    return run(["objdump", "-d", "--no-show-raw-insn", "-M", "intel", str(obj)])


def functions(disassembly: str) -> list[tuple[str, list[str]]]:
    result: list[tuple[str, list[str]]] = []
    for line in disassembly.splitlines():
        if FUNCTION_HEADER.match(line) and not line.startswith(("Dump of", "File Type")):
            result.append((line.rstrip(":"), []))
        elif result:
            result[-1][1].append(line)
    return result


def is_simd(line: str) -> bool:
    text = line.lower()
    if NON_SIMD_V.search(text):
        return False
    return bool(WIDE_REGISTER.search(text) or VEX_ON_XMM.search(text))


def unguarded_simd(disassembly: str) -> tuple[list[str], int]:
    """(unguarded hits as "function: instruction", number of guarded functions skipped)."""
    hits = []
    guarded = 0
    for name, body in functions(disassembly):
        simd = [line.strip() for line in body if is_simd(line)]
        if not simd:
            continue
        if any(RUNTIME_GUARD.search(line) for line in body):
            guarded += 1
            continue
        hits.append(f"{name[:120]}: {simd[0]}")
    return hits, guarded


def exported_functions(obj: Path) -> list[str]:
    """Demangled names of defined, externally visible (incl. weak/COMDAT) functions."""
    names = []
    if obj.suffix == ".obj" and shutil.which("dumpbin"):
        for line in run(["dumpbin", "/nologo", "/symbols", str(obj)]).splitlines():
            if "External" not in line or " SECT" not in line or "()" not in line.split("|", 1)[0]:
                continue
            demangled = re.search(r"\((.*)\)\s*$", line)
            names.append(demangled.group(1) if demangled else line.split("|", 1)[-1].strip())
        return names
    for line in run(["nm", "-C", "--defined-only", str(obj)]).splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[1] in ("T", "W"):
            names.append(parts[2])
    return names


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    build_dir = Path(sys.argv[1])
    failures = []

    generic = [o for t in GENERIC_TARGETS for o in objects(build_dir, t)]
    if not generic:
        print(f"error: no generic objects found under {build_dir}")
        return 2
    guarded_total = 0
    for obj in generic:
        hits, guarded = unguarded_simd(disassemble(obj))
        guarded_total += guarded
        for hit in hits:
            failures.append(f"{obj}: unguarded AVX instruction in {hit}")

    avx2 = objects(build_dir, AVX2_TARGET)
    cache = build_dir / "CMakeCache.txt"
    expects_avx2 = cache.is_file() and "VF_AVX2_KERNELS:INTERNAL=TRUE" in cache.read_text(errors="replace")
    if not avx2 and expects_avx2:
        failures.append("the build enables AVX2 kernels but has no vf_simd_avx2 object (not built?)")
    elif not avx2:
        print("note: no AVX2 kernel object (VF_ENABLE_AVX2=OFF or non-x86-64); generic check only")
    for obj in avx2:
        if not any(WIDE_REGISTER.search(line.lower()) for line in disassemble(obj).splitlines()):
            failures.append(f"{obj}: no ymm instructions; AVX2 flags were not applied")
        for name in exported_functions(obj):
            if ALLOWED_NAMESPACE not in name:
                failures.append(f"{obj}: exports '{name}' outside {ALLOWED_NAMESPACE}")

    print(f"checked {len(generic)} generic object(s) and {len(avx2)} AVX2 object(s); "
          f"{guarded_total} toolchain runtime-dispatched function(s) allowed")
    for failure in failures[:40]:
        print(f"FAIL {failure}")
    if failures:
        print(f"{len(failures)} failure(s)")
        return 1
    print("OK: AVX instructions are confined to the AVX2 kernel object")
    return 0


if __name__ == "__main__":
    sys.exit(main())
