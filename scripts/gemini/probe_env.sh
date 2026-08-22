#!/usr/bin/env bash
# Probe the GEMINI H200 node's environment and write a small, patient-data-free
# text report. See docs/gemini.md for why this exists: nobody on this team
# can log into GEMINI directly, so this is how the rest of the team learns
# what the node actually looks like (GPU/driver/CUDA, Python, package
# manager, libstdc++)
# before writing the environment recipe.
#
# Usage (run on the GEMINI node, from the repo root):
#   scripts/gemini/probe_env.sh
#
# Writes scripts/gemini/out/env_probe.txt. Safe to commit and push: GPU/tool
# version strings and disk/RAM totals only, no database access, no patient
# data of any kind.
set -u

REPO_DIR=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
OUT_DIR="$REPO_DIR/scripts/gemini/out"
OUT_FILE="$OUT_DIR/env_probe.txt"
mkdir -p "$OUT_DIR"

have() { command -v "$1" >/dev/null 2>&1; }

{
    echo "=== env_probe: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    echo "host: $(hostname 2>/dev/null || echo unknown)"

    echo
    echo "--- GPU (nvidia-smi) ---"
    if have nvidia-smi; then
        nvidia-smi --query-gpu=name,driver_version,memory.total \
            --format=csv 2>&1 || echo "nvidia-smi query failed"
        echo
        nvidia-smi 2>&1 | head -20
    else
        echo "nvidia-smi not found"
    fi

    echo
    echo "--- CUDA toolkit ---"
    if have nvcc; then
        nvcc --version 2>&1
    else
        echo "nvcc not found on PATH"
    fi
    echo "CUDA_HOME=${CUDA_HOME:-<unset>}"
    echo "candidate install dirs:"
    ls -d /usr/local/cuda* 2>/dev/null || echo "  none found under /usr/local"

    echo
    echo "--- environment modules (module avail) ---"
    if have module; then
        module avail 2>&1 | head -40
    else
        echo "module command not found"
    fi

    echo
    echo "--- Python ---"
    if have python3; then
        python3 --version 2>&1
        python3 -c "import sys; print(sys.executable)" 2>&1
    else
        echo "python3 not found"
    fi

    echo
    echo "--- package managers ---"
    for tool in uv poetry conda mamba; do
        if have "$tool"; then
            printf '%-8s %s\n' "$tool" "$("$tool" --version 2>&1 | head -1)"
        else
            printf '%-8s not found\n' "$tool"
        fi
    done

    echo
    echo "--- glibc / libstdc++ ---"
    if have ldd; then
        ldd --version 2>&1 | head -1
    else
        echo "ldd not found"
    fi
    LIBSTDCPP=$(find /usr/lib/x86_64-linux-gnu /usr/local/lib -name "libstdc++.so.6" 2>/dev/null | head -1)
    if [[ -n "$LIBSTDCPP" ]]; then
        echo "libstdc++: $LIBSTDCPP"
        if have strings; then
            echo "highest GLIBCXX symbols:"
            strings "$LIBSTDCPP" | grep -E '^GLIBCXX_[0-9.]+$' | sort -V | tail -5
        fi
    else
        echo "libstdc++.so.6 not found under common paths"
    fi

    echo
    echo "--- disk (repo filesystem) ---"
    df -h "$REPO_DIR" 2>&1

    echo
    echo "--- memory ---"
    if have free; then
        free -h 2>&1
    else
        echo "free not found"
    fi
} >"$OUT_FILE" 2>&1

echo "Wrote $OUT_FILE"
