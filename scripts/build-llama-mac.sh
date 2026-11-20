#!/usr/bin/env bash
# Aion 対応パッチ版 llama.cpp を Mac (Apple Silicon) でビルドするスクリプト。
#
# 上流 llama.cpp は Aion アーキテクチャ未対応のため、AtomicChat が
# GGUF リポジトリで配布する aion-port.patch を、指定コミットの
# llama.cpp に当ててビルドします。
#
# 使い方:  bash scripts/build-llama-mac.sh
# 成果物:  .llama-src/llama.cpp-<commit>/build/bin/llama-server
set -euo pipefail

LLAMA_COMMIT="bafa0420c1e57edf04d71587d32fe57d45df7b95"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_PARENT="$ROOT/.llama-src"
SRC_DIR="$SRC_PARENT/llama.cpp-$LLAMA_COMMIT"
PATCH_FILE="$ROOT/llama.cpp/aion-port.patch"
TARBALL="$SRC_PARENT/llama.tar.gz"

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake が見つかりません。先に実行してください: brew install cmake" >&2
  exit 1
fi

mkdir -p "$SRC_PARENT"

if [ ! -d "$SRC_DIR" ]; then
  echo "llama.cpp ($LLAMA_COMMIT) をダウンロードします…"
  curl -L -o "$TARBALL" \
    "https://github.com/ggml-org/llama.cpp/archive/$LLAMA_COMMIT.tar.gz"
  tar -xzf "$TARBALL" -C "$SRC_PARENT"
fi

cd "$SRC_DIR"

if patch -p1 --dry-run < "$PATCH_FILE" >/dev/null 2>&1; then
  echo "aion-port.patch を適用します…"
  patch -p1 < "$PATCH_FILE"
else
  echo "パッチは適用済み（または適用不要）のようです。続行します。"
fi

echo "ビルドします（Metal は Apple Silicon では既定で有効）…"
cmake -B build -DLLAMA_USE_PREBUILT_UI=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server

echo ""
echo "完了: $SRC_DIR/build/bin/llama-server"
echo "起動例:"
echo "  \"$SRC_DIR/build/bin/llama-server\" -m models/Aion-1.0-Instruct-2.3B-AD-Q4_K_M.gguf --host 127.0.0.1 --port 8080 -c 8192"
