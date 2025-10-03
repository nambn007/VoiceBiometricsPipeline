#!/usr/bin/env bash
set -euo pipefail

# Download and install ONNX Runtime (CPU) into third_party/onnxruntime
# Usage: bash setup_onnxruntime.sh [VERSION]
# Default VERSION: 1.18.0

VERSION="${1:-1.18.0}"
PLATFORM="linux-x64"
REPO_BASE="https://github.com/microsoft/onnxruntime/releases/download"
PKG_NAME="onnxruntime-${PLATFORM}-${VERSION}"
TARBALL="${PKG_NAME}.tgz"
URL="${REPO_BASE}/v${VERSION}/${TARBALL}"

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
TP_DIR="${ROOT_DIR}/third_party"
DEST_DIR="${TP_DIR}/onnxruntime"
TMP_DIR="${TP_DIR}/.onnxruntime_tmp"

mkdir -p "${TP_DIR}"
rm -rf "${TMP_DIR}"
mkdir -p "${TMP_DIR}"

echo "Downloading ${PKG_NAME} from ${URL} ..."
curl -L --fail --retry 3 --retry-connrefused -o "${TMP_DIR}/${TARBALL}" "${URL}"

echo "Extracting ${TARBALL} ..."
tar -xzf "${TMP_DIR}/${TARBALL}" -C "${TMP_DIR}"

# The extracted folder contains include/ and lib/
EXTRACTED_DIR="${TMP_DIR}/${PKG_NAME}"
if [[ ! -d "${EXTRACTED_DIR}/include" || ! -d "${EXTRACTED_DIR}/lib" ]]; then
  echo "Unexpected ONNX Runtime package layout at ${EXTRACTED_DIR}" >&2
  exit 1
fi

echo "Installing to ${DEST_DIR} ..."
rm -rf "${DEST_DIR}"
mkdir -p "${DEST_DIR}"
cp -R "${EXTRACTED_DIR}/include" "${DEST_DIR}/"
cp -R "${EXTRACTED_DIR}/lib" "${DEST_DIR}/"

echo "ONNX Runtime installed at: ${DEST_DIR}"
echo "Includes: ${DEST_DIR}/include"
echo "Libs:     ${DEST_DIR}/lib"

echo "Done. Update/build with CMake will now find ONNX Runtime in third_party/onnxruntime."


