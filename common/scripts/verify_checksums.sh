#!/bin/bash
#
# This script generates or verifies SHA256 checksums for model weights.
#
# Usage:
#   To generate a checksum file:
#   ./common/scripts/verify_checksums.sh generate /path/to/weights_dir
#
#   To verify checksums against a file:
#   ./common/scripts/verify_checksums.sh verify /path/to/SHA256SUMS

set -e

MODE=$1
TARGET=$2

if [ -z "$MODE" ] || [ -z "$TARGET" ]; then
    echo "Usage: $0 <generate|verify> <directory|file>"
    exit 1
fi

if [ "$MODE" == "generate" ]; then
    if [ ! -d "$TARGET" ]; then
        echo "Error: Directory not found at $TARGET"
        exit 1
    fi
    echo "Generating SHA256SUMS for files in $TARGET..."
    cd "$TARGET"
    sha256sum * > SHA256SUMS
    echo "SHA256SUMS created."
elif [ "$MODE" == "verify" ];
    if [ ! -f "$TARGET" ]; then
        echo "Error: Checksum file not found at $TARGET"
        exit 1
    fi
    echo "Verifying checksums in $TARGET..."
    cd "$(dirname "$TARGET")"
    sha256sum -c "$(basename "$TARGET")"
    echo "Verification successful."
else
    echo "Error: Invalid mode '$MODE'. Use 'generate' or 'verify'."
    exit 1
fi
