#!/bin/bash
# Script to completely erase all database information

set -e

echo "======================================"
echo "DATABASE CLEANUP SCRIPT"
echo "======================================"

# Get the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Directories to clean
INDICES_DIR="$PROJECT_ROOT/data/indices"
PROCESSED_DIR="$PROJECT_ROOT/data/processed"
RAW_DIR="$PROJECT_ROOT/data/raw"
LOGS_DIR="$PROJECT_ROOT/logs"

echo ""
echo "This will delete:"
echo "  - FAISS indices ($INDICES_DIR)"
echo "  - Processed documents ($PROCESSED_DIR)"
echo ""
read -p "Do you want to proceed? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Cleanup cancelled."
    exit 0
fi

# Clean FAISS indices
if [ -d "$INDICES_DIR" ]; then
    echo ""
    echo "Cleaning FAISS indices..."
    rm -rf "$INDICES_DIR"/*
    echo "✓ FAISS indices deleted"
else
    echo "✓ FAISS indices directory not found, skipping..."
fi

# Clean processed documents
if [ -d "$PROCESSED_DIR" ]; then
    echo ""
    echo "Cleaning processed documents..."
    rm -rf "$PROCESSED_DIR"/*
    echo "✓ Processed documents deleted"
else
    echo "✓ Processed documents directory not found, skipping..."
fi

# Ask about optional cleanup
echo ""
read -p "Do you also want to delete raw documents? (yes/no): " DELETE_RAW

if [ "$DELETE_RAW" = "yes" ]; then
    if [ -d "$RAW_DIR" ]; then
        echo "Cleaning raw documents..."
        rm -rf "$RAW_DIR"/*
        echo "✓ Raw documents deleted"
    fi
fi

echo ""
read -p "Do you want to delete logs? (yes/no): " DELETE_LOGS

if [ "$DELETE_LOGS" = "yes" ]; then
    if [ -d "$LOGS_DIR" ]; then
        echo "Cleaning logs..."
        rm -rf "$LOGS_DIR"/*
        echo "✓ Logs deleted"
    fi
fi

echo ""
echo "======================================"
echo "CLEANUP COMPLETED!"
echo "======================================"
echo ""
echo "All database information has been erased."
echo "You can now rebuild the index with fresh data using:"
echo "  python scripts/build_index.py"
