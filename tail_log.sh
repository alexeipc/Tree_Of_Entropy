#!/usr/bin/env bash

set -e

FILE=""

for arg in "$@"; do
    case $arg in
        --file_name=*)
            FILE="${arg#*=}"
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

if [[ -z "$FILE" ]]; then
    echo "Usage: $0 --file_name=<file>"
    exit 1
fi

if [[ ! -f "$FILE" ]]; then
    echo "Error: File '$FILE' does not exist."
    exit 1
fi

# Number of rows in the current terminal
LINES=$(tput lines)

# Leave one line for the prompt
TAIL_LINES=$((LINES - 1))
((TAIL_LINES < 1)) && TAIL_LINES=1

tail -n "$TAIL_LINES" -f "$FILE"
