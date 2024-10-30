#!/bin/bash
# run like:
#   ./repo/test.sh fast       [runs only those NOT marked as 'slow']
#   ./repo/test.sh focus      [runs those marked with 'focus']
#   ./repo/test.sh            [runs all]
marker=""
case "$1" in
    "focus")
        marker="focus"
        ;;
    "fast")
        marker="not slow"
        ;;
    "")
        marker=""
        ;;
    *)
        echo "Usage: $0 {focus|fast}"
        exit 1
        ;;
esac

if [ -z "$marker" ]; then
    pytest -x -s --cov-report term-missing:skip-covered --cov=. tests/ -W ignore::DeprecationWarning -W ignore::FutureWarning -W ignore::UserWarning
else
    pytest -x -s -m "$marker" --cov-report term-missing:skip-covered --cov=. tests/ -W ignore::DeprecationWarning -W ignore::FutureWarning -W ignore::UserWarning
fi