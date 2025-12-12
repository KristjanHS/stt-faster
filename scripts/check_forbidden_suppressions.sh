#!/usr/bin/env bash
# Check for forbidden Bandit suppressions (B101, B310, B615)
# These test IDs must NEVER be suppressed with # nosec per project security policy
#
# Usage:
#   scripts/check_forbidden_suppressions.sh [files...]
#   If no files provided, checks all Python files in backend/ and scripts/

set -euo pipefail

# Forbidden test IDs that must never be suppressed
FORBIDDEN_TESTS=("B101" "B310" "B615")
FOUND_VIOLATIONS=0
VIOLATION_FILES=()

# Determine which files to check
if [ $# -eq 0 ]; then
    # No arguments: check all Python files in backend/ and scripts/
    FILES_TO_CHECK=$(find backend scripts -name "*.py" -type f 2>/dev/null | sort || true)
else
    # Arguments provided: check only those files
    FILES_TO_CHECK="$@"
fi

if [ -z "$FILES_TO_CHECK" ]; then
    echo "No Python files found to check"
    exit 0
fi

# Check each forbidden test ID
for test_id in "${FORBIDDEN_TESTS[@]}"; do
    while IFS= read -r file; do
        if [ -f "$file" ] && grep -q "# nosec.*${test_id}" "$file" 2>/dev/null; then
            echo "❌ FORBIDDEN: Found suppression of ${test_id} in ${file}"
            grep -n "# nosec.*${test_id}" "$file" | while IFS=: read -r line_num line_content; do
                echo "   Line ${line_num}: ${line_content}"
            done
            FOUND_VIOLATIONS=$((FOUND_VIOLATIONS + 1))
            VIOLATION_FILES+=("${file}:${test_id}")
        fi
    done <<< "$FILES_TO_CHECK"
done

# Report results
if [ $FOUND_VIOLATIONS -gt 0 ]; then
    echo ""
    echo "❌ SECURITY POLICY VIOLATION: Found ${FOUND_VIOLATIONS} forbidden Bandit suppression(s)"
    echo ""
    echo "The following test IDs must NEVER be suppressed with # nosec:"
    for test_id in "${FORBIDDEN_TESTS[@]}"; do
        echo "  - ${test_id}"
    done
    echo ""
    echo "Reason: These are critical security checks that must be fixed, not suppressed."
    echo "See pyproject.toml [tool.bandit] section for policy details."
    echo ""
    echo "Files with violations:"
    for violation in "${VIOLATION_FILES[@]}"; do
        echo "  - ${violation}"
    done
    exit 1
fi

echo "✅ No forbidden Bandit suppressions found"
exit 0

