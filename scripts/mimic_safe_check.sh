#!/bin/bash
# MIMIC Data Safety Check
# Prevents real/processed clinical data from being committed to git.
# Used as pre-commit hook and can be run standalone.
#
# Usage:
#   ./scripts/mimic_safe_check.sh          # check staged files
#   ./scripts/mimic_safe_check.sh --all    # check entire working tree

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

BLOCKED=0
WARNINGS=()

# ---- Determine files to check ----
if [[ "${1:-}" == "--all" ]]; then
    FILES=$(find . -not -path './.git/*' -type f 2>/dev/null)
else
    FILES=$(git diff --cached --name-only --diff-filter=ACMR 2>/dev/null || true)
fi

if [[ -z "$FILES" ]]; then
    echo -e "${GREEN}✓ No files to check.${NC}"
    exit 0
fi

# ---- 1. Block raw MIMIC table names (case-insensitive) ----
MIMIC_TABLES=(
    ADMISSIONS PATIENTS DIAGNOSES_ICD PROCEDURES_ICD PRESCRIPTIONS
    LABEVENTS NOTEEVENTS ICUSTAYS CHARTEVENTS TRANSFERS DRGCODES
    CPTEVENTS MICROBIOLOGYEVENTS INPUTEVENTS_MV INPUTEVENTS_CV
    OUTPUTEVENTS CAREGIVERS SERVICES CALLOUT DATETIMEEVENTS
    PROCEDUREEVENTS_MV D_ICD_DIAGNOSES D_ICD_PROCEDURES D_ITEMS D_LABITEMS
)

for f in $FILES; do
    basename_upper=$(basename "$f" | tr '[:lower:]' '[:upper:]')
    for table in "${MIMIC_TABLES[@]}"; do
        if [[ "$basename_upper" == "${table}.CSV"* ]] || [[ "$basename_upper" == "${table}.CSV.GZ" ]]; then
            WARNINGS+=("BLOCK: $f — matches MIMIC table '${table}'")
            BLOCKED=1
        fi
    done
done

# ---- 2. Block dangerous file extensions ----
BLOCKED_EXTS="pkl pickle h5 hdf5 feather parquet npy npz sql sql.gz db"

for f in $FILES; do
    ext="${f##*.}"
    ext_lower=$(echo "$ext" | tr '[:upper:]' '[:lower:]')
    for blocked_ext in $BLOCKED_EXTS; do
        if [[ "$ext_lower" == "$blocked_ext" ]]; then
            WARNINGS+=("BLOCK: $f — binary data file (.${ext_lower}) may contain patient data")
            BLOCKED=1
        fi
    done
done

# ---- 3. Block paths containing mimic/physionet data directories ----
# Match directory components like /mimic3/, /mimiciii/, /physionet/ but NOT
# tool filenames like mimic_safe_check.sh or mimic-safe/SKILL.md
for f in $FILES; do
    f_lower=$(echo "$f" | tr '[:upper:]' '[:lower:]')
    if [[ "$f_lower" =~ /(mimic[_-]?(iii|iv|3|4)|physionet|pyhealth_cache)/ ]] || \
       [[ "$f_lower" == *"/.cache/pyhealth/"* ]]; then
        WARNINGS+=("BLOCK: $f — path contains clinical data directory name")
        BLOCKED=1
    fi
done

# ---- 4. Content scan: hardcoded real patient IDs in staged text files ----
TEXT_EXTS="py json jsonl tsv csv md yaml yml txt cfg ini toml"

for f in $FILES; do
    [[ -f "$f" ]] || continue
    ext="${f##*.}"
    ext_lower=$(echo "$ext" | tr '[:upper:]' '[:lower:]')

    is_text=0
    for text_ext in $TEXT_EXTS; do
        [[ "$ext_lower" == "$text_ext" ]] && is_text=1 && break
    done
    [[ $is_text -eq 0 ]] && continue

    # Check for numeric patient IDs (subject_id = 12345, hadm_id = 67890, etc.)
    if grep -qPi '(subject_id|hadm_id|icustay_id|stay_id)\s*[=:,]\s*[0-9]{4,}' "$f" 2>/dev/null; then
        WARNINGS+=("BLOCK: $f — contains hardcoded MIMIC patient identifiers")
        BLOCKED=1
    fi

    # Check for MIMIC data paths used as string literals (not in comments/help text)
    # Only flag definitive assignments like root="/data/mimiciii/"
    if grep -qPi '(=|:)\s*["\x27]/[^\s]*(mimiciii|mimic-iii|mimic3|mimiciv|mimic-iv|physionet)[^\s]*["\x27]' "$f" 2>/dev/null; then
        # Exclude lines that are clearly help text or comments
        real_hits=$(grep -Pi '(=|:)\s*["\x27]/[^\s]*(mimiciii|mimic-iii|mimic3|mimiciv|mimic-iv|physionet)[^\s]*["\x27]' "$f" 2>/dev/null | grep -v '^\s*#' | grep -v '^\s*"""' | grep -v 'help=' | grep -v 'e\.g\.' | grep -v 'example' | grep -v '/path/to/' || true)
        if [[ -n "$real_hits" ]]; then
            WARNINGS+=("WARN: $f — contains hardcoded MIMIC data path (verify it's not a real path)")
        fi
    fi
done

# ---- 5. Check for large files that might be data dumps ----
for f in $FILES; do
    [[ -f "$f" ]] || continue
    size=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
    # Flag files > 10MB
    if [[ $size -gt 10485760 ]]; then
        WARNINGS+=("WARN: $f — large file ($(( size / 1048576 ))MB), verify no patient data")
    fi
done

# ---- Report ----
if [[ ${#WARNINGS[@]} -eq 0 ]]; then
    echo -e "${GREEN}✓ MIMIC safety check passed. No clinical data detected.${NC}"
    exit 0
fi

echo ""
echo -e "${RED}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${RED}║           MIMIC DATA SAFETY CHECK FAILED                ║${NC}"
echo -e "${RED}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

for w in "${WARNINGS[@]}"; do
    if [[ "$w" == BLOCK:* ]]; then
        echo -e "  ${RED}✗ ${w}${NC}"
    else
        echo -e "  ${YELLOW}⚠ ${w}${NC}"
    fi
done

echo ""
if [[ $BLOCKED -eq 1 ]]; then
    echo -e "${RED}Commit BLOCKED. Remove or .gitignore the flagged files:${NC}"
    echo "  git reset HEAD <file>        # unstage"
    echo "  echo '<pattern>' >> .gitignore  # prevent future staging"
    exit 1
else
    echo -e "${YELLOW}Warnings only — commit allowed, but please verify the flagged files.${NC}"
    exit 0
fi
