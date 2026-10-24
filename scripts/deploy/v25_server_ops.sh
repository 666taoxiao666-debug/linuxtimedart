#!/usr/bin/env bash
# v25 deployment helper, run from the LOCAL repository (Windows Git Bash or Linux).
# Safety: fast-forward only, never reset/force/overwrite; refuses when an incoming commit touches tracked files
# that are modified on the server; never kills, pauses or renices any process; changes no device/network setting.
#
#   bash scripts/deploy/v25_server_ops.sh resolve-pin     # pin upstream TSMixer commit + file SHA-256, commit locally
#   bash scripts/deploy/v25_server_ops.sh push            # git push origin codex/self-evolving-wiki
#   bash scripts/deploy/v25_server_ops.sh inspect         # live server Git/process/GPU state (read-only)
#   bash scripts/deploy/v25_server_ops.sh sync            # ff-only sync of the pushed local HEAD on the server
#   bash scripts/deploy/v25_server_ops.sh bundle-sync     # same, via a verified git bundle when GitHub is unreachable
#   bash scripts/deploy/v25_server_ops.sh preflight       # tests + A0 + synthetic smoke on the server (detached)
#   bash scripts/deploy/v25_server_ops.sh preflight-status
#   bash scripts/deploy/v25_server_ops.sh launch          # first real run (detached; refuses if a v25 run exists)
#   bash scripts/deploy/v25_server_ops.sh status | resume | report
#   bash scripts/deploy/v25_server_ops.sh fetch           # copy reports/figures/metrics (no weights) back locally
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
KEY="${V25_SSH_KEY:-C:/Users/XT/.ssh/codex_sdwpf_monitor_ed25519}"
HOST="${V25_HOST:-XT@100.89.238.37}"
REPO="/home/XT/nuist/pythoncode/TimeDARTFirst"
BRANCH="codex/self-evolving-wiki"
PY="/home/XT/miniconda3/envs/timedart/bin/python"
LAUNCH="scripts/train/Kelmarsh_launch_backbone_information.sh"
OPTS=(-i "${KEY}" -o BatchMode=yes -o ConnectTimeout=180 -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
remote() { ssh "${OPTS[@]}" "${HOST}" "cd '${REPO}' && $*"; }
# ff_checks is embedded in single quotes on the remote side, so it must not contain single quotes itself.

ff_checks='set -e
incoming=$(git rev-parse FETCH_HEAD)
[ "$incoming" = "$TARGET" ] || { echo "fetched $incoming but local HEAD is $TARGET; push/bundle first"; exit 4; }
[ "$(git branch --show-current)" = "'"${BRANCH}"'" ] || { echo "server is not on '"${BRANCH}"'"; exit 4; }
git merge-base --is-ancestor HEAD FETCH_HEAD || { echo "not a fast-forward; refusing (no reset/force)"; exit 4; }
overlap=$(comm -12 <(git diff --name-only HEAD FETCH_HEAD | sort) <(git diff HEAD --name-only | sort))
[ -z "$overlap" ] || { echo "incoming commit touches locally modified tracked files: $overlap"; exit 4; }
echo "server tracked modifications kept:"; git status --porcelain --untracked-files=no
git merge --ff-only FETCH_HEAD
echo "SERVER_HEAD=$(git rev-parse HEAD)"'
[[ "${ff_checks}" != *"'"* ]] || { echo "internal quoting error" >&2; exit 5; }

case "${1:-}" in
  resolve-pin)
    PIN="configs/tsmixer_upstream_pin_v25.json"
    grep -q '"commit": null' "${PIN}" || { echo "Pin already resolved; not overwriting:"; cat "${PIN}"; exit 0; }
    API="https://api.github.com/repos/google-research/google-research/commits?path=tsmixer/tsmixer_basic/models/tsmixer.py&per_page=1"
    COMMIT="$(curl -fsSL "${API}" | grep -m1 '"sha"' | sed -E 's/.*"sha": *"([0-9a-f]{40})".*/\1/')"
    [[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]] || { echo "Could not resolve upstream commit" >&2; exit 4; }
    RAW="https://raw.githubusercontent.com/google-research/google-research/${COMMIT}"
    mkdir -p output/kelmarsh_hourly/tsmixer_upstream
    curl -fsSL "${RAW}/tsmixer/tsmixer_basic/models/tsmixer.py" -o "output/kelmarsh_hourly/tsmixer_upstream/tsmixer_${COMMIT}.py"
    curl -fsSL "${RAW}/LICENSE" -o "output/kelmarsh_hourly/tsmixer_upstream/LICENSE_${COMMIT}"
    FILE_SHA="$(sha256sum "output/kelmarsh_hourly/tsmixer_upstream/tsmixer_${COMMIT}.py" | cut -d' ' -f1)"
    LICENSE_SHA="$(sha256sum "output/kelmarsh_hourly/tsmixer_upstream/LICENSE_${COMMIT}" | cut -d' ' -f1)"
    grep -q "Apache License" "output/kelmarsh_hourly/tsmixer_upstream/LICENSE_${COMMIT}" || { echo "Upstream LICENSE is not Apache" >&2; exit 4; }
    printf '{\n  "repository": "https://github.com/google-research/google-research",\n  "path": "tsmixer/tsmixer_basic/models/tsmixer.py",\n  "license": "Apache-2.0",\n  "license_path": "LICENSE",\n  "commit": "%s",\n  "file_sha256": "%s",\n  "license_sha256": "%s",\n  "resolved_utc": "%s",\n  "resolved_with": "GitHub commits API (last commit touching the file) + raw file sha256sum",\n  "note": "Port differences: PyTorch, RevIN off, fixed 2 blocks/ff 32/dropout 0.1, power-slot head, optional power-only mask; see docs/KELMARSH_BACKBONE_INFORMATION_V25.md"\n}\n' \
      "${COMMIT}" "${FILE_SHA}" "${LICENSE_SHA}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${PIN}"
    cat "${PIN}"
    echo "Review output/kelmarsh_hourly/tsmixer_upstream/tsmixer_${COMMIT}.py against models/TSMixerBasicSmall.py, then:"
    echo "  git add ${PIN} && git commit -m 'Pin upstream TSMixer source for v25'"
    ;;
  push) git push origin "${BRANCH}" ;;
  inspect)
    remote 'echo "HEAD=$(git rev-parse HEAD) BRANCH=$(git branch --show-current)"; echo "--- tracked modifications"; git status --porcelain --untracked-files=no;
            echo "--- v25 identifiers"; ls configs | grep -i v25 || echo none; cat outputs/logs/Kelmarsh/backbone_information_latest.txt 2>/dev/null || echo NO_V25_POINTER;
            echo "--- GPU"; nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu --format=csv;
            nvidia-smi --query-compute-apps=pid,used_memory --format=csv; echo "--- processes"; ps -eo pid,user,etime,pcpu,pmem,args --sort=-pcpu | head -15; uptime'
    ;;
  sync)
    TARGET="$(git rev-parse HEAD)"
    remote "git fetch origin '${BRANCH}' && TARGET='${TARGET}' bash -c '${ff_checks}'"
    ;;
  bundle-sync)
    TARGET="$(git rev-parse HEAD)"
    BASE="$(remote 'git rev-parse HEAD')"
    git cat-file -e "${BASE}^{commit}" || { echo "Server HEAD ${BASE} unknown locally; fetch first" >&2; exit 4; }
    BUNDLE="output/kelmarsh_hourly/backbone_information_v25_${TARGET:0:7}.bundle"
    git bundle create "${BUNDLE}" "${BASE}..${BRANCH}"
    git bundle verify "${BUNDLE}"
    remote "mkdir -p output/kelmarsh_hourly"
    scp "${OPTS[@]}" "${BUNDLE}" "${HOST}:${REPO}/${BUNDLE}"
    remote "sha256sum '${BUNDLE}' && git bundle verify '${BUNDLE}' && git fetch '${BUNDLE}' '${BRANCH}' && TARGET='${TARGET}' bash -c '${ff_checks}'"
    sha256sum "${BUNDLE}"
    ;;
  preflight) remote "KELMARSH_PYTHON='${PY}' bash '${LAUNCH}' --preflight-only --detach" ;;
  preflight-status)
    remote 'ls -t outputs/logs/Kelmarsh/launcher/backbone_information_preflight_*.log 2>/dev/null | head -1 | xargs -r tail -n 25;
            latest=$(ls -td outputs/logs/Kelmarsh/backbone_information_preflight/*/ 2>/dev/null | head -1);
            [ -n "$latest" ] && grep -m3 -E "\"passed\"|full_source_digest" "$latest/preflight.json" || echo "no preflight record yet"'
    ;;
  launch) remote "KELMARSH_PYTHON='${PY}' bash '${LAUNCH}' --detach" ;;
  status) remote "KELMARSH_PYTHON='${PY}' bash '${LAUNCH}' --status" ;;
  resume) remote "KELMARSH_PYTHON='${PY}' bash '${LAUNCH}' --resume --detach" ;;
  report) remote "KELMARSH_PYTHON='${PY}' bash '${LAUNCH}' --report-only --detach" ;;
  fetch)
    DIR="$(remote 'cat outputs/logs/Kelmarsh/backbone_information_latest.txt')"
    DEST="output/kelmarsh_hourly/backbone_information_v25_review"
    mkdir -p "${DEST}"
    ssh "${OPTS[@]}" "${HOST}" "tar czf - -C '${DIR}' --exclude='*.pt' --exclude='*.tmp' ." | tar xzf - -C "${DEST}"
    echo "Copied (no weights) to ${DEST}"
    ;;
  *) sed -n '2,17p' "$0"; exit 2 ;;
esac
