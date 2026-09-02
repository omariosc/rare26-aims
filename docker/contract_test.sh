#!/usr/bin/env bash
# RARE26 — build the submission image and EXECUTE it under the VERBATIM contract command.
#
# The single most expensive lesson of this campaign (CLiMB, two silent INVALIDs):
# a container that passes every native/apptainer proxy test can still fail on the
# platform, because the proxy never exercised the real read/permission path.  So this
# script performs a real `podman run` with the organisers' own flags from
# RARE25-Submission/do_test_run.sh and validates the JSON contract on the output.
#
# AIRE specifics (learned the hard way, twice, today):
#   * this account has NO /etc/subuid range, so rootless podman is single-mapped and
#     crun fails EVERY `RUN` step with the misleading "clone: No space left on device".
#     The working pattern on this cluster -- the one the CLiMB container was built with --
#     is to enter a user namespace ONCE with `buildah unshare` and do everything inside it,
#     using vfs + ignore_chown_errors + --isolation=oci + --network=host for the build.
#     This script re-execs itself under `buildah unshare` automatically.
#   * there is no CDI/nvidia hook for rootless podman, so the executed test is CPU-mode.
#     That proves the container RUNS and emits a contract-valid file; the GPU path adds
#     only autocast.  Cost is bounded by RARE_MAX_MEMBERS.
#   * inside the single mapping only uid 0 is addressable, so the "foreign uid" run that
#     would directly reproduce the CLiMB failure mode is not executable here.  Step 6
#     substitutes a DIRECT filesystem-permission scan, which is a stronger statement than
#     any one uid happening to work.
#
# env: RARE_MAX_MEMBERS (default 3), PODMAN_STORE (default $TMPDIR/rare26_store_$SLURM_JOB_ID)

# --- enter the user namespace once, then re-run this script inside it ---------------
if [ -z "${_CONTAINERS_USERNS_CONFIGURED:-}" ]; then
  echo "[contract_test] re-exec under \`buildah unshare\` (AIRE has no subuid range)"
  exec buildah unshare "$0" "$@"
fi

set -uo pipefail

CTX=/scratch/sc20osc/miccai-2026/RARE26/docker_ctx
IMAGE=localhost/rare26-aims-uk:v1
TEST=/scratch/sc20osc/miccai-2026/RARE26/docker_test
TPL=/users/sc20osc/RARE26/RARE25-Submission/test/input/interface_0
NMEM=${RARE_MAX_MEMBERS:-3}
ST=${PODMAN_STORE:-${TMPDIR:-/tmp}/rare26_store_${SLURM_JOB_ID:-$$}}
FAIL=0

mkdir -p "$ST/root" "$ST/runroot"
STORE="--root $ST/root --runroot $ST/runroot --storage-driver=vfs"
OPTS="--isolation=oci --network=host --storage-opt vfs.ignore_chown_errors=true"
P="podman $STORE --storage-opt vfs.ignore_chown_errors=true"

step() { echo; echo "=+=+= $* =+=+="; }
ck()   { if [ "$1" -eq 0 ]; then echo "  PASS  $2"; else echo "  FAIL  $2"; FAIL=1; fi; }

step "0. environment"
podman --version; buildah --version
echo "store: $ST"; df -h "$(dirname "$ST")" | tail -1
echo "context: $CTX  ($(du -sh "$CTX" 2>/dev/null | cut -f1))"
echo "members staged: $(ls "$CTX"/resources/members/*.pt 2>/dev/null | wc -l)"

# Base-image cache: Docker Hub is rate-limited and the pull is ~5 min. Keep one copy as
# an oci-archive on scratch and seed the (per-job, node-local) store from it.
BASE=docker.io/pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime
BASECACHE=/scratch/sc20osc/miccai-2026/RARE26/base_pytorch271.oci
step "0b. base image"
if [ -d "$BASECACHE" ]; then
  echo "  seeding store from cache $BASECACHE"
  buildah $STORE --storage-opt vfs.ignore_chown_errors=true pull "oci:$BASECACHE:latest" && buildah $STORE tag "$(buildah $STORE images -q | head -1)" "$BASE"
else
  echo "  no cache; pulling $BASE from the registry (this is the slow path)"
  buildah $STORE --storage-opt vfs.ignore_chown_errors=true pull "$BASE" && buildah $STORE --storage-opt vfs.ignore_chown_errors=true push "$BASE" "oci:$BASECACHE:latest" && \
    echo "  cached to $BASECACHE for next time"
fi

step "1. build (buildah bud, vfs, inside the unshared namespace)"
( cd "$CTX" && buildah $STORE bud $OPTS -t "$IMAGE" -f Dockerfile . )
ck $? "buildah build"
[ $FAIL -eq 1 ] && exit 1
buildah $STORE inspect --type image "$IMAGE" >/dev/null 2>&1
ck $? "image present in store"
buildah $STORE images --format '{{.Name}}:{{.Tag}} {{.Size}}' | grep rare26 || true

step "2. stage test input (the organisers' own 16-slice example batch)"
rm -rf "$TEST"; mkdir -p "$TEST/input" "$TEST/output"
cp -r "$TPL"/. "$TEST/input/"
chmod -R a+rX "$TEST/input"; chmod -R a+rwX "$TEST/output"
find "$TEST/input" -type f | sed 's/^/  /'

# Whether `--network none` is usable on this node is decided BY ATTEMPTING IT, not by
# reading /proc/sys/user/max_net_namespaces: inside `buildah unshare` that sysctl reports
# the INNER namespace's limit, which is not the one enforced (himem01 reads non-zero and
# still fails).  `--network none` still CREATES an empty netns, so on a node whose real
# limit is 0 every run dies with `crun: clone: No space left on device`.
NETNS=none
DEGRADED=0

run_contract () {   # $1 = label; rest = extra podman flags
  local label="$1"; shift
  rm -f "$TEST/output/stacked-neoplastic-lesion-likelihoods.json"
  local t0=$SECONDS err rc
  err=$($P run --rm --network "$NETNS" \
    --volume "$TEST/input":/input:ro \
    --volume "$TEST/output":/output \
    --volume rare26-noop:/tmp \
    -e RARE_MAX_MEMBERS="$NMEM" \
    "$@" "$IMAGE" 2>&1 | tee /dev/stderr)
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ] && [ "$NETNS" = none ] && echo "$err" | grep -q "clone: No space left on device"; then
    echo
    echo "############################################################################"
    echo "## This node cannot create a network namespace, so \`--network none\` is"
    echo "## UNRUNNABLE here (the error above is NOT a disk problem)."
    if [ "${RARE_DEGRADED_OK:-0}" = "1" ]; then
      echo "## RARE_DEGRADED_OK=1 -> retrying with --network=host. Execution, IO, output"
      echo "## schema and permissions ARE still tested; the no-network condition is NOT."
      echo "## This is a DEGRADED run and must not be treated as a contract pass."
      echo "############################################################################"
      NETNS=host; DEGRADED=1
      t0=$SECONDS
      $P run --rm --network host \
        --volume "$TEST/input":/input:ro \
        --volume "$TEST/output":/output \
        --volume rare26-noop:/tmp \
        -e RARE_MAX_MEMBERS="$NMEM" \
        "$@" "$IMAGE"
      rc=$?
    else
      echo "## Re-run on a node where a netns can be created (try the gpu partition),"
      echo "## or set RARE_DEGRADED_OK=1 for a marked, non-authoritative smoke run."
      echo "############################################################################"
    fi
  fi
  echo "  [wall] $((SECONDS - t0)) s for $NMEM members on 16 images (CPU, --network $NETNS)"
  ck $rc "run [$label] exit code"
  return $rc
}

validate () {
  python3 - "$TEST/output/stacked-neoplastic-lesion-likelihoods.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert isinstance(d, list), f"output is {type(d)}, the contract requires a plain list"
assert len(d) == 16, f"expected 16 likelihoods (example batch has 16 slices), got {len(d)}"
assert all(isinstance(v, float) for v in d), "non-float entry"
assert all(0.0 <= v <= 1.0 for v in d), "value outside [0,1]"
const = len(set(round(v, 9) for v in d)) == 1
print(f"  {len(d)} floats, range [{min(d):.4f}, {max(d):.4f}]")
if const:
    print("  *** ALL-CONSTANT -> THIS IS THE FALLBACK, THE MODEL DID NOT RUN ***"); sys.exit(2)
print("  varying scores -> the model really ran")
PY
}

step "3. VERBATIM contract run (--network none, read-only /input, volume on /tmp)"
run_contract "uid0 (AIRE single-mapping)" --user 0:0
step "3b. validate the output against the contract"
validate
ck $? "output schema + the model actually ran (not the fallback)"

step "4. no-network proof — the image must not be able to reach out at all"
$P run --rm --network "$NETNS" --user 0:0 --entrypoint /bin/sh "$IMAGE" -c \
  'python -c "
import socket, sys
socket.setdefaulttimeout(2)
try:
    socket.create_connection((\"1.1.1.1\", 443)); print(\"NETWORK REACHABLE\"); sys.exit(1)
except Exception as e:
    print(\"no network, as expected:\", type(e).__name__); sys.exit(0)"'
if [ "$NETNS" = host ]; then echo "  SKIPPED (see the max_net_namespaces warning above)"; else ck $? "container has no network"; fi

step "5. PERMISSION SCAN — proof that ANY uid the platform picks can read the payload"
$P run --rm --network "$NETNS" --user 0:0 --entrypoint /bin/sh "$IMAGE" -c '
bad=$(find /opt/app \( -type f ! -perm -o+r \) -o \( -type d ! -perm -o+rx \) | head -5)
if [ -n "$bad" ]; then echo "NOT world-readable:"; echo "$bad"; exit 1; fi
echo "  every file under /opt/app is o+r and every dir o+rx"
grep -q "^user:x:1000:1000" /etc/passwd || { echo "user 1000 missing from /etc/passwd"; exit 1; }
echo "  non-root user 1000 exists in /etc/passwd (Grand Challenge requires this)"
python -c "import SimpleITK, numpy, torchvision, PIL, torch; print(\"  deps import OK:\", torch.__version__)"
echo "  members: $(ls /opt/app/resources/members/*.pt | wc -l)"'
ck $? "/opt/app world-readable + non-root user exists + deps import"

step "6. declared image config (what the platform will honour)"
$P inspect "$IMAGE" --format 'USER={{.Config.User}} ENTRYPOINT={{.Config.Entrypoint}} WORKDIR={{.Config.WorkingDir}}'
ck $? "image config"

$P volume rm rare26-noop >/dev/null 2>&1
echo "$STORE" > "$CTX/.podman_store"
echo
if [ "$DEGRADED" = "1" ]; then echo "=+=+= CONTRACT TEST: DEGRADED (no --network none on this node) =+=+="; fi
if [ $FAIL -eq 0 ]; then echo "=+=+= CONTRACT TEST: ALL CHECKS PASS =+=+="
else echo "=+=+= CONTRACT TEST: FAILURES ABOVE =+=+="; fi
exit $FAIL
