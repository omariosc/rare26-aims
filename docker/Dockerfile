# RARE26 — AIMS_UK submission container.
# Based on TUE-ARIA/RARE25-Submission @ dbcce3d, with the campaign's hard-won
# container lessons applied (MICCAI2026_LESSONS.md) and with the constraints from
# https://grand-challenge.org/documentation/runtime-environment/ taken literally:
#   * "Containers should run as a non-root user ... the user should exist"
#     -> user 1000 is written straight into /etc/passwd + /etc/group.  We do NOT
#        call useradd/groupadd: AIRE has no /etc/subuid range for this account, so
#        rootless podman runs single-mapped and any command that needs a second uid
#        dies with a misleading "clone: No space left on device".
#   * everything under /opt/app is world-readable (a+rX) as the LAST step that
#     touches it -> immune to whatever uid the platform actually picks.  This exact
#     defect silently INVALIDated two CLiMB submissions.
#   * "/tmp will be completely empty at runtime" -> nothing is stored there at build;
#     it is used only as scratch, and HOME/caches point at it because it is writable.
#   * "No network access" -> ALL weights are baked in; rare_model.py builds the
#     backbone with pretrained=False + torch.load, so no torch.hub URL path exists.
#
# Base pinned to the exact torch/CUDA our models were trained under (rare_env:
# torch 2.7.1+cu126) and to the *runtime* flavour, not devel (~3 GB smaller, nothing
# here compiles). The template's floating `pytorch/pytorch` tag would drift.
FROM --platform=linux/amd64 pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime AS rare26-aims-uk

ENV PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    TORCH_HOME=/tmp/torch \
    XDG_CACHE_HOME=/tmp/cache \
    HF_HOME=/tmp/hf \
    MPLCONFIGDIR=/tmp/mpl \
    OMP_NUM_THREADS=4

# a real, existing non-root user -- created without useradd (see header)
RUN echo 'user:x:1000:1000:algorithm user:/tmp:/bin/sh' >> /etc/passwd && \
    echo 'user:x:1000:' >> /etc/group

WORKDIR /opt/app

# installed as root into the system site-packages => 0644/0755 => readable by any uid.
# (The upstream template's `pip install --user` lands in an 0700 home and is exactly
#  the permission trap we are avoiding.)
COPY requirements.txt /opt/app/
RUN python -m pip install --no-cache-dir --no-color --requirement /opt/app/requirements.txt

# code first (cheap layers), weights last (expensive layer, changes rarely)
COPY rare_model.py inference.py /opt/app/
COPY dino_repo /opt/app/dino_repo
COPY resources /opt/app/resources

# LAST step touching /opt/app: make everything world-readable/traversable, and
# ASSERT it -- a build that would have shipped an unreadable or incomplete payload
# must fail here, loudly, rather than silently score at chance on the platform.
RUN chmod -R a+rX /opt/app && chmod a+rX /opt && \
    test -r /opt/app/inference.py && \
    test -r /opt/app/rare_model.py && \
    test -d /opt/app/resources/members && \
    test -n "$(ls /opt/app/resources/members/*.pt)" && \
    test -r /opt/app/resources/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth && \
    test -r /opt/app/dino_repo/dinov3/hub/backbones.py && \
    test -z "$(find /opt/app \( -type f ! -perm -o+r \) -o \( -type d ! -perm -o+rx \) | head -1)" && \
    python -c "import SimpleITK, torchvision, PIL, torch; print('deps OK', torch.__version__, torchvision.__version__)" && \
    echo "BUILD ASSERTIONS PASS: $(ls /opt/app/resources/members/*.pt | wc -l) members baked, /opt/app fully world-readable"

USER user
ENTRYPOINT ["python", "inference.py"]
