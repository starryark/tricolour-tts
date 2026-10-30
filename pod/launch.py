"""RunPod lifecycle for T1, from the local machine. The API key is read from RUNPOD_API_KEY
and never printed or written.

    python pod/launch.py quote                 # read-only: price and stock of the GPU choices
    python pod/launch.py create --hours 6      # paid: watchdog first, then one Secure pod
    python pod/launch.py status                # pod state, cost/h, ssh command
    python pod/launch.py terminate             # stop billing now (retries until confirmed gone)
    python pod/launch.py watchdog              # (spawned detached by create) hard deadline
State: <WORK>/pod/lease.json. Statuses: creating -> running -> terminated; "unknown" means a
create whose outcome is not known - the watchdog reconciles it by the pod's unique name, and
no new pod can be created until it is resolved.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))
import common as C  # noqa: E402

GRAPHQL = "https://api.runpod.io/graphql"
IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
GPUS = ["NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL"]      # preference order
DISK_GB = 150
MAX_USD_PER_HOUR = 4.5
KEY = Path.home() / ".ssh" / "runpod_ed25519"
LEASE = C.WORK / "pod" / "lease.json"


class ApiError(Exception):
    pass


def gql(query: str, variables: dict | None = None) -> dict:
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(GRAPHQL, data=body, method="POST", headers={
        "Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"], "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise ApiError(f"HTTP {e.code}") from None
    except Exception as e:                                  # timeouts, resets, bad JSON
        raise ApiError(type(e).__name__) from None
    if out.get("errors"):
        raise ApiError("; ".join(str(x.get("message", ""))[:200] for x in out["errors"]))
    return out["data"]


def now() -> datetime:
    return datetime.now(timezone.utc)


def read_lease() -> dict:
    return json.loads(LEASE.read_text(encoding="utf-8")) if LEASE.exists() else {}


def write_lease(d: dict) -> None:
    LEASE.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEASE.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
    os.replace(tmp, LEASE)


def update_lease(**kw) -> dict:
    d = {**read_lease(), **kw}
    write_lease(d)
    return d


def quote() -> list[dict]:
    q = "query { gpuTypes { id memoryInGb lowestPrice(input: {gpuCount: 1, secureCloud: true}) { uninterruptablePrice stockStatus } } }"
    rows = [g for g in gql(q)["gpuTypes"] if g["id"] in GPUS]
    rows.sort(key=lambda g: GPUS.index(g["id"]))
    return rows


def my_pods() -> list[dict]:
    return gql("query { myself { pods { id name desiredStatus costPerHr } } }")["myself"]["pods"]


def pod_info(pod_id: str) -> dict | None:
    q = ("query($id: String!) { pod(input: {podId: $id}) { id name desiredStatus costPerHr "
         "runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } } }")
    return gql(q, {"id": pod_id}).get("pod")


def ssh_target(info: dict | None) -> tuple[str, int] | None:
    for p in ((info or {}).get("runtime") or {}).get("ports") or []:
        if p["privatePort"] == 22 and p["isIpPublic"]:
            return p["ip"], int(p["publicPort"])
    return None


def resolve_pod_id(lease: dict) -> str | None:
    """The lease's pod, or - for a create whose response was lost - the pod with our unique name."""
    if lease.get("pod_id"):
        return lease["pod_id"]
    matches = [p for p in my_pods() if p.get("name") == lease.get("name")]
    if len(matches) > 1:
        raise ApiError("more than one pod carries this lease's name")
    if matches:
        update_lease(pod_id=matches[0]["id"])
        return matches[0]["id"]
    return None


def terminate_until_gone(max_minutes: float = 30) -> bool:
    """Terminate and confirm, retrying with backoff. True once no pod of this lease exists."""
    t_end = time.monotonic() + max_minutes * 60
    delay = 10
    while True:
        lease = read_lease()
        try:
            pid = resolve_pod_id(lease)
            if pid is None:
                # nothing with our name: gone, or a create that never materialised
                created = datetime.fromisoformat(lease.get("created_at", now().isoformat()))
                if now() - created > timedelta(minutes=10):
                    update_lease(status="terminated", terminated_at=now().isoformat(), note="no pod found")
                    return True
            elif pod_info(pid) is None:
                update_lease(status="terminated", terminated_at=now().isoformat())
                return True
            else:
                gql("mutation($id: String!) { podTerminate(input: {podId: $id}) }", {"id": pid})
        except (ApiError, KeyError) as e:
            print(f"{now().isoformat()} terminate attempt failed: {e}", flush=True)
        if time.monotonic() > t_end:
            update_lease(status="cleanup_failed")
            return False
        time.sleep(delay)
        delay = min(delay * 2, 120)


def spawn_watchdog() -> None:
    flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW) \
        if os.name == "nt" else 0
    log = open(LEASE.with_name("watchdog.log"), "a", encoding="utf-8")
    subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "watchdog"], stdout=log, stderr=log,
                     stdin=subprocess.DEVNULL, creationflags=flags, close_fds=True)


def create(hours: float) -> None:
    lease = read_lease()
    if lease.get("status") in ("creating", "running", "unknown", "cleanup_failed"):
        raise SystemExit(f"lease status is {lease.get('status')!r} ({lease.get('pod_id')}) - resolve it first")
    offer = next((g for g in quote() if (g["lowestPrice"] or {}).get("stockStatus") in ("High", "Medium", "Low")
                  and (g["lowestPrice"] or {}).get("uninterruptablePrice")), None)
    if offer is None:
        raise SystemExit("no Secure H100 in stock")
    price = float(offer["lowestPrice"]["uninterruptablePrice"])
    if price > MAX_USD_PER_HOUR:
        raise SystemExit(f"{offer['id']} quoted at ${price}/h, above the ${MAX_USD_PER_HOUR}/h cap")
    name = "tri-t1-" + uuid.uuid4().hex[:10]
    write_lease({"status": "creating", "name": name, "gpu": offer["id"], "quoted_usd_per_hour": price,
                 "created_at": now().isoformat(), "deadline": (now() + timedelta(hours=hours)).isoformat()})
    spawn_watchdog()                     # before the paid call: it reconciles a lost response by name
    time.sleep(3)
    pub = KEY.with_suffix(".pub").read_text(encoding="utf-8").strip()
    fields = {"name": name, "imageName": IMAGE, "gpuTypeId": offer["id"], "gpuCount": 1, "cloudType": "SECURE",
              "containerDiskInGb": DISK_GB, "volumeInGb": 0, "ports": "22/tcp", "startSsh": True,
              "supportPublicIp": True, "env": [{"key": "PUBLIC_KEY", "value": pub}]}
    try:
        pod = gql("mutation($input: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $input) "
                  "{ id name costPerHr } }", {"input": fields})["podFindAndDeployOnDemand"]
    except Exception as e:                                  # outcome unknown: the watchdog reconciles
        update_lease(status="unknown", error=str(e)[:200])
        raise SystemExit(f"create failed ({e}); lease marked unknown - the watchdog will find and remove any pod")
    cost = float(pod.get("costPerHr") or 0)
    update_lease(status="running", pod_id=pod["id"], usd_per_hour=cost)
    if not 0 < cost <= MAX_USD_PER_HOUR:
        print(f"[pod] actual price ${cost}/h is outside (0, {MAX_USD_PER_HOUR}] - terminating")
        terminate_until_gone()
        raise SystemExit("terminated on price")
    print(f"[pod] created {pod['id']} ({offer['id']}, ${cost}/h), hard deadline "
          f"{datetime.fromisoformat(read_lease()['deadline']):%H:%M} UTC")


def status() -> None:
    lease = read_lease()
    print(json.dumps(lease, indent=1))
    if lease.get("pod_id") and lease.get("status") == "running":
        info = pod_info(lease["pod_id"])
        t = ssh_target(info)
        print("[pod]", (info or {}).get("desiredStatus"), "uptime", ((info or {}).get("runtime") or {}).get("uptimeInSeconds"), "s")
        if t:
            print(f"ssh -i {KEY} -p {t[1]} -o StrictHostKeyChecking=accept-new root@{t[0]}")


def keep_awake() -> None:
    """Ask Windows not to sleep while this process lives (released automatically on exit)."""
    if os.name == "nt":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def watchdog() -> None:
    """Hard stop at the deadline; also resolves an 'unknown' create by name. Never gives up
    while a pod may exist: terminate_until_gone retries, and the outer loop retries that.
    To extend a run, edit only "deadline" in lease.json (re-read every minute); the pod's own
    self-destruct at 8 h is the fixed outer bound."""
    keep_awake()
    while True:
        try:
            lease = read_lease()
            st = lease.get("status")
            if st in ("terminated",):
                return
            if st == "unknown" or now() >= datetime.fromisoformat(lease["deadline"]):
                print(f"{now().isoformat()} status={st} deadline={lease['deadline']} - terminating", flush=True)
                if terminate_until_gone():
                    print(f"{now().isoformat()} confirmed gone", flush=True)
                    return
        except Exception as e:                              # keep watching whatever happens
            print(f"{now().isoformat()} watchdog error: {type(e).__name__}: {e}", flush=True)
        time.sleep(60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["quote", "create", "status", "terminate", "watchdog"])
    ap.add_argument("--hours", type=float, default=6.0)
    a = ap.parse_args()
    if a.cmd == "quote":
        print(json.dumps(quote(), indent=1))
    elif a.cmd == "create":
        create(a.hours)
    elif a.cmd == "terminate":
        print("confirmed gone" if terminate_until_gone() else "NOT confirmed gone - check the RunPod console")
    else:
        {"status": status, "watchdog": watchdog}[a.cmd]()
