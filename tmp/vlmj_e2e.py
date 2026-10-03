# SPDX-License-Identifier: Apache-2.0
"""Throwaway end-to-end smoke against a running vlm-jlens-demo service (default tiny :8787)."""
import time

import httpx

BASE = "http://127.0.0.1:8787"
c = httpx.Client(base_url=BASE, timeout=180)


def j(r):
    r.raise_for_status()
    return r.json()


def poll(job_id, timeout=120.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        state = j(c.get(f"/api/jobs/{job_id}"))
        if state["status"] in ("done", "error"):
            assert state["status"] == "done", state.get("error")
            return state["result"]
        time.sleep(0.3)
    raise SystemExit(f"job {job_id} timed out")


meta = j(c.get("/api/meta"))
layers = meta["lens"]["source_layers"]
assert meta["mode"] == "tiny", meta
assert meta["lens"]["available"] is True, meta["lens"]
print("meta:", meta["mode"], "| lens layers:", layers)

resp = j(c.get("/api/samples"))
samples = resp["samples"] if isinstance(resp, dict) else resp
sample = samples[0]
payload = {"prompt": sample.get("prompt")}
for key in ("image_b64", "image", "image_path"):
    if sample.get(key):
        payload[key] = sample[key]
        break
sess = j(c.post("/api/session", json=payload))
sid = sess["session_id"]
image = sess["image"]
assert image["count"] == 576 and image["grid"] == [24, 24], image
assert (image["start"], image["end"]) == (2, 578), image
print("session:", sid, "| image tokens:", image["count"], f"[{image['start']}, {image['end']})")

gen = poll(c.post("/api/generate", json={"session_id": sid, "max_new_tokens": 8}).json()["job_id"])
tokens = [t["str"] for t in gen["tokens"]]
print("caption:", repr(gen["caption"]), "| tokens:", len(tokens))

lens = poll(
    c.post(
        "/api/lens",
        json={"session_id": sid, "targets": [{"kind": "gen", "i": 0}], "layers": layers,
              "topk": 5, "track": [tokens[0]]},
    ).json()["job_id"]
)
target = lens["targets"][0]
assert len(target["per_layer"]) == len(layers)
row = target["model_row"]["topk"][0]
assert row["rank"] >= 1
print("lens: layer", target["per_layer"][0]["layer"],
      "top1:", target["per_layer"][0]["topk"][0]["str"], "| model_row rank:", row["rank"])

att = poll(c.post("/api/attribution", json={"session_id": sid, "layer": layers[0],
                                            "token": tokens[0]}).json()["job_id"])
assert len(att["grid"]) == 24 and len(att["grid"][0]) == 24
assert att["grid_rank"] is not None and att["vmax"] > att["vmin"]
print("attribution: vmin %.4g vmax %.4g" % (att["vmin"], att["vmax"]))

att2 = poll(c.post("/api/attribution", json={"session_id": sid, "layer": layers[0],
                                             "token": tokens[0], "metric": "attn_rollout",
                                             "rollout": True}).json()["job_id"])
assert att2["grid_rank"] is None
flat = [v for r in att2["grid"] for v in r]
assert all(0.0 <= v <= 1.0 for v in flat) and att2["vmax"] > 0.0
print("attn_rollout: vmax %.4g sum %.4f" % (att2["vmax"], sum(flat)))

ko = poll(c.post("/api/knockout", json={"session_id": sid, "patches": list(range(16)),
                                        "mode": "zero", "target_tokens": [tokens[0]]}).json()["job_id"])
assert ko["outcomes"]
print("knockout:", [(o["token"], o["class"], round(o["delta"], 3)) for o in ko["outcomes"]])

steer = poll(c.post("/api/steer", json={"session_id": sid, "layer": layers[0], "mode": "add",
                                        "token": tokens[0], "alpha": 0.0}).json()["job_id"])
assert steer["caption_after"] == steer["caption_before"], "alpha=0 changed the caption"
print("steer alpha=0 identity ok | diagnostics:", steer["diagnostics"])

twin = j(c.post("/api/session", json={"variant": "no_image",
                                      "prompt": "USER: Describe this image.\nASSISTANT:"}))
assert twin["variant"] == "no_image" and twin["image"] is None
gen2 = poll(c.post("/api/generate", json={"session_id": twin["session_id"],
                                          "max_new_tokens": 8}).json()["job_id"])
r = c.post("/api/attribution", json={"session_id": twin["session_id"], "layer": layers[0],
                                     "token": tokens[0]})
assert r.status_code == 409, r.status_code
print("twin:", repr(gen2["caption"]), "| attribution -> 409 ok")

print("E2E_PASS")
