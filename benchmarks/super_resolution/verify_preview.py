"""Exercise post-login HTTP → control socket → inference → native upload."""
import argparse
import json
import time
import urllib.parse
import urllib.request
import urllib.error


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--url",default="http://127.0.0.1:18782")
    parser.add_argument("--require-video",action="store_true",help="Require advancing decoded video and bounded queues during switching")
    args=parser.parse_args()
    evidence=[]
    deadline=time.monotonic()+25
    while True:
        try:
            with urllib.request.urlopen(args.url+"/dashboard",timeout=2) as response:
                assert response.status==200
            break
        except (urllib.error.URLError,OSError):
            if time.monotonic()>deadline:raise
            time.sleep(.1)
    with urllib.request.urlopen(args.url+"/events",timeout=60) as events:
        # The HTTP listener can be ready before the first GPU-presented frame
        # and enhancement snapshot. Wait for the control surface to be ready.
        while True:
            line=events.readline().decode()
            if not line:raise RuntimeError("SSE disconnected during startup")
            if line.startswith("data:"):
                data=json.loads(line[5:].strip())
                if isinstance(data,dict) and data.get("enhancement"):break
        previous_state=data["enhancement"]
        last_video_emitted=(data.get("playback") or {}).get("emitted",-1)
        last_video_delivered=(data.get("playback") or {}).get("delivered",-1)
        def post_model(key):
            body=urllib.parse.urlencode({"action":"enhancement","model":key}).encode()
            with urllib.request.urlopen(args.url+"/action",data=body,timeout=10) as response:
                assert response.status==200

        def wait_displayed(key,minimum_generation):
            nonlocal last_video_emitted,last_video_delivered
            start=time.monotonic()
            while time.monotonic()-start<90:
                line=events.readline().decode()
                if not line:raise RuntimeError("SSE disconnected")
                if not line.startswith("data:"):continue
                data=json.loads(line[5:].strip())
                if not isinstance(data,dict) or "enhancement" not in data:continue
                state=data["enhancement"]
                if state.get("state")=="error":raise RuntimeError(state["error"])
                playback=data.get("playback") or {}
                if args.require_video:
                    if playback.get("error"):raise RuntimeError(playback["error"])
                    if state.get("pending",0)>1 or playback.get("pending",0)>1:
                        raise AssertionError("Recorded playback or inference queue grew beyond one pending item")
                    if not (playback.get("emitted",-1)>last_video_emitted and
                            playback.get("delivered",-1)>last_video_delivered):continue
                if (state.get("generation",-1)>=minimum_generation and
                        state.get("selected")==key and state.get("displayed")==key):
                    if args.require_video:
                        last_video_emitted=playback["emitted"];last_video_delivered=playback["delivered"]
                    evidence.append(state);print(json.dumps(dict(state,playback=playback) if args.require_video else state),flush=True);return state
            else:raise RuntimeError(f"{key} never reached displayed state")

        for key in ["bicubic","edsr","imdn","safmn","off"]:
            generation=previous_state["generation"]+int(previous_state["selected"]!=key)
            post_model(key)
            previous_state=wait_displayed(key,generation)
            if key=="bicubic":
                # Deliberately send both requests without waiting for a render
                # between them. A static fixture must be submitted again even
                # if the final selected model is the same as before.
                old_generation=previous_state["generation"]
                post_model("off");post_model(key)
                previous_state=wait_displayed(key,old_generation+2)
                print("VERIFIED rapid Original→same-model cycle",flush=True)
        try:
            post_model("__not_installed__")
        except urllib.error.HTTPError as exc:
            assert exc.code==400
            print("VERIFIED unavailable model rejected by session HTTP endpoint",flush=True)
        else:
            raise AssertionError("Unavailable model request was accepted")
    updates="recorded-video updates" if args.require_video else "static-frame redraw"
    print(f"VERIFIED: actual control dispatch, all installed neural models, {updates}, rapid reselection and bypass",flush=True)
    if args.require_video:print("VERIFIED: advancing recorded-video tiles and bounded source/inference queues during every model switch",flush=True)


if __name__=="__main__":main()
