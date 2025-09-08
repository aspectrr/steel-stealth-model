# rl_env.py (pseudocode)
import requests
import json

NAVIGATOR = "http://navigator/navigate"
CHECKER = "http://checker/check"
MODEL_API = "http://model/infer"  # could be local HF endpoint or TGI/vLLM endpoint

def build_prompt(markdown, metadata, required_keys):
    return f"### Page metadata:\n{json.dumps(metadata)}\n\n### Page content:\n{markdown}\n\n### Instruction:\nReturn JSON with keys: {required_keys}\n"

def step(url):
    # 1) get page from navigator (spins a browser)
    nav = requests.get(NAVIGATOR, params={"url": url}).json()
    prompt = build_prompt(nav["markdown"], nav["metadata"], ["name","price","desc"])
    # 2) ask policy to generate (model API)
    out = requests.post(MODEL_API, json={"prompt": prompt, "max_tokens": 512}).json()["text"]
    # 3) try parse JSON
    try:
        parsed = json.loads(out)
        valid_json = True
    except Exception:
        parsed = None
        valid_json = False

    # 4) check result
    if not valid_json:
        reward = 0
        info = {"valid_json": False}
    else:
        check = requests.post(CHECKER, json={"url": url, "parsed_json": parsed}).json()
        reward = 1 if check["success"] else 0
        info = check

    return out, reward, info
