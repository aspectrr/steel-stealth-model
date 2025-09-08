# checker_service.py
from jsonschema import validate, ValidationError

SCHEMA = {...}  # your JSON schema

@app.post("/check")
def check_result(payload: {"url":str, "parsed_json":dict, "gold":dict=None}):
    # 1) syntactic validity
    try:
        validate(instance=payload["parsed_json"], schema=SCHEMA)
        syntactic = True
    except ValidationError:
        syntactic = False

    # 2) semantic check: compare fields to gold OR run site action
    if syntactic:
        # either compare to gold labels or emulate submit + readback
        success = semantic_check(payload["parsed_json"], payload.get("gold"))
    else:
        success = False

    return {"valid_json": syntactic, "success": bool(success)}
