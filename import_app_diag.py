import json
import traceback

result = {"stage": "start"}
try:
    result["stage"] = "before_import"
    import app  # noqa: F401
    result["stage"] = "after_import"
except BaseException as e:
    result["stage"] = "exception"
    result["exc_type"] = type(e).__name__
    result["exc"] = str(e)
    result["trace"] = traceback.format_exc()

with open("import_app_diag.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
