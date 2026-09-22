"""FastAPI application: industrial event-log clock calibration.

Endpoints
---------
* ``GET  /health`` -- liveness probe
* ``POST /align``  -- calibrate a drifted target JSONL against a reference

The ``/align`` endpoint accepts two equivalent content types:

* ``application/json`` -- body
  ``{"reference": [...], "target": [...], "time_field": ..., ...}``;
  ``reference`` / ``target`` may also be raw JSONL strings.  Optional
  ``anchors`` is a list of ``{"reference_index": i, "target_index": j}``
  manual matches.
* ``multipart/form-data`` -- ``reference`` and ``target`` uploaded as
  ``.jsonl`` files plus optional form fields (``time_field`` ...); anchors go
  in a repeatable ``anchors`` JSON form part (either one JSON array or one
  JSON object per anchor).
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .io import AlignConfig, InputError, ManualAnchorSpec, run_alignment

app = FastAPI(
    title="Industrial Event-Log Clock Alignment API",
    version=__version__,
    description=(
        "Aligns a clock-drifting event log (JSONL) against a reference log "
        "with monotonic label/DP matching and a piecewise-linear clock map."
    ),
)


# --- request / response schemas (OpenAPI) -----------------------------------


class AlignRequest(BaseModel):
    reference: Any = Field(
        ..., description="Reference log: JSON array of events or JSONL string"
    )
    target: Any = Field(
        ..., description="Log to calibrate: JSON array of events or JSONL string"
    )
    time_field: Optional[str] = Field(
        default=None,
        description="Timestamp field name (auto-detected if omitted)",
    )
    label_field: Optional[str] = Field(
        default=None,
        description="Event tag field name (auto-detected if omitted)",
    )
    iterations: int = Field(
        default=3, ge=1, le=10,
        description="Match/fit refinement iterations",
    )
    time_weight: float = Field(
        default=0.5, ge=0.0, le=2.0,
        description="Weight of the temporal Gaussian term vs. the label bonus",
    )
    anchors: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description=(
            "Manual anchors that must be matched; each item is "
            "{reference_index, target_index}. They split the streams into "
            "intervals and automatic matching cannot cross them."
        ),
    )


# Manual anchors are documented once and referenced from both the JSON body
# schema and the multipart form description below.
_ANCHOR_ITEM_SCHEMA = {
    "type": "object",
    "required": ["reference_index", "target_index"],
    "properties": {
        "reference_index": {
            "type": "integer", "minimum": 0,
            "description": "0-based index into the reference stream",
        },
        "target_index": {
            "type": "integer", "minimum": 0,
            "description": "0-based index into the target stream",
        },
    },
    "additionalProperties": True,
}

_ALIGN_OPENAPI_EXTRA = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "schema": {"$ref": "#/components/schemas/AlignRequest"},
            },
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["reference", "target"],
                    "properties": {
                        "reference": {"type": "string", "format": "binary"},
                        "target": {"type": "string", "format": "binary"},
                        "time_field": {"type": "string"},
                        "label_field": {"type": "string"},
                        "iterations": {"type": "integer",
                                       "minimum": 1, "maximum": 10,
                                       "default": 3},
                        "time_weight": {"type": "number",
                                        "minimum": 0, "maximum": 2,
                                        "default": 0.5},
                        "anchors": {
                            "type": "array",
                            "description": (
                                "Manual anchors. Each part is JSON text -- "
                                "either one part holding an array of "
                                "{reference_index, target_index} objects, or "
                                "one such object per repeatable 'anchors' "
                                "part."
                            ),
                            "items": {"$ref": "#/components/schemas/AnchorItem"},
                        },
                    },
                },
            },
        },
    },
}


def _custom_openapi() -> dict[str, Any]:
    """Serve the hand-written dual content-type request body for /align.

    The endpoint parses a raw ``Request`` (JSON *or* multipart), so FastAPI
    cannot derive the body schema from a typed parameter; register the
    ``AlignRequest`` / ``AnchorItem`` component schemas here explicitly.
    """
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    components = schema.setdefault("components", {}).setdefault("schemas", {})
    components["AnchorItem"] = _ANCHOR_ITEM_SCHEMA
    request_schema = AlignRequest.model_json_schema(
        ref_template="#/components/schemas/{model}"
    )
    # Point the anchors array at the shared AnchorItem component.
    request_schema["properties"]["anchors"]["items"] = {
        "$ref": "#/components/schemas/AnchorItem"
    }
    components["AlignRequest"] = request_schema
    app.openapi_schema = schema
    return schema


app.openapi = _custom_openapi  # type: ignore[assignment]


def _anchor_specs(items: Any, *, where: str) -> list[ManualAnchorSpec]:
    """Validate raw anchor payload shape into ``ManualAnchorSpec`` objects."""
    if items is None:
        return []
    if not isinstance(items, list):
        raise InputError(f"{where}: must be a list of anchor objects")
    specs: list[ManualAnchorSpec] = []
    for pos, item in enumerate(items):
        loc = f"{where}[{pos}]"
        if not isinstance(item, dict):
            raise InputError(f"{loc}: must be an object")
        if "reference_index" not in item:
            raise InputError(f"{loc}: missing field 'reference_index'")
        if "target_index" not in item:
            raise InputError(f"{loc}: missing field 'target_index'")
        ri, ti = item["reference_index"], item["target_index"]
        if isinstance(ri, bool) or not isinstance(ri, int) or ri < 0:
            raise InputError(
                f"{loc}.reference_index: expected a non-negative integer, "
                f"got {ri!r}"
            )
        if isinstance(ti, bool) or not isinstance(ti, int) or ti < 0:
            raise InputError(
                f"{loc}.target_index: expected a non-negative integer, "
                f"got {ti!r}"
            )
        specs.append(ManualAnchorSpec(reference_index=ri, target_index=ti))
    return specs


def _to_config(
    time_field: Optional[str],
    label_field: Optional[str],
    iterations: int,
    time_weight: float,
    anchors: Optional[list[ManualAnchorSpec]] = None,
) -> AlignConfig:
    return AlignConfig(
        time_field=time_field or None,
        label_field=label_field or None,
        iterations=iterations,
        time_weight=time_weight,
        manual_anchors=list(anchors or []),
    )


# --- endpoints --------------------------------------------------------------


@app.get("/health", tags=["ops"])
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "clock-align", "version": __version__}


@app.post("/align", tags=["alignment"], response_model=None,
          openapi_extra=_ALIGN_OPENAPI_EXTRA)
async def align(request: Request) -> JSONResponse:
    """Calibrate the target log. Works with JSON or multipart/form-data."""
    content_type = request.headers.get("content-type", "")

    try:
        if content_type.startswith("application/json"):
            try:
                body = await request.json()
            except Exception as exc:  # noqa: BLE001
                raise InputError(f"invalid JSON body: {exc}") from exc
            if not isinstance(body, dict):
                raise InputError(
                    "JSON body must be an object with 'reference' and 'target'"
                )
            if "reference" not in body or "target" not in body:
                raise InputError("body requires 'reference' and 'target'")
            cfg = _to_config(
                body.get("time_field"),
                body.get("label_field"),
                int(body.get("iterations", 3)),
                float(body.get("time_weight", 0.5)),
                _anchor_specs(body.get("anchors"), where="anchors"),
            )
            reference, target = body["reference"], body["target"]

        elif content_type.startswith("multipart/form-data"):
            form = await request.form()
            if "reference" not in form or "target" not in form:
                raise InputError(
                    "form requires 'reference' and 'target' file parts"
                )
            reference_file: UploadFile = form["reference"]
            target_file: UploadFile = form["target"]
            reference = await reference_file.read()
            target = await target_file.read()

            # ``form.getlist("anchors")`` accepts either a single JSON array
            # part or one JSON object per repeatable part; blank parts ignored.
            anchor_items: list[Any] = []
            for pno, part in enumerate(form.getlist("anchors")):
                if not isinstance(part, str):
                    raise InputError(
                        f"anchors form part #{pno}: must be a JSON text "
                        "field, not a file upload"
                    )
                if not part.strip():
                    continue
                try:
                    parsed = json.loads(part)
                except json.JSONDecodeError as exc:
                    raise InputError(
                        f"anchors form part #{pno}: invalid JSON: {exc}"
                    ) from exc
                if isinstance(parsed, list):
                    anchor_items.extend(parsed)
                elif isinstance(parsed, dict):
                    anchor_items.append(parsed)
                else:
                    raise InputError(
                        f"anchors form part #{pno}: expected a JSON array or "
                        "object"
                    )
            cfg = _to_config(
                form.get("time_field"),
                form.get("label_field"),
                int(form.get("iterations", 3)),
                float(form.get("time_weight", 0.5)),
                _anchor_specs(anchor_items, where="anchors"),
            )
        else:
            raise HTTPException(
                status_code=415,
                detail=(
                    "unsupported content type; use application/json or "
                    "multipart/form-data"
                ),
            )

        started = time.perf_counter()
        result = run_alignment(reference, target, cfg)
        result["summary"]["elapsed_seconds"] = round(
            time.perf_counter() - started, 6
        )
        return JSONResponse(result)

    except InputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
