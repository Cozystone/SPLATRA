"""Heuristic (no-LLM) multi-object routing: '지구와 달' -> explain_scene, not one blob.

The offline heuristic used to send any multi-thing phrase as a SINGLE generate_3d_object
call, so '지구와 달' (earth and moon) rendered as one merged globe. It now detects a
multi-object request and emits explain_scene — with two guards proven here:
  * Korean 와/과 only splits at a MORPHEME BOUNDARY (trailing space), so 사과(apple),
    고양이(cat), 종이(paper), 과학(science) are NOT split apart.
  * a bare English "and" only splits a real list (comma-punctuated, or every side names a
    shape), so "black and white cube" stays ONE object.
"""

from atanor_core.llm.heuristic import HeuristicLLM, split_objects

_TOOLS = [
    {"type": "function", "function": {"name": n}}
    for n in ("generate_3d_object", "render_knowledge_hologram", "explain_scene")
]


def _call(text):
    out = HeuristicLLM().chat([{"role": "user", "content": text}], _TOOLS)
    return out["tool_calls"][0]


def test_multi_object_splits():
    assert split_objects("지구와 달") == ["지구", "달"]
    assert split_objects("지구와 달을 보여줘") == ["지구", "달"]
    assert split_objects("sun, earth and moon") == ["sun", "earth", "moon"]
    assert split_objects("빨간 공과 파란 공 만들어줘") == ["빨간 공", "파란 공"]
    assert split_objects("고양이 그리고 강아지") == ["고양이", "강아지"]  # 고양이 kept whole
    assert split_objects("커피잔과 사과") == ["커피잔", "사과"]           # 사과 kept whole
    assert split_objects("a red sphere and a blue cube") == ["red sphere", "blue cube"]


def test_single_object_never_splits():
    # morpheme-boundary + English-adjective traps must all stay ONE object
    for text in ["사과 만들어줘", "사과를 보여줘", "고양이", "종이", "과학 물체",
                 "black and white cube", "파란 토러스 만들어줘", "피카츄"]:
        assert split_objects(text) == [], text


def test_multi_object_routes_to_explain_scene():
    call = _call("지구와 달")
    assert call["name"] == "explain_scene"
    prompts = [o["prompt"] for o in call["arguments"]["objects"]]
    assert prompts == ["지구", "달"]
    assert call["arguments"]["links"] == []          # no relation word -> no links


def test_relation_word_links_objects():
    call = _call("지구와 달 비교해줘")     # 비교(compare) is a relation hint
    assert call["name"] == "explain_scene"
    prompts = [o["prompt"] for o in call["arguments"]["objects"]]
    assert prompts == ["지구", "달"]
    assert call["arguments"]["links"] == [["지구", "달"]]


def test_single_object_still_generates():
    call = _call("파란 토러스 만들어줘")
    assert call["name"] == "generate_3d_object"


def test_multi_object_ignored_without_scene_tool():
    # if the caller does not offer explain_scene, fall back to single generation
    tools = [{"type": "function", "function": {"name": "generate_3d_object"}}]
    out = HeuristicLLM().chat([{"role": "user", "content": "지구와 달"}], tools)
    assert out["tool_calls"][0]["name"] == "generate_3d_object"


def test_knowledge_graph_still_routes():
    call = _call("show a knowledge graph with 24 nodes")
    assert call["name"] == "render_knowledge_hologram"
