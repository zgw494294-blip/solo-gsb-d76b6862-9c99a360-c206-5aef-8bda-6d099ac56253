"""工作副本「两个已记录版本之间的改动比较」：纯函数、存储层与 HTTP 测试。

比较只读**已存快照**（不读取当前状态补值、不改副本与历史）：按批注完整
字符串 id 对齐（不按数组位置配对），仅比较 start / end / label / holder
四字段，顺序变化不算改动；清单按 id 的 Unicode 码点序排列，并返回两端
status。允许倒序与相同版本；未知副本或任一未记录版本 404，非法参数 422。
"""

import importlib

import pytest

from app.copies import (
    CopyNotFound,
    CopyStore,
    RevisionNotFound,
    diff_annotation_snapshots,
)

TEXT = "x" * 30


def ann(id, start, end, label="L", holder=None):
    a = {"id": id, "start": start, "end": end, "label": label}
    if holder is not None:
        a["holder"] = holder
    return a


@pytest.fixture()
def store(tmp_path):
    return CopyStore(tmp_path / "copies.db")


def _ids_holders(snap):
    return {a["id"]: a["holder"] for a in snap["annotations"]}


# ---------- 纯函数：对齐 / 字段 / 排序 / 方向 ----------

def _snap(annotations, status="open"):
    return {"status": status,
            "annotations": [dict(a, holder=a.get("holder")) for a in annotations]}


def test_diff_matches_by_full_string_id_not_array_position():
    # 两端数组顺序故意错位：不得按位置把 a 对到 b
    old = _snap([ann("a", 0, 2, "甲"), ann("b", 5, 7, "乙")])
    new = _snap([ann("b", 5, 8, "乙"), ann("a", 0, 3, "甲")])
    changes = diff_annotation_snapshots(old, new)
    assert {c["id"] for c in changes} == {"a", "b"}
    by_id = {c["id"]: c for c in changes}
    assert by_id["a"]["end"] == {"from": 2, "to": 3}
    assert by_id["b"]["end"] == {"from": 7, "to": 8}
    # 未变字段也附旧值新值，但不出现在 changed_fields
    assert by_id["a"]["changed_fields"] == ["end"]
    assert by_id["a"]["label"] == {"from": "甲", "to": "甲"}


def test_diff_order_only_change_is_not_a_change():
    old = _snap([ann("z", 0, 2), ann("a", 5, 7)])
    new = _snap([ann("a", 5, 7), ann("z", 0, 2)])  # 仅数组顺序相反
    assert diff_annotation_snapshots(old, new) == []


def test_diff_same_snapshot_is_empty():
    snap = _snap([ann("a", 0, 2, "甲", holder="alice")])
    assert diff_annotation_snapshots(snap, snap) == []


def test_diff_lists_all_four_fields_old_new_and_changed_names():
    old = _snap([ann("a", 0, 2, "旧标签", holder=None)])
    new = _snap([ann("a", 1, 4, "新标签", holder="bob")])
    changes = diff_annotation_snapshots(old, new)
    assert len(changes) == 1
    entry = changes[0]
    assert entry["id"] == "a"
    assert entry["changed_fields"] == ["start", "end", "label", "holder"]
    assert entry["start"] == {"from": 0, "to": 1}
    assert entry["end"] == {"from": 2, "to": 4}
    assert entry["label"] == {"from": "旧标签", "to": "新标签"}
    assert entry["holder"] == {"from": None, "to": "bob"}


def test_diff_partial_field_changes():
    old = _snap([ann("a", 0, 2, "L"), ann("b", 5, 7, "M", holder="alice")])
    new = _snap([ann("a", 0, 2, "L", holder="alice"),   # 仅 holder 变
                 ann("b", 5, 7, "M2", holder="alice")])  # 仅 label 变
    by_id = {c["id"]: c for c in diff_annotation_snapshots(old, new)}
    assert by_id["a"]["changed_fields"] == ["holder"]
    assert by_id["a"]["holder"] == {"from": None, "to": "alice"}
    assert by_id["a"]["start"] == {"from": 0, "to": 0}
    assert by_id["b"]["changed_fields"] == ["label"]
    assert by_id["b"]["label"] == {"from": "M", "to": "M2"}


def test_diff_holder_null_to_null_unchanged():
    old = _snap([ann("a", 0, 2)])
    new = _snap([ann("a", 0, 2)])
    assert diff_annotation_snapshots(old, new) == []


def test_diff_sorted_by_unicode_codepoint():
    old = _snap([ann("z", 0, 1), ann("a", 0, 1), ann("中", 0, 1),
                 ann("", 0, 1), ann("a\n", 0, 1)])
    new = _snap([ann("z", 0, 2), ann("a", 0, 2), ann("中", 0, 2),
                 ann("", 0, 2), ann("a\n", 0, 2)])
    ids = [c["id"] for c in diff_annotation_snapshots(old, new)]
    assert ids == sorted(ids)  # Unicode 码点序（Python str 序）
    assert ids == ["", "a", "a\n", "z", "中"]


def test_diff_special_ids_are_not_rewritten():
    special = ["", "审阅/1", "a%2Fb", "a\nb"]
    old = _snap([ann(i, k, k + 1) for k, i in enumerate(special)])
    new = _snap([ann(i, k, k + 2) for k, i in enumerate(special)])
    changes = diff_annotation_snapshots(old, new)
    assert [c["id"] for c in changes] == sorted(special)
    assert all(c["id"] in special for c in changes)


def test_diff_direction_follows_arguments():
    # 同一对快照反向调用：from/to 互换，changed_fields 不变
    old = _snap([ann("a", 0, 2, "旧")])
    new = _snap([ann("a", 1, 4, "新")])
    forward = diff_annotation_snapshots(old, new)[0]
    backward = diff_annotation_snapshots(new, old)[0]
    assert forward["start"] == {"from": 0, "to": 1}
    assert backward["start"] == {"from": 1, "to": 0}
    assert forward["label"] == {"from": "旧", "to": "新"}
    assert backward["label"] == {"from": "新", "to": "旧"}
    assert forward["changed_fields"] == backward["changed_fields"]


# ---------- 存储层：只读已存快照、状态与错误 ----------

def _build_history(store):
    """rev1 创建 → rev2 alice 领取 {a,b} → rev3 bob 改 c → rev4 结稿。"""
    cid = store.create(TEXT, [ann("a", 0, 4, "甲"), ann("b", 3, 7, "乙"),
                              ann("c", 12, 15, "丙")])["id"]
    store.claim(cid, "a", "alice")   # rev2：{a, b} 持有者 alice
    store.claim(cid, "c", "bob")     # rev3：c 归 bob
    store.modify(cid, "c", "bob", 13, 16, "丙改")  # rev4
    store.release(cid, "b", "alice")  # rev5
    store.release(cid, "c", "bob")    # rev6
    store.finalize(cid, 6)            # rev7
    return cid


def test_diff_revisions_forward(store):
    cid = _build_history(store)
    result = store.diff_revisions(cid, 1, 4)
    assert result["id"] == cid
    assert result["from_revision"] == 1 and result["to_revision"] == 4
    assert result["from_status"] == "open"
    assert result["to_status"] == "open"
    by_id = {c["id"]: c for c in result["changes"]}
    # a、b：仅 holder 变化（None → alice）
    assert by_id["a"]["changed_fields"] == ["holder"]
    assert by_id["a"]["holder"] == {"from": None, "to": "alice"}
    assert by_id["a"]["start"] == {"from": 0, "to": 0}
    assert by_id["b"]["holder"] == {"from": None, "to": "alice"}
    # c：holder 与范围 / label 全部变化
    assert by_id["c"]["changed_fields"] == ["start", "end", "label", "holder"]
    assert by_id["c"]["start"] == {"from": 12, "to": 13}
    assert by_id["c"]["end"] == {"from": 15, "to": 16}
    assert by_id["c"]["label"] == {"from": "丙", "to": "丙改"}
    assert by_id["c"]["holder"] == {"from": None, "to": "bob"}
    # 清单按 id 的 Unicode 码点序
    assert [c["id"] for c in result["changes"]] == ["a", "b", "c"]


def test_diff_revisions_reverse_direction(store):
    cid = _build_history(store)
    forward = store.diff_revisions(cid, 1, 4)
    backward = store.diff_revisions(cid, 4, 1)
    assert backward["from_status"] == "open" and backward["to_status"] == "open"
    fwd = {c["id"]: c for c in forward["changes"]}
    rev = {c["id"]: c for c in backward["changes"]}
    for ann_id in ("a", "b", "c"):
        for f in ("start", "end", "label", "holder"):
            assert rev[ann_id][f] == {
                "from": fwd[ann_id][f]["to"], "to": fwd[ann_id][f]["from"]}
        assert rev[ann_id]["changed_fields"] == fwd[ann_id]["changed_fields"]


def test_diff_revisions_same_revision_empty(store):
    cid = _build_history(store)
    for rev in (1, 4, 7):
        result = store.diff_revisions(cid, rev, rev)
        assert result["changes"] == []
        assert result["from_status"] == result["to_status"]


def test_diff_finalize_only_change_has_empty_changes(store):
    cid = _build_history(store)  # rev6 无持有者 open → rev7 finalized
    result = store.diff_revisions(cid, 6, 7)
    assert result["changes"] == []  # 批注四字段未变
    assert result["from_status"] == "open"
    assert result["to_status"] == "finalized"
    # 反向：finalized → open，清单同样为空
    back = store.diff_revisions(cid, 7, 6)
    assert back["changes"] == []
    assert back["from_status"] == "finalized"
    assert back["to_status"] == "open"


def test_diff_claim_release_round_trip_is_empty(store):
    cid = store.create(TEXT, [ann("a", 0, 3)])["id"]
    store.claim(cid, "a", "alice")    # rev2
    store.release(cid, "a", "alice")  # rev3：holder 回到 None
    result = store.diff_revisions(cid, 1, 3)
    assert result["changes"] == []


def test_diff_reads_only_stored_snapshots_not_current_state(store):
    cid = _build_history(store)
    # 结稿后当前状态停留在 rev7；比较 rev1/rev4 不受当前状态影响
    assert store.get(cid)["revision"] == 7
    result = store.diff_revisions(cid, 1, 4)
    assert result["to_status"] == "open"  # 不被当前 finalized 污染
    # 比较是只读操作：不改副本、不补写历史
    before = store.list_revisions(cid)
    store.diff_revisions(cid, 1, 7)
    assert store.list_revisions(cid) == before
    assert store.get(cid)["revision"] == 7


def test_diff_unknown_copy_and_unrecorded_revision_404(store):
    with pytest.raises(CopyNotFound):
        store.diff_revisions("deadbeef", 1, 2)
    cid = _build_history(store)
    for r1, r2 in ((0, 1), (1, 0), (8, 9), (1, 99), (99, 1)):
        with pytest.raises(RevisionNotFound):
            store.diff_revisions(cid, r1, r2)


def test_diff_old_db_baseline_no_invented_history(tmp_path):
    # 旧库仅有当前 revision=5 的 baseline：缺失的早期版本不推测、返回 404
    import sqlite3
    db = tmp_path / "copies.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE copies (
            id TEXT PRIMARY KEY, text TEXT NOT NULL,
            revision INTEGER NOT NULL DEFAULT 1);
        CREATE TABLE annotations (
            copy_id TEXT NOT NULL, ann_id TEXT NOT NULL, pos INTEGER NOT NULL,
            start INTEGER NOT NULL, end INTEGER NOT NULL, label TEXT NOT NULL,
            holder TEXT, PRIMARY KEY (copy_id, ann_id));
    """)
    conn.execute("INSERT INTO copies(id, text, revision) VALUES('c1', ?, 5)",
                 (TEXT,))
    conn.execute("INSERT INTO annotations VALUES('c1', 'a', 0, 0, 3, 'L', NULL)")
    conn.commit()
    conn.close()

    s = CopyStore(db)
    with pytest.raises(RevisionNotFound):
        s.diff_revisions("c1", 1, 5)  # 不编造 revision 1..4
    assert s.diff_revisions("c1", 5, 5)["changes"] == []


# ---------- HTTP 端到端 ----------

@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("COPY_DB_PATH", str(tmp_path / "copies.db"))
    import app.main as main
    importlib.reload(main)
    from fastapi.testclient import TestClient
    with TestClient(main.app) as c:
        yield c
    main._store = None


def _create(client, anns):
    r = client.post("/api/copies", json={"text": TEXT, "annotations": anns})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def test_http_diff_forward_reverse_same_and_finalize(client):
    cid = _create(client, [ann("a", 0, 4, "甲"), ann("b", 3, 7, "乙"),
                   ann("c", 12, 15, "丙")])
    client.post(f"/api/copies/{cid}/annotations/a/claim",
                json={"reviewer": "alice"})
    client.post(f"/api/copies/{cid}/annotations/c/claim",
                json={"reviewer": "bob"})
    client.patch(f"/api/copies/{cid}/annotations/c",
                 json={"reviewer": "bob", "start": 13, "end": 16,
                       "label": "丙改"})
    client.post(f"/api/copies/{cid}/annotations/b/release",
                json={"reviewer": "alice"})
    client.post(f"/api/copies/{cid}/annotations/c/release",
                json={"reviewer": "bob"})
    client.post(f"/api/copies/{cid}/finalize",
                json={"expected_revision": 6})

    # 正向 1 → 4
    r = client.get(f"/api/copies/{cid}/revisions/diff",
                   params={"from_revision": 1, "to_revision": 4})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["from_revision"] == 1 and body["to_revision"] == 4
    assert body["from_status"] == body["to_status"] == "open"
    assert [c["id"] for c in body["changes"]] == ["a", "b", "c"]
    c = {x["id"]: x for x in body["changes"]}["c"]
    assert c["changed_fields"] == ["start", "end", "label", "holder"]
    assert c["start"] == {"from": 12, "to": 13}
    assert c["end"] == {"from": 15, "to": 16}
    assert c["label"] == {"from": "丙", "to": "丙改"}
    assert c["holder"] == {"from": None, "to": "bob"}
    a = {x["id"]: x for x in body["changes"]}["a"]
    assert a["changed_fields"] == ["holder"]
    assert a["holder"] == {"from": None, "to": "alice"}

    # 反向 4 → 1：旧值 / 新值方向随入参翻转
    r = client.get(f"/api/copies/{cid}/revisions/diff",
                   params={"from_revision": 4, "to_revision": 1})
    assert r.status_code == 200
    body = r.json()
    c = {x["id"]: x for x in body["changes"]}["c"]
    assert c["end"] == {"from": 16, "to": 15}
    assert c["holder"] == {"from": "bob", "to": None}

    # 相同版本 → 空清单
    r = client.get(f"/api/copies/{cid}/revisions/diff",
                   params={"from_revision": 3, "to_revision": 3})
    assert r.status_code == 200
    assert r.json()["changes"] == []

    # 结稿变化（6 → 7）：批注未变 → 空清单，但两端 status 均返回
    r = client.get(f"/api/copies/{cid}/revisions/diff",
                   params={"from_revision": 6, "to_revision": 7})
    assert r.status_code == 200
    body = r.json()
    assert body["changes"] == []
    assert body["from_status"] == "open"
    assert body["to_status"] == "finalized"


def test_http_diff_order_only_change_empty(client):
    # 创建顺序 z → a（几何顺序 a → z）；历史回看按原序，两端位置互换
    # 但四字段相同 → 空清单
    cid = _create(client, [ann("z", 15, 18), ann("a", 0, 3)])
    client.post(f"/api/copies/{cid}/annotations/z/claim",
                json={"reviewer": "alice"})  # rev2：仅 holder 变
    r1 = client.get(f"/api/copies/{cid}/revisions/diff",
                    params={"from_revision": 1, "to_revision": 2}).json()
    assert [c["id"] for c in r1["changes"]] == ["z"]
    assert r1["changes"][0]["changed_fields"] == ["holder"]
    # rev2 与 rev2 比较：即使两版内部数组顺序都与几何序不同，也不算改动
    r2 = client.get(f"/api/copies/{cid}/revisions/diff",
                    params={"from_revision": 1, "to_revision": 1}).json()
    assert r2["changes"] == []


def test_http_diff_special_ids_aligned_by_full_string(client):
    # TestClient 会把路径百分号解码两次（见 test_copy_special_ids_http.py
    # 模块说明），含 "/" 与字面 "%2F" 的 id 在真实 uvicorn 测试中覆盖；
    # 这里用空串与换行 id 验证按完整字符串对齐、不改写 id。
    cid = _create(client, [ann("", 0, 2, "空"), ann("a\nb", 16, 18, "换")])
    assert client.post(f"/api/copies/{cid}/annotations//claim",
                       json={"reviewer": "r0"}).status_code == 200
    assert client.post(f"/api/copies/{cid}/annotations/a%0Ab/claim",
                       json={"reviewer": "r1"}).status_code == 200
    client.patch(f"/api/copies/{cid}/annotations/",
                 json={"reviewer": "r0", "start": 0, "end": 1,
                       "label": "空改"})
    body = client.get(f"/api/copies/{cid}/revisions/diff",
                      params={"from_revision": 1, "to_revision": 4}).json()
    ids = [c["id"] for c in body["changes"]]
    assert ids == sorted(ids)  # Unicode 码点序
    # id 原样出现在结果中（空串、换行均不改写）
    assert ids == ["", "a\nb"]
    by_id = {c["id"]: c for c in body["changes"]}
    assert by_id[""]["changed_fields"] == ["end", "label", "holder"]
    assert by_id[""]["label"] == {"from": "空", "to": "空改"}
    assert by_id["a\nb"]["changed_fields"] == ["holder"]


def test_http_diff_unknown_copy_and_revision_404(client):
    r = client.get("/api/copies/deadbeef/revisions/diff",
                   params={"from_revision": 1, "to_revision": 2})
    assert r.status_code == 404

    cid = _create(client, [ann("a", 0, 3)])  # 仅记录 revision=1
    for params in ({"from_revision": 2, "to_revision": 1},
                   {"from_revision": 1, "to_revision": 2},
                   {"from_revision": 99, "to_revision": 1}):
        r = client.get(f"/api/copies/{cid}/revisions/diff", params=params)
        assert r.status_code == 404, params


@pytest.mark.parametrize("params", [
    {"from_revision": 0, "to_revision": 1},
    {"from_revision": 1, "to_revision": 0},
    {"from_revision": -1, "to_revision": 1},
    {"from_revision": 1, "to_revision": 1.5},
    {"from_revision": "abc", "to_revision": 1},
    {"from_revision": True, "to_revision": 1},   # 布尔不是合法版本号
    {"from_revision": 1, "to_revision": False},
    {"to_revision": 1},                          # 缺少 from_revision
    {"from_revision": 1},                        # 缺少 to_revision
    {},
])
def test_http_diff_invalid_params_422(client, params):
    cid = _create(client, [ann("a", 0, 3)])
    r = client.get(f"/api/copies/{cid}/revisions/diff", params=params)
    assert r.status_code == 422, params


def test_http_diff_does_not_mutate_state(client):
    cid = _create(client, [ann("a", 0, 4), ann("b", 3, 7)])
    client.post(f"/api/copies/{cid}/annotations/a/claim",
                json={"reviewer": "alice"})
    revs_before = client.get(f"/api/copies/{cid}/revisions").json()
    snap_before = client.get(f"/api/copies/{cid}").json()
    client.get(f"/api/copies/{cid}/revisions/diff",
               params={"from_revision": 1, "to_revision": 2})
    client.get(f"/api/copies/{cid}/revisions/diff",
               params={"from_revision": 2, "to_revision": 1})
    assert client.get(f"/api/copies/{cid}/revisions").json() == revs_before
    assert client.get(f"/api/copies/{cid}").json() == snap_before
