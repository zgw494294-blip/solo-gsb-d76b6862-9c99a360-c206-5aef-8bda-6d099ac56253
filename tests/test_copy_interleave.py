"""工作副本成功响应必须对应**本次提交**的快照：可控交错回归。

历史缺陷：领取已提交 revision=2，响应返回前同一审阅者又释放并提交
revision=3，领取响应却重读数据库拿到 revision=3（revision 与持有者被后续
请求污染）。根因是 ``_apply`` 在解锁后才 ``self.get`` 构造响应。

本组测试通过 ``CopyStore._after_commit``（锁内、提交后、响应返回前的可注入
交错点）**确定性地**把「下一次完整变更」插进「本次提交已完成、响应尚未
返回」的窗口，覆盖：

1. 领取后释放：领取响应仍为 revision=2 且整组由本人持有；
2. 连续修改：第一次修改响应只含该次修改，不含交错进来的第二次修改；
3. 重复领取后释放：幂等领取返回判定时快照，不增加版本或历史。

每次都按**完整批注 id** 把响应与「该版历史快照」逐条核对
（start/end/label/holder），并核对 revision/status/groups；并发操作仍逐次
完整执行，交错中的失败请求不改副本或历史。
"""

import threading

import pytest

from app.copies import CopyStore, GroupHeldError

TEXT = "x" * 30


def ann(id, start, end, label="L"):
    return {"id": id, "start": start, "end": end, "label": label}


@pytest.fixture()
def store(tmp_path):
    return CopyStore(tmp_path / "copies.db")


def _by_id(snap):
    """按完整字符串 id 建索引（不按数组位置）：id→(start,end,label,holder)。"""
    return {a["id"]: (a["start"], a["end"], a["label"], a["holder"])
            for a in snap["annotations"]}


def _assert_response_matches_history(store, cid, resp, revision, status):
    """响应必须与该版历史快照一致：revision/status/批注四字段/groups。

    响应按几何序、历史按创建序返回批注，故按**完整 id** 对齐逐条核对。
    """
    assert resp["revision"] == revision
    assert resp["status"] == status
    hist = store.get_revision(cid, revision)
    assert hist["status"] == status
    assert _by_id(resp) == _by_id(hist)
    assert resp["groups"] == hist["groups"]


def _one_shot_interleave(store, action):
    """仅让下一次成功 _apply 在返回前执行一次 action；嵌套调用不再交错。"""
    armed = {"on": True}
    original = store._after_commit

    def hook(copy_id, op, changed, snapshot):
        if armed["on"]:
            armed["on"] = False  # 交错动作内部的 _apply 不再触发
            action()

    store._after_commit = hook
    return original


# ---------- 1. 领取已提交 rev2，响应返回前同一审阅者释放提交 rev3 ----------

def test_claim_response_is_rev2_even_if_release_commits_before_return(store):
    cid = store.create(TEXT, [ann("a", 0, 4), ann("b", 3, 7),
                              ann("c", 12, 15)])["id"]
    original = _one_shot_interleave(
        store, lambda: store.release(cid, "b", "alice"))  # rev3：整组释放
    try:
        resp = store.claim(cid, "a", "alice")  # rev2：整组 {a,b} 归 alice
    finally:
        store._after_commit = original

    # 领取响应对应本次提交 revision=2：a、b 由 alice 持有，与 rev2 历史一致
    _assert_response_matches_history(store, cid, resp, 2, "open")
    assert _by_id(resp) == {
        "a": (0, 4, "L", "alice"),
        "b": (3, 7, "L", "alice"),
        "c": (12, 15, "L", None),
    }
    assert resp["groups"] == [
        {"annotation_ids": ["a", "b"], "holder": "alice"},
        {"annotation_ids": ["c"], "holder": None}]

    # 交错的释放确实已串行完成：当前状态已是 revision=3、无人持有
    current = store.get(cid)
    assert current["revision"] == 3
    assert {h for _, _, _, h in _by_id(current).values()} == {None}
    # 历史完整：1 create、2 claim（alice）、3 release（alice）
    assert store.list_revisions(cid) == [
        {"revision": 1, "op": "create", "reviewer": None},
        {"revision": 2, "op": "claim", "reviewer": "alice"},
        {"revision": 3, "op": "release", "reviewer": "alice"}]
    # rev3 历史与领取响应不同：按完整 id 核对，a、b 的 holder 已回到 None
    rev3 = store.get_revision(cid, 3)
    assert _by_id(rev3)["a"] == (0, 4, "L", None)
    assert _by_id(rev3)["b"] == (3, 7, "L", None)


# ---------- 2. 连续修改：第一次修改响应不混入交错的第二次修改 ----------

def test_first_modify_response_excludes_interleaved_second_modify(store):
    cid = store.create(TEXT, [ann("a", 0, 4), ann("b", 3, 7),
                              ann("c", 12, 15)])["id"]
    store.claim(cid, "a", "alice")  # rev2：{a,b} 归 alice
    original = _one_shot_interleave(
        store,
        # rev4：把 b 改到 [4,7)「Y」（rev3 已把 a 改到 [0,2)，a、b 拆开，
        # 碎片 b 保留原持有者 alice，故同一审阅者可继续修改 b）
        lambda: store.modify(cid, "b", "alice", 4, 7, "Y"))
    try:
        resp = store.modify(cid, "a", "alice", 0, 2, "X")  # rev3
    finally:
        store._after_commit = original

    # 第一次修改响应只对应 revision=3：a 已改为 [0,2)「X」，b 仍是原样
    _assert_response_matches_history(store, cid, resp, 3, "open")
    assert _by_id(resp) == {
        "a": (0, 2, "X", "alice"),
        "b": (3, 7, "L", "alice"),
        "c": (12, 15, "L", None),
    }
    # rev3 几何：a[0,2) 与 b[3,7) 相接不连边，各成一组且均由 alice 持有
    assert resp["groups"] == [
        {"annotation_ids": ["a"], "holder": "alice"},
        {"annotation_ids": ["b"], "holder": "alice"},
        {"annotation_ids": ["c"], "holder": None}]

    # 交错的第二次修改已提交：当前为 revision=4，b 是 [4,7)「Y」
    current = store.get(cid)
    assert current["revision"] == 4
    assert _by_id(current)["b"] == (4, 7, "Y", "alice")
    assert [r["op"] for r in store.list_revisions(cid)] == \
        ["create", "claim", "modify", "modify"]
    # rev4 历史只比 rev3 多 b 的改动；按完整 id 与 rev4 核对当前状态
    rev4 = store.get_revision(cid, 4)
    assert _by_id(rev4) == _by_id(current)


# ---------- 3. 重复领取（幂等）返回判定时快照，交错释放也不污染 ----------

def test_idempotent_claim_returns_decision_snapshot_despite_release(store):
    cid = store.create(TEXT, [ann("a", 0, 4), ann("b", 3, 7)])["id"]
    store.claim(cid, "a", "alice")  # rev2：{a,b} 归 alice
    revisions_before = store.list_revisions(cid)

    original = _one_shot_interleave(
        store, lambda: store.release(cid, "a", "alice"))  # rev3：整组释放
    try:
        resp = store.claim(cid, "b", "alice")  # 本人重复领取：不产生新版本
    finally:
        store._after_commit = original

    # 判定时（rev2）快照：仍为 revision=2、整组由 alice 持有，与 rev2 历史一致
    _assert_response_matches_history(store, cid, resp, 2, "open")
    assert _by_id(resp) == {
        "a": (0, 4, "L", "alice"),
        "b": (3, 7, "L", "alice"),
    }
    assert resp["groups"] == [
        {"annotation_ids": ["a", "b"], "holder": "alice"}]

    # 幂等领取本身不增版本或历史；交错的释放提交为 rev3
    assert revisions_before == [
        {"revision": 1, "op": "create", "reviewer": None},
        {"revision": 2, "op": "claim", "reviewer": "alice"}]
    assert store.list_revisions(cid) == revisions_before + [
        {"revision": 3, "op": "release", "reviewer": "alice"}]
    current = store.get(cid)
    assert current["revision"] == 3
    assert {h for _, _, _, h in _by_id(current).values()} == {None}


# ---------- 交错窗口内的失败请求不改副本或历史 ----------

def test_failure_inside_interleave_window_changes_nothing(store):
    cid = store.create(TEXT, [ann("a", 0, 4), ann("b", 3, 7)])["id"]
    captured = {}

    def conflicting_claim():
        # bob 领取 alice 刚持有的组 → 409；失败不得写入任何内容
        try:
            store.claim(cid, "a", "bob")
        except GroupHeldError as exc:
            captured["error"] = exc

    original = _one_shot_interleave(store, conflicting_claim)
    try:
        resp = store.claim(cid, "a", "alice")  # rev2 正常提交
    finally:
        store._after_commit = original

    assert isinstance(captured.get("error"), GroupHeldError)
    # 成功响应仍是本次提交的 rev2 快照
    _assert_response_matches_history(store, cid, resp, 2, "open")
    # 失败请求不改副本或历史：当前停留在 rev2，历史只有 create/claim
    current = store.get(cid)
    assert current["revision"] == 2
    assert _by_id(current) == {
        "a": (0, 4, "L", "alice"),
        "b": (3, 7, "L", "alice"),
    }
    assert [r["op"] for r in store.list_revisions(cid)] == \
        ["create", "claim"]


# ---------- 多线程压力：每个响应都必须能在历史中找到完全一致的该版快照 ----------

def test_threaded_responses_always_match_their_committed_revision(store):
    anns = [ann(f"a{i}", 3 * i, 3 * i + 2, "L") for i in range(8)]
    cid = store.create(TEXT, anns)["id"]
    # 8 个互不重叠的批注各自领取，再由持有者并行修改两次后释放
    for i in range(8):
        store.claim(cid, f"a{i}", f"r{i}")
    base_rev = 1 + 8

    responses: list[dict] = []
    barrier = threading.Barrier(16)

    def edit(i, round_):
        barrier.wait()
        # 两轮都保持正长度区间（i=7 时 3i=21，不能把 end 收到 21）：
        # round0 缩短为 [3i, 3i+1)；round1 恢复为 [3i, 3i+2)，各改 label
        end = 3 * i + 1 if round_ == 0 else 3 * i + 2
        snap = store.modify(cid, f"a{i}", f"r{i}", 3 * i, end,
                            f"lab{i}-{round_}")
        responses.append(snap)

    threads = []
    for i in range(8):
        for round_ in (0, 1):
            threads.append(threading.Thread(target=edit, args=(i, round_)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 16 次修改全部成功；每个响应 revision 互不重复，且响应内容与该版历史一致
    assert store.get(cid)["revision"] == base_rev + 16
    revs = sorted(s["revision"] for s in responses)
    assert revs == list(range(base_rev + 1, base_rev + 17))
    for resp in responses:
        hist = store.get_revision(cid, resp["revision"])
        assert resp["status"] == hist["status"] == "open"
        assert _by_id(resp) == _by_id(hist)
        assert resp["groups"] == hist["groups"]

    # 最后由各持有者并行释放，响应同样不得混入后续释放
    released: list[dict] = []
    barrier2 = threading.Barrier(8)

    def free(i):
        barrier2.wait()
        released.append(store.release(cid, f"a{i}", f"r{i}"))

    threads = [threading.Thread(target=free, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.get(cid)["revision"] == base_rev + 24
    rel_revs = sorted(s["revision"] for s in released)
    assert rel_revs == list(range(base_rev + 17, base_rev + 25))
    for resp in released:
        hist = store.get_revision(cid, resp["revision"])
        assert _by_id(resp) == _by_id(hist)
        assert resp["groups"] == hist["groups"]
    # 终态全部无人持有
    assert {h for _, _, _, h in _by_id(store.get(cid)).values()} == {None}
