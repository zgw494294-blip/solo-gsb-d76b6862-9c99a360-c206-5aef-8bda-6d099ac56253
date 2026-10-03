"""工作副本变更响应的版本语义回归：成功响应必须对应本次提交的完整快照。

回归背景：变更响应曾在锁外重新读取副本——若另一请求在本次提交之后、
响应生成之前又提交了新版本（如领取提交 rev2 后同一审阅者释放并提交
rev3），响应会混入后续版本的 revision / status / annotations / groups。

这里用**可控交错**（本次事务结束后、响应生成前，在另一连接上插入另一
请求）覆盖：领取后释放、连续修改、重复领取后释放、释放后领取；并按
**完整 id** 核对响应与该版历史快照的批注内容（响应按几何排序、历史
按该版保存的创建原序，二者只能按 id 对齐比较）。
"""

import contextlib

import pytest

from app.copies import CopyStore

TEXT = "x" * 30


def ann(id, start, end, label="L"):
    return {"id": id, "start": start, "end": end, "label": label}


@pytest.fixture()
def db_path(tmp_path):
    return tmp_path / "copies.db"


@pytest.fixture()
def store(db_path):
    return CopyStore(db_path)


@pytest.fixture()
def other(db_path):
    """同一数据库文件上的第二个仓库：模拟另一并发请求（另一连接）。"""
    return CopyStore(db_path)


@contextlib.contextmanager
def interleave_before_response(store, fn):
    """可控交错：store 下一次变更的事务结束后、响应生成前执行一次 fn。

    包装 store._connect 返回的连接：写路径在 commit 后触发、幂等路径在
    连接 close 后触发（此刻本次提交已落库、响应快照尚未生成），模拟另一
    请求的提交抢在响应之前到达。fn 须用另一仓库实例执行，避免与 store
    的进程锁纠缠。产出单元素列表 fired 供断言交错确已发生。
    """
    real_connect = store._connect
    fired = []

    class HookConn:
        def __init__(self, real):
            self._real = real

        def _fire(self):
            if not fired:
                fired.append(True)
                fn()

        def commit(self):
            self._real.commit()
            self._fire()

        def close(self):
            self._real.close()
            self._fire()

        def __getattr__(self, name):
            return getattr(self._real, name)

    store._connect = lambda: HookConn(real_connect())
    try:
        yield fired
    finally:
        store._connect = real_connect


def _anns_by_id(snap):
    """按完整字符串 id 对齐的批注内容（响应按几何排序、历史按创建原序）。"""
    return {a["id"]: (a["start"], a["end"], a["label"], a["holder"])
            for a in snap["annotations"]}


def _assert_response_is_revision(store, cid, resp, revision):
    """响应必须是该版本的完整快照：按完整 id 核对批注内容及其余字段。"""
    rev = store.get_revision(cid, revision)
    assert resp["revision"] == revision == rev["revision"]
    assert resp["status"] == rev["status"]
    assert resp["text"] == rev["text"]
    assert _anns_by_id(resp) == _anns_by_id(rev)  # 按完整 id 对齐批注内容
    assert resp["groups"] == rev["groups"]


# ---------- 领取后释放 ----------

def test_claim_response_is_own_commit_when_release_lands_first(store, other):
    # 创建顺序 b → a 与几何顺序 a → b 不同：响应与历史只能按完整 id 对齐
    cid = store.create(TEXT, [ann("b", 3, 7), ann("a", 0, 4)])["id"]
    with interleave_before_response(
            store, lambda: other.release(cid, "a", "alice")) as fired:
        resp = store.claim(cid, "a", "alice")  # 本次提交 rev2；随后 rev3 抢先生效
    assert fired
    # 响应是本次提交（rev2：整组 {a, b} 归 alice）的完整快照，
    # 不得混入 rev3（无人持有）的 revision / annotations / groups
    assert resp["revision"] == 2
    assert resp["status"] == "open"
    assert _anns_by_id(resp) == {"a": (0, 4, "L", "alice"),
                                 "b": (3, 7, "L", "alice")}
    assert resp["groups"] == [{"annotation_ids": ["a", "b"], "holder": "alice"}]
    _assert_response_is_revision(store, cid, resp, 2)
    # 交错的释放确已生效：当前副本为 rev3、无人持有
    current = store.get(cid)
    assert current["revision"] == 3
    assert _anns_by_id(current) == {"a": (0, 4, "L", None),
                                    "b": (3, 7, "L", None)}
    assert [r["op"] for r in store.list_revisions(cid)] == \
        ["create", "claim", "release"]


# ---------- 连续修改 ----------

def test_modify_response_is_own_commit_when_next_modify_lands_first(store, other):
    cid = store.create(TEXT, [ann("b", 5, 7), ann("a", 0, 2)])["id"]
    store.claim(cid, "a", "alice")  # rev2
    store.claim(cid, "b", "alice")  # rev3
    with interleave_before_response(
            store, lambda: other.modify(cid, "b", "alice", 5, 6, "乙改")) as fired:
        resp = store.modify(cid, "a", "alice", 0, 1, "甲改")  # 本次提交 rev4
    assert fired
    # 响应是 rev4 的完整快照：a 已改、b 尚未改——不得混入 rev5 对 b 的修改
    assert resp["revision"] == 4
    assert _anns_by_id(resp) == {"a": (0, 1, "甲改", "alice"),
                                 "b": (5, 7, "L", "alice")}
    _assert_response_is_revision(store, cid, resp, 4)
    # 交错的连续修改确已生效：当前副本 rev5，b 已改
    current = store.get(cid)
    assert current["revision"] == 5
    assert _anns_by_id(current) == {"a": (0, 1, "甲改", "alice"),
                                    "b": (5, 6, "乙改", "alice")}
    assert [r["op"] for r in store.list_revisions(cid)] == \
        ["create", "claim", "claim", "modify", "modify"]


# ---------- 重复领取后释放 ----------

def test_repeated_claim_response_is_decision_snapshot_then_release(store, other):
    cid = store.create(TEXT, [ann("b", 3, 7), ann("a", 0, 4)])["id"]
    store.claim(cid, "a", "alice")  # rev2：整组 {a, b} 归 alice
    with interleave_before_response(
            store, lambda: other.release(cid, "b", "alice")) as fired:
        resp = store.claim(cid, "a", "alice")  # 幂等：不增版本；随后 rev3 抢先生效
    assert fired
    # 本人重复领取返回判定时的快照（rev2：本人持有整组），不增加版本或历史，
    # 不得混入 rev3（无人持有）
    assert resp["revision"] == 2
    assert _anns_by_id(resp) == {"a": (0, 4, "L", "alice"),
                                 "b": (3, 7, "L", "alice")}
    _assert_response_is_revision(store, cid, resp, 2)
    # 交错的释放确已生效：当前副本 rev3、无人持有
    current = store.get(cid)
    assert current["revision"] == 3
    assert _anns_by_id(current) == {"a": (0, 4, "L", None),
                                    "b": (3, 7, "L", None)}
    # 幂等领取不产生版本与历史：只有 create / claim / release 三条记录
    assert store.list_revisions(cid) == [
        {"revision": 1, "op": "create", "reviewer": None},
        {"revision": 2, "op": "claim", "reviewer": "alice"},
        {"revision": 3, "op": "release", "reviewer": "alice"}]


# ---------- 释放后领取 ----------

def test_release_response_is_own_commit_when_claim_lands_first(store, other):
    cid = store.create(TEXT, [ann("b", 3, 7), ann("a", 0, 4)])["id"]
    store.claim(cid, "a", "alice")  # rev2
    with interleave_before_response(
            store, lambda: other.claim(cid, "a", "bob")) as fired:
        resp = store.release(cid, "b", "alice")  # 本次提交 rev3；随后 rev4 抢先生效
    assert fired
    # 响应是本次提交（rev3：整组释放、无人持有）的完整快照，
    # 不得混入 rev4（bob 持有）
    assert resp["revision"] == 3
    assert _anns_by_id(resp) == {"a": (0, 4, "L", None),
                                 "b": (3, 7, "L", None)}
    assert resp["groups"] == [{"annotation_ids": ["a", "b"], "holder": None}]
    _assert_response_is_revision(store, cid, resp, 3)
    # 交错的领取确已生效：当前副本 rev4，整组归 bob
    current = store.get(cid)
    assert current["revision"] == 4
    assert _anns_by_id(current) == {"a": (0, 4, "L", "bob"),
                                    "b": (3, 7, "L", "bob")}
    assert [r["op"] for r in store.list_revisions(cid)] == \
        ["create", "claim", "release", "claim"]
