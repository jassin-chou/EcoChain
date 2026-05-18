"""
firebase_db.py — EcoChain Firebase Firestore 資料庫層
======================================================
取代原本的 models/sheets_db.py，API 介面完全相容。

Firestore 集合設計：
  accounts/{slot_id}
    └── player_name   : str
        password_hash : str    (SHA-256 hex, salted with slot_id)
        created_at    : str    (ISO timestamp)

  saves/{slot_id}
    └── save_json     : dict   (完整存檔 JSON)
        updated_at    : str    (ISO timestamp)

  leaderboard/{slot_id}
    └── player_name   : str
        eco_score     : float
        coins         : float
        species_count : int
        updated_at    : str

  analytics/{auto_id}
    └── player_id     : str
        cells_json    : list
        eco_score     : float
        coins         : float
        timestamp     : str

環境變數（擇一設定）：
  FIREBASE_CREDS_JSON   — 完整 serviceAccountKey.json 內容（Render 用）
  FIREBASE_CREDS_PATH   — 本地 serviceAccountKey.json 路徑（本地開發用）
  FIREBASE_PROJECT_ID   — 若使用 Application Default Credentials 時需要
"""

import os
import json
import hashlib
import datetime
from typing import Optional

# ── Firebase Admin SDK ──────────────────────────────────────────
import firebase_admin
from firebase_admin import credentials, firestore


class FirestoreDB:
    def __init__(self):
        self._db: Optional[firestore.Client] = None
        self._ready = False

    def connect(self):
        """啟動時呼叫一次，初始化 Firebase Admin SDK。"""
        if self._ready:
            return

        try:
            creds_json = os.getenv("FIREBASE_CREDS_JSON", "")
            creds_path = os.getenv("FIREBASE_CREDS_PATH", "serviceAccountKey.json")

            if creds_json:
                # Render / 正式環境：從環境變數讀取整個 JSON
                cred_dict = json.loads(creds_json)
                cred = credentials.Certificate(cred_dict)
            elif os.path.exists(creds_path):
                # 本地開發：從檔案讀取
                cred = credentials.Certificate(creds_path)
            else:
                raise FileNotFoundError(
                    "找不到 Firebase 憑證。請設定 FIREBASE_CREDS_JSON 或 FIREBASE_CREDS_PATH"
                )

            # 避免重複初始化（uvicorn reload 時）
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred)

            self._db = firestore.client()
            self._ready = True
            print("[Firebase] 連線成功 ✅")

        except Exception as e:
            print(f"[Firebase] 連線失敗：{e}")
            self._ready = False

    @property
    def db(self) -> firestore.Client:
        if not self._ready or self._db is None:
            raise RuntimeError("Firebase 尚未初始化，請先呼叫 connect()")
        return self._db

    def _now(self) -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    # ── password hashing ──────────────────────────────────────────

    @staticmethod
    def _hash_password(slot_id: str, password: str) -> str:
        """
        SHA-256 with slot_id as salt.
        slot_id acts as a per-account salt, so rainbow tables don't work
        across accounts even if passwords are reused.
        Format: SHA256(slot_id + ":" + password)
        """
        raw = f"{slot_id}:{password}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ── accounts ──────────────────────────────────────────────────

    def create_account(
        self,
        slot_id: str,
        player_name: str,
        password: str,
    ) -> tuple[bool, str]:
        """
        建立新帳號。
        回傳 (success, error_message)。
        - 若 slot_id 已存在 → False, "already_exists"
        - 密碼為空 → False, "empty_password"
        """
        if not password:
            return False, "empty_password"
        try:
            ref = self.db.collection("accounts").document(slot_id)
            existing = ref.get()
            if existing.exists:
                return False, "already_exists"
            pw_hash = self._hash_password(slot_id, password)
            ref.set({
                "player_name": player_name,
                "password_hash": pw_hash,
                "created_at": self._now(),
            })
            return True, ""
        except Exception as e:
            print(f"[Firebase] create_account 失敗：{e}")
            return False, "db_error"

    def verify_password(self, slot_id: str, password: str) -> tuple[bool, str]:
        """
        驗證密碼。
        回傳 (success, player_name)。
        - 找不到帳號 → False, ""
        - 密碼錯誤   → False, ""
        - 成功       → True, player_name
        """
        try:
            doc = self.db.collection("accounts").document(slot_id).get()
            if not doc.exists:
                return False, ""
            data = doc.to_dict()
            expected = data.get("password_hash", "")
            candidate = self._hash_password(slot_id, password)
            # constant-time comparison
            import hmac as _hmac
            if not _hmac.compare_digest(expected, candidate):
                return False, ""
            return True, data.get("player_name", "")
        except Exception as e:
            print(f"[Firebase] verify_password 失敗：{e}")
            return False, ""

    def account_exists(self, slot_id: str) -> bool:
        """檢查 slot_id 是否已有帳號記錄。"""
        try:
            return self.db.collection("accounts").document(slot_id).get().exists
        except Exception:
            return False

    def delete_account(self, slot_id: str) -> bool:
        """
        刪除帳號、存檔、排行榜記錄（呼叫前須已驗密碼）。
        Firestore 不支援跨集合 batch delete，逐一刪除。
        """
        try:
            batch = self.db.batch()
            for col in ("accounts", "saves", "leaderboard"):
                ref = self.db.collection(col).document(slot_id)
                batch.delete(ref)
            batch.commit()
            return True
        except Exception as e:
            print(f"[Firebase] delete_account 失敗：{e}")
            return False

    # ── saves ─────────────────────────────────────────────────────

    def save_game(self, slot_id: str, save_data: dict) -> bool:
        """
        寫入或覆蓋存檔。
        Firestore 文件路徑：saves/{slot_id}
        """
        try:
            self.db.collection("saves").document(slot_id).set({
                "save_json": save_data,
                "updated_at": self._now(),
            })
            return True
        except Exception as e:
            print(f"[Firebase] save_game 失敗：{e}")
            return False

    def load_game(self, slot_id: str) -> Optional[dict]:
        """
        讀取存檔，回傳 save_data dict 或 None。
        """
        try:
            doc = self.db.collection("saves").document(slot_id).get()
            if doc.exists:
                data = doc.to_dict()
                return data.get("save_json")
            return None
        except Exception as e:
            print(f"[Firebase] load_game 失敗：{e}")
            return None

    # ── leaderboard ───────────────────────────────────────────────

    def upsert_leaderboard(
        self,
        slot_id: str,
        player_name: str,
        eco_score: float,
        coins: float,
        species_count: int,
    ) -> bool:
        """
        更新排行榜（upsert）。
        只在新分數 ≥ 舊分數時才覆蓋，避免存一個爛存檔把好成績蓋掉。
        """
        try:
            ref = self.db.collection("leaderboard").document(slot_id)
            existing = ref.get()

            if existing.exists:
                old_score = existing.to_dict().get("eco_score", 0)
                if eco_score < old_score:
                    return True  # 舊分數更高，不覆蓋

            ref.set({
                "player_name": player_name,
                "eco_score": eco_score,
                "coins": coins,
                "species_count": species_count,
                "updated_at": self._now(),
            })
            return True
        except Exception as e:
            print(f"[Firebase] upsert_leaderboard 失敗：{e}")
            return False

    def list_accounts(self) -> list[dict]:
        """
        列出所有帳號（只回傳 slot_id + player_name + created_at）。
        不含密碼 hash。供讀檔 modal 顯示用。
        """
        try:
            docs = self.db.collection("accounts").stream()
            return [
                {
                    "slot_id": d.id,
                    "player_name": d.to_dict().get("player_name", ""),
                    "created_at": d.to_dict().get("created_at", ""),
                }
                for d in docs
            ]
        except Exception as e:
            print(f"[Firebase] list_accounts 失敗：{e}")
            return []

    def find_slot_by_name(self, player_name: str) -> Optional[str]:
        """
        用 player_name 查詢對應的 slot_id。
        找到回傳 slot_id，找不到回傳 None。
        注意：player_name 不保證唯一，回傳第一筆。
        """
        try:
            docs = (
                self.db.collection("accounts")
                .where("player_name", "==", player_name)
                .limit(1)
                .stream()
            )
            for doc in docs:
                return doc.id
            return None
        except Exception as e:
            print(f"[Firebase] find_slot_by_name 失敗：{e}")
            return None

    def get_leaderboard(self, limit: int = 20) -> list[dict]:
        """
        取排行榜前 N 名（依 eco_score 降冪）。
        """
        try:
            docs = (
                self.db.collection("leaderboard")
                .order_by("eco_score", direction=firestore.Query.DESCENDING)
                .limit(limit)
                .stream()
            )
            return [{"slot_id": d.id, **d.to_dict()} for d in docs]
        except Exception as e:
            print(f"[Firebase] get_leaderboard 失敗：{e}")
            return []

    # ── analytics ─────────────────────────────────────────────────

    def log_analytics(
        self,
        player_id: str,
        cells_json: list,
        eco_score: float,
        coins: float,
    ) -> bool:
        """
        寫入一筆分析記錄（auto-ID 文件）。
        """
        try:
            self.db.collection("analytics").add({
                "player_id": player_id,
                "cells_json": cells_json,
                "eco_score": eco_score,
                "coins": coins,
                "timestamp": self._now(),
            })
            return True
        except Exception as e:
            print(f"[Firebase] log_analytics 失敗：{e}")
            return False


# 單例，供 main.py import
db = FirestoreDB()
