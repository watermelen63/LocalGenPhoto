# /// script
# requires-python = ">=3.10"
# dependencies = ["PySide6"]
# ///
"""
吳式 AI 圖片生成 — PySide6 桌面版
====================================
由原本的 HTML / CSS / JavaScript 介面改寫而成，功能對應如下：

  * 左側邊欄：新增創作（單次生成 / 互動式生成）、歷史紀錄（重新命名、刪除）、收合
  * 互動式生成圖片：聊天畫面，呼叫本地端 Ollama（qwen3:4b），AI 回覆以 Markdown 顯示
  * 單次圖片生成：提示詞、負面提示詞、風格、比例、解析度、數量、參考圖片、預覽區
  * 紀錄儲存：原本的 localStorage 改為同資料夾下的 chat_sessions.json

執行前準備：
    ollama pull qwen3:4b      # 並確認 Ollama 已在背景執行

執行（uv 會依照檔案開頭的 script 區塊自動安裝 PySide6）：
    uv run image_studio_app.py
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from PySide6.QtCore import (
    QEasingCurve,
    QObject,
    QPropertyAnimation,
    QRect,
    QSize,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QFont, QPixmap, QTextDocument
from PySide6.QtWidgets import (
    QApplication,
    QBoxLayout,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


# ==================== 基本設定 ====================

# 本地端 Ollama API 與模型
OLLAMA_ENDPOINT = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "qwen3:4b"

# 本地 AI 的系統指示
SYSTEM_PROMPT = "你是一個親切且專業的 AI 助手，請使用繁體中文回答。"

# 取代 localStorage：紀錄存放在程式旁邊的 JSON 檔
DATA_FILE = Path(__file__).with_name("chat_sessions.json")

SIDEBAR_WIDTH = 260
CONTENT_MAX_WIDTH = 850
NARROW_BREAKPOINT = 1050  # 對應原本 @media (max-width: 1050px)

SESSION_TITLES = {
    "single": "單次生成圖片",
    "interactive": "互動式生成圖片",
}


# ==================== 全域顏色（對應原本的 CSS Variables） ====================

COLORS = {
    "@bg-dark": "#faf6f0",      # 主背景：清爽暖米白
    "@bg-sidebar": "#dbcab5",   # 側邊欄背景：加深燕麥色
    "@bg-card": "#ffffff",      # 卡片與輸入框背景：純白
    "@primary": "#5c341a",      # 主要亮點/按鈕：深核桃棕
    "@text-main": "#1f140c",    # 主要文字：極深咖黑
    "@text-muted": "#5e4d40",   # 次要文字：中深褐色
    "@border": "#b59f88",       # 邊框顏色：暖灰褐色
    "@hover": "#eee2d4",        # 懸停背景
}

STYLE_SHEET = """
QWidget { color: @text-main; font-size: 14px; }
QMainWindow, #central, #chatView, #singleCreateView { background: @bg-dark; }

QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget#qt_scrollarea_viewport { background: transparent; }
#messagesContent, #workspaceContent, #historyContent { background: transparent; }

/* ---------- 1. 側邊欄 ---------- */
#sidebar { background: @bg-sidebar; border-right: 1px solid @border; }

#btnNewChat {
    background: @bg-card; border: 1px solid @border; border-radius: 8px;
    padding: 12px; text-align: left; font-weight: 500;
}
#btnNewChat:hover { background: @hover; }

#creationOption { background: @bg-card; border: 1px solid @border; border-radius: 8px; }
#creationOption:hover { background: @hover; }
#optionIcon { color: @primary; font-size: 20px; }
#optionTitle { font-size: 14px; font-weight: bold; }
#optionDesc { font-size: 12px; color: @text-muted; }

#historyItem { border-radius: 6px; }
#historyItem[highlight="true"] { background: @bg-card; }
#historyTitle { color: @text-muted; }
#historyItem[highlight="true"] #historyTitle { color: @text-main; }
#historyAction { background: transparent; border: none; color: @text-muted; padding: 4px; }
#historyAction:hover { color: @primary; }

/* ---------- 2. 聊天畫面 ---------- */
#chatHeader { border-bottom: 1px solid @border; }
#chatTitle { font-size: 17px; font-weight: bold; }
#menuToggle { background: transparent; border: none; font-size: 19px; padding: 4px; }

#avatarUser { background: #ca9ee6; color: #11111b; border-radius: 18px; font-weight: bold; }
#avatarAi { background: @primary; color: #11111b; border-radius: 18px; font-weight: bold; }
#messageBody { background: transparent; font-size: 15px; }

#welcomeTitle { font-size: 22px; font-weight: bold; color: @text-muted; }
#welcomeDesc { color: @text-muted; }

/* ---------- 3. 輸入框 ---------- */
#inputBox { background: @bg-card; border: 1px solid @border; border-radius: 12px; }
#chatInput { background: transparent; border: none; font-size: 16px; }
#btnSend { background: @primary; color: #11111b; border: none; border-radius: 8px; font-size: 15px; }
#btnSend:disabled { background: rgba(92, 52, 26, 0.4); }

/* ---------- 4. 單次圖片生成工作區 ---------- */
#creationHeader { border-bottom: 1px solid @border; }
#eyebrow { color: @text-muted; font-size: 10px; font-weight: 700; }
#creationTitle { font-size: 19px; font-weight: bold; }
#btnBack {
    background: @bg-card; border: 1px solid @border; border-radius: 8px; padding: 9px 14px;
}
#btnBack:hover { background: @hover; }

#panel { background: @bg-card; border: 1px solid @border; border-radius: 14px; }
#panelTitle { font-size: 20px; font-weight: bold; }
#panelDesc { color: @text-muted; font-size: 13px; }
#fieldLabel { font-size: 14px; font-weight: 600; }
#fieldHint { color: @text-muted; font-size: 11px; }

QPlainTextEdit#field, QComboBox#field {
    background: @bg-dark; border: 1px solid @border; border-radius: 8px;
    padding: 8px 10px; font-size: 14px;
}
QPlainTextEdit#field:focus, QComboBox#field:focus { border-color: @primary; }
QComboBox#field QAbstractItemView {
    background: @bg-card; selection-background-color: @hover; selection-color: @text-main;
}
#divider { background: @border; }

QRadioButton#ratio {
    background: @bg-dark; border: 1px solid @border; border-radius: 8px;
    padding: 10px 6px; color: @text-muted; font-size: 13px;
}
QRadioButton#ratio:checked {
    background: @hover; border-color: @primary; color: @primary; font-weight: 700;
}

#dropzone { background: @bg-dark; border: 1px dashed @border; border-radius: 10px; }
#dropzone:hover { background: @hover; }
#dropzoneIcon { color: @primary; font-size: 24px; }
#dropzoneTitle { color: @text-muted; font-size: 13px; font-weight: 600; }
#dropzoneDesc { color: @text-muted; font-size: 11px; }

#refPreview { background: @bg-dark; border: 1px solid @border; border-radius: 8px; }
#refName { font-size: 12px; }
#btnRemoveRef { color: #aa3333; background: transparent; border: none; padding: 3px 8px; }

#btnGenerate {
    background: @primary; border: none; border-radius: 9px; padding: 14px;
    color: white; font-size: 15px; font-weight: 700;
}
#btnGenerate:hover { background: #6f4428; }
#statusText { color: @text-muted; font-size: 12px; }

#previewState {
    background: @bg-dark; border-radius: 12px; padding: 6px 10px;
    color: @text-muted; font-size: 11px;
}
#previewCanvas {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ffffff, stop:1 #faf6f0);
    border: 1px dashed @border; border-radius: 12px;
}
#previewIcon { background: @hover; border-radius: 20px; color: @primary; font-size: 30px; }
#emptyTitle { font-size: 17px; font-weight: bold; }
#emptyDesc { color: @text-muted; font-size: 13px; }
#previewFooter { color: @text-muted; font-size: 12px; }
"""


def build_style_sheet() -> str:
    qss = STYLE_SHEET
    for token, value in COLORS.items():
        qss = qss.replace(token, value)
    return qss


# ==================== 小工具 ====================

def escape_html(text: str) -> str:
    """安全工具：避免使用者輸入被當成 HTML 解析（對應原本的 escapeHtml）"""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#039;")
    )


def markdown_to_html(markdown_text: str) -> str:
    """將 AI 回覆的 Markdown 轉為 HTML（取代 Marked.js）"""
    doc = QTextDocument()
    doc.setMarkdown(markdown_text)
    return doc.toHtml()


def repolish(widget: QWidget) -> None:
    """屬性改變後重新套用樣式表"""
    for w in [widget, *widget.findChildren(QWidget)]:
        w.style().unpolish(w)
        w.style().polish(w)


def clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item.widget() is not None:
            item.widget().deleteLater()
        elif item.layout() is not None:
            clear_layout(item.layout())


def make_label(text: str, object_name: str, wrap: bool = False) -> QLabel:
    label = QLabel(text)
    label.setObjectName(object_name)
    label.setWordWrap(wrap)
    return label


class ElidedLabel(QLabel):
    """文字過長時顯示省略號（對應 CSS text-overflow: ellipsis）"""

    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self._full_text = text
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setToolTip(text)
        self._refresh()

    def set_full_text(self, text: str) -> None:
        self._full_text = text
        self.setToolTip(text)
        self._refresh()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        elided = self.fontMetrics().elidedText(
            self._full_text, Qt.ElideRight, max(self.width(), 0)
        )
        super().setText(elided)


class ClickableCard(QPushButton):
    """內含多個文字標籤的按鈕（創作選項、上傳區）"""

    def __init__(self, object_name: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName(object_name)
        self.setCursor(Qt.PointingHandCursor)

    def add_child(self, widget: QWidget) -> QWidget:
        # 讓滑鼠事件穿透標籤，點擊任何位置都算點按鈕
        widget.setAttribute(Qt.WA_TransparentForMouseEvents)
        return widget


# ==================== 狀態管理（取代 state + localStorage） ====================

class SessionStore:
    def __init__(self, path: Path):
        self.path = path
        self.sessions: list[dict] = self._load()
        self.current_id: str | None = None

    def _load(self) -> list[dict]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return []

    def save(self) -> None:
        """將所有創作與對話紀錄儲存到 JSON 檔"""
        self.path.write_text(
            json.dumps(self.sessions, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def current(self) -> dict | None:
        return next((s for s in self.sessions if s["id"] == self.current_id), None)

    def migrate(self) -> None:
        """舊版聊天紀錄統一轉為互動式創作"""
        for session in self.sessions:
            session["mode"] = "single" if session.get("mode") == "single" else "interactive"
            session.setdefault("messages", [])
            session.setdefault("titleEdited", False)

    def create(self, mode: str = "interactive") -> dict:
        """建立新的創作或對話紀錄"""
        new_id = str(int(time.time() * 1000))
        while any(s["id"] == new_id for s in self.sessions):
            new_id = str(int(new_id) + 1)

        session = {
            "id": new_id,
            "mode": mode,                                   # 記錄屬於哪一種模式
            "title": SESSION_TITLES.get(mode, "互動式生成圖片"),
            "titleEdited": False,                           # 是否已手動修改名稱
            "messages": [],                                 # 對話紀錄
        }
        self.sessions.insert(0, session)
        self.current_id = session["id"]
        self.save()
        return session

    def delete(self, session_id: str) -> None:
        self.sessions = [s for s in self.sessions if s["id"] != session_id]
        if self.sessions and self.current_id == session_id:
            self.current_id = self.sessions[0]["id"]


# ==================== 呼叫本地端 AI（Ollama） ====================

def call_local_ai(messages: list[dict]) -> str:
    api_messages = []

    # 加入 System Prompt
    if SYSTEM_PROMPT:
        api_messages.append({"role": "system", "content": SYSTEM_PROMPT})

    # 將對話紀錄轉換成 Ollama 格式
    api_messages += [
        {"role": "user" if m["role"] == "user" else "assistant", "content": m["content"]}
        for m in messages
    ]

    body = json.dumps(
        {"model": MODEL_NAME, "messages": api_messages, "stream": False}
    ).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_ENDPOINT,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    # 本地連線不走系統代理
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    try:
        with opener.open(request, timeout=300) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        try:
            message = json.loads(err.read().decode("utf-8")).get("error")
        except Exception:
            message = None
        raise RuntimeError(message or "本地 AI 請求失敗") from err
    except urllib.error.URLError as err:
        raise RuntimeError(
            f"無法連線到本地 AI（{err.reason}），請確認 Ollama 是否已啟動。"
        ) from err

    # 取得 AI 回傳的文字
    return data["message"]["content"]


class AIBridge(QObject):
    """背景執行緒把結果送回主執行緒（避免介面卡住）"""

    succeeded = Signal(object, str)  # (session, 回覆內容)
    failed = Signal(object, str)     # (session, 錯誤訊息)

    def request(self, session: dict) -> None:
        messages = [dict(m) for m in session["messages"]]

        def worker():
            try:
                self.succeeded.emit(session, call_local_ai(messages))
            except Exception as err:  # noqa: BLE001
                self.failed.emit(session, str(err))

        threading.Thread(target=worker, daemon=True).start()


# ==================== 1. 側邊欄 ====================

class HistoryItem(QFrame):
    clicked = Signal()
    rename_requested = Signal()
    delete_requested = Signal()

    def __init__(self, session: dict, active: bool, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("historyItem")
        self.setCursor(Qt.PointingHandCursor)
        self._active = active
        self.setProperty("highlight", active)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 6, 8)
        layout.setSpacing(6)

        # 根據模式決定圖示
        icon = {"single": "🖼", "interactive": "💬"}.get(session.get("mode"), "🗨")
        self.title = ElidedLabel(f"{icon}  {session['title']}")
        self.title.setObjectName("historyTitle")
        layout.addWidget(self.title, 1)

        self.btn_rename = QToolButton(text="✎", toolTip="重新命名")
        self.btn_delete = QToolButton(text="🗑", toolTip="刪除紀錄")
        for btn in (self.btn_rename, self.btn_delete):
            btn.setObjectName("historyAction")
            btn.setCursor(Qt.PointingHandCursor)
            btn.hide()  # 預設隱藏，滑鼠移入才顯示
            layout.addWidget(btn)

        self.btn_rename.clicked.connect(lambda: self.rename_requested.emit())
        self.btn_delete.clicked.connect(lambda: self.delete_requested.emit())

    def _set_hover(self, hovered: bool) -> None:
        self.btn_rename.setVisible(hovered)
        self.btn_delete.setVisible(hovered)
        self.setProperty("highlight", hovered or self._active)
        repolish(self)

    def enterEvent(self, event):
        self._set_hover(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._set_hover(False)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class Sidebar(QFrame):
    single_create_clicked = Signal()
    interactive_create_clicked = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setMinimumWidth(0)
        self.setMaximumWidth(SIDEBAR_WIDTH)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ---- 頂部：新增創作 ----
        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(16, 16, 16, 6)
        header_layout.setSpacing(10)

        self.btn_new = QPushButton("＋  新增創作")
        self.btn_new.setObjectName("btnNewChat")
        self.btn_new.setCursor(Qt.PointingHandCursor)
        header_layout.addWidget(self.btn_new)

        # 創作模式選單（預設隱藏）
        self.creation_menu = QWidget()
        menu_layout = QVBoxLayout(self.creation_menu)
        menu_layout.setContentsMargins(0, 0, 0, 0)
        menu_layout.setSpacing(8)
        btn_single = self._make_option("🖼", "單次生成", "獨立生成一張圖片")
        btn_interactive = self._make_option("💬", "互動式生成圖片", "透過對話逐步修改圖片")
        menu_layout.addWidget(btn_single)
        menu_layout.addWidget(btn_interactive)
        self.creation_menu.hide()
        header_layout.addWidget(self.creation_menu)
        layout.addWidget(header)

        # ---- 歷史紀錄清單 ----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        history_content = QWidget()
        history_content.setObjectName("historyContent")
        self.history_layout = QVBoxLayout(history_content)
        self.history_layout.setContentsMargins(16, 10, 16, 10)
        self.history_layout.setSpacing(4)
        scroll.setWidget(history_content)
        layout.addWidget(scroll, 1)

        self.btn_new.clicked.connect(
            lambda: self.creation_menu.setVisible(not self.creation_menu.isVisible())
        )
        btn_single.clicked.connect(lambda: self.single_create_clicked.emit())
        btn_interactive.clicked.connect(lambda: self.interactive_create_clicked.emit())

    def sizeHint(self) -> QSize:
        return QSize(SIDEBAR_WIDTH, super().sizeHint().height())

    def _make_option(self, icon: str, title: str, desc: str) -> ClickableCard:
        card = ClickableCard("creationOption")
        card.setMinimumHeight(62)
        row = QHBoxLayout(card)
        row.setContentsMargins(12, 10, 12, 10)
        row.setSpacing(12)
        row.addWidget(card.add_child(make_label(icon, "optionIcon")))

        texts = QVBoxLayout()
        texts.setSpacing(2)
        texts.addWidget(card.add_child(make_label(title, "optionTitle")))
        texts.addWidget(card.add_child(make_label(desc, "optionDesc")))
        row.addLayout(texts, 1)
        return card

    def hide_menu(self) -> None:
        self.creation_menu.hide()


# ==================== 2. 互動式生成圖片（聊天畫面） ====================

class ChatInput(QTextEdit):
    """多行輸入框：Enter 發送、Shift + Enter 換行，高度自動調整"""

    send_requested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("chatInput")
        self.setAcceptRichText(False)
        self.setPlaceholderText("輸入訊息... (Shift + Enter 換行)")
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.document().documentLayout().documentSizeChanged.connect(self._adjust_height)
        self._adjust_height()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not (
            event.modifiers() & Qt.ShiftModifier
        ):
            self.send_requested.emit()
            return
        super().keyPressEvent(event)

    def _adjust_height(self, *_):
        height = int(self.document().size().height()) + 4
        self.setFixedHeight(min(max(height, 30), 200))


class MessageRow(QWidget):
    """單筆訊息：頭像 + 內容"""

    def __init__(self, role: str, html_text: str, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        avatar = QLabel("U" if role == "user" else "AI")
        avatar.setObjectName("avatarUser" if role == "user" else "avatarAi")
        avatar.setFixedSize(36, 36)
        avatar.setAlignment(Qt.AlignCenter)
        layout.addWidget(avatar, 0, Qt.AlignTop)

        body = QLabel(html_text)
        body.setObjectName("messageBody")
        body.setTextFormat(Qt.RichText)
        body.setWordWrap(True)
        body.setOpenExternalLinks(True)
        body.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse
        )
        body.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(body, 1)


class ChatView(QWidget):
    toggle_sidebar_clicked = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("chatView")
        self._stick_to_bottom = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ---- 頂部標題列 ----
        header = QFrame()
        header.setObjectName("chatHeader")
        header.setFixedHeight(56)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 0, 16, 0)
        header_layout.setSpacing(12)
        btn_toggle = QPushButton("☰")
        btn_toggle.setObjectName("menuToggle")
        btn_toggle.setCursor(Qt.PointingHandCursor)
        btn_toggle.clicked.connect(lambda: self.toggle_sidebar_clicked.emit())
        header_layout.addWidget(btn_toggle)
        header_layout.addWidget(make_label("吳式互動式生成圖片", "chatTitle"), 1)
        layout.addWidget(header)

        # ---- 訊息顯示容器 ----
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        content.setObjectName("messagesContent")
        content_row = QHBoxLayout(content)
        content_row.setContentsMargins(0, 0, 0, 0)

        column = QWidget()
        column.setMaximumWidth(CONTENT_MAX_WIDTH)
        self.messages_layout = QVBoxLayout(column)
        self.messages_layout.setContentsMargins(24, 24, 24, 24)
        self.messages_layout.setSpacing(20)

        # 兩側留白，中間最寬 850px 並置中
        content_row.addStretch(1)
        content_row.addWidget(column, 1000)
        content_row.addStretch(1)
        self.scroll.setWidget(content)
        layout.addWidget(self.scroll, 1)

        scrollbar = self.scroll.verticalScrollBar()
        scrollbar.rangeChanged.connect(self._on_range_changed)

        # ---- 訊息輸入區塊 ----
        input_row = QHBoxLayout()
        input_row.setContentsMargins(0, 0, 0, 0)
        input_container = QWidget()
        input_container.setMaximumWidth(CONTENT_MAX_WIDTH)
        container_layout = QVBoxLayout(input_container)
        container_layout.setContentsMargins(24, 16, 24, 24)

        input_box = QFrame()
        input_box.setObjectName("inputBox")
        box_layout = QHBoxLayout(input_box)
        box_layout.setContentsMargins(16, 10, 12, 10)
        box_layout.setSpacing(10)

        self.input = ChatInput()
        self.btn_send = QPushButton("➤")
        self.btn_send.setObjectName("btnSend")
        self.btn_send.setFixedSize(32, 32)
        self.btn_send.setCursor(Qt.PointingHandCursor)
        box_layout.addWidget(self.input, 1, Qt.AlignBottom)
        box_layout.addWidget(self.btn_send, 0, Qt.AlignBottom)
        container_layout.addWidget(input_box)

        input_row.addStretch(1)
        input_row.addWidget(input_container, 1000)
        input_row.addStretch(1)
        layout.addLayout(input_row)

    # ---------- 訊息渲染 ----------

    def render_messages(self, session: dict | None) -> None:
        clear_layout(self.messages_layout)

        # 空對話：顯示歡迎提示
        if not session or not session["messages"]:
            self.messages_layout.addStretch(1)
            title = make_label("👋 讓我們一起把想像變成畫面", "welcomeTitle", wrap=True)
            desc = make_label(
                "告訴我你想呈現的內容，我們可以逐步完善構圖、風格與細節", "welcomeDesc", wrap=True
            )
            for label in (title, desc):
                label.setAlignment(Qt.AlignCenter)
                self.messages_layout.addWidget(label)
            self.messages_layout.addStretch(4)
            return

        for msg in session["messages"]:
            self.append_message(msg["role"], msg["content"])
        self.messages_layout.addStretch(1)
        self.scroll_to_bottom()

    def append_message(self, role: str, content: str) -> MessageRow:
        # AI 回覆以 Markdown 顯示；使用者輸入先轉義避免被當成 HTML
        html_text = (
            markdown_to_html(content)
            if role == "ai"
            else escape_html(content).replace("\n", "<br>")
        )
        row = MessageRow(role, html_text)
        self._insert_row(row)
        return row

    def show_loading(self) -> MessageRow:
        row = MessageRow("ai", "⏳ 思考中...")
        self._insert_row(row)
        self.scroll_to_bottom()
        return row

    def _insert_row(self, row: QWidget) -> None:
        # 插在最後的 stretch 之前
        count = self.messages_layout.count()
        last = self.messages_layout.itemAt(count - 1) if count else None
        if last is not None and last.spacerItem() is not None:
            self.messages_layout.insertWidget(count - 1, row)
        else:
            self.messages_layout.addWidget(row)

    # ---------- 自動捲動到最底部 ----------

    def scroll_to_bottom(self) -> None:
        self._stick_to_bottom = True
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())
        QTimer.singleShot(150, self._release_bottom)

    def _release_bottom(self) -> None:
        self._stick_to_bottom = False

    def _on_range_changed(self, _min: int, maximum: int) -> None:
        if self._stick_to_bottom:
            self.scroll.verticalScrollBar().setValue(maximum)


# ==================== 3. 單次圖片生成工作區 ====================

class SingleCreateView(QWidget):
    toggle_sidebar_clicked = Signal()
    back_clicked = Signal()

    STYLES = [
        ("auto", "不指定"),
        ("realistic", "寫實攝影"),
        ("illustration", "插畫"),
        ("anime", "動漫風格"),
        ("3d", "3D 立體風格"),
        ("watercolor", "水彩畫"),
        ("pixel", "像素藝術"),
    ]
    RATIOS = ["1:1", "3:4", "4:3", "16:9"]

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("singleCreateView")
        self.reference_image_path: str | None = None
        self._narrow: bool | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_header())

        # ---- 左右主要工作區 ----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        workspace = QWidget()
        workspace.setObjectName("workspaceContent")
        self.workspace_layout = QBoxLayout(QBoxLayout.LeftToRight, workspace)
        self.workspace_layout.setContentsMargins(20, 20, 20, 20)
        self.workspace_layout.setSpacing(20)

        self.settings_panel = self._build_settings_panel()
        self.preview_panel = self._build_preview_panel()
        self.workspace_layout.addWidget(self.settings_panel, 0)
        self.workspace_layout.addWidget(self.preview_panel, 1)

        scroll.setWidget(workspace)
        layout.addWidget(scroll, 1)

    # ---------- 頂部標題列 ----------

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("creationHeader")
        header.setMinimumHeight(68)
        row = QHBoxLayout(header)
        row.setContentsMargins(24, 10, 24, 10)
        row.setSpacing(14)

        btn_toggle = QPushButton("☰", toolTip="切換側邊欄")
        btn_toggle.setObjectName("menuToggle")
        btn_toggle.setCursor(Qt.PointingHandCursor)
        btn_toggle.clicked.connect(lambda: self.toggle_sidebar_clicked.emit())
        row.addWidget(btn_toggle)

        titles = QVBoxLayout()
        titles.setSpacing(3)
        titles.addWidget(make_label("IMAGE STUDIO", "eyebrow"))
        titles.addWidget(make_label("吳式單次圖片生成", "creationTitle"))
        row.addLayout(titles, 1)

        btn_back = QPushButton("←  返回聊天")
        btn_back.setObjectName("btnBack")
        btn_back.setCursor(Qt.PointingHandCursor)
        btn_back.clicked.connect(lambda: self.back_clicked.emit())
        row.addWidget(btn_back)
        return header

    # ---------- 左側：生成設定 ----------

    def _field(self, layout: QVBoxLayout, label_text: str, widget: QWidget,
               hint: str | None = None) -> None:
        group = QVBoxLayout()
        group.setSpacing(8)
        if label_text:
            group.addWidget(make_label(label_text, "fieldLabel"))
        group.addWidget(widget)
        if hint:
            group.addWidget(make_label(hint, "fieldHint", wrap=True))
        layout.addLayout(group)
        layout.addSpacing(12)

    def _build_settings_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("panel")
        panel.setMinimumWidth(300)
        panel.setMaximumWidth(370)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(0)

        # 標題說明
        layout.addWidget(make_label("CONFIGURATION", "eyebrow"))
        layout.addSpacing(6)
        layout.addWidget(make_label("生成設定", "panelTitle"))
        layout.addSpacing(7)
        layout.addWidget(make_label("調整你想要的圖片內容與輸出條件。", "panelDesc", wrap=True))
        layout.addSpacing(24)

        # 提示詞
        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setObjectName("field")
        self.prompt_edit.setPlaceholderText(
            "例如：一座位於森林深處的玻璃建築，陽光穿過樹葉，電影級光影效果..."
        )
        self.prompt_edit.setFixedHeight(130)
        self._field(layout, "圖片提示詞", self.prompt_edit,
                    "描述主體、場景、光線、構圖與畫面風格。")

        # 負面提示詞
        self.negative_edit = QPlainTextEdit()
        self.negative_edit.setObjectName("field")
        self.negative_edit.setPlaceholderText("例如：模糊、低解析度、文字錯誤...")
        self.negative_edit.setFixedHeight(70)
        self._field(layout, "負面提示詞（選填）", self.negative_edit)

        # 分隔線
        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFixedHeight(1)
        layout.addSpacing(10)
        layout.addWidget(divider)
        layout.addSpacing(22)

        # 圖片風格
        self.style_combo = QComboBox()
        self.style_combo.setObjectName("field")
        for value, text in self.STYLES:
            self.style_combo.addItem(text, value)
        self._field(layout, "圖片風格", self.style_combo)

        # 圖片比例
        ratio_widget = QWidget()
        ratio_grid = QGridLayout(ratio_widget)
        ratio_grid.setContentsMargins(0, 0, 0, 0)
        ratio_grid.setSpacing(8)
        self.ratio_group = QButtonGroup(self)
        for i, ratio in enumerate(self.RATIOS):
            radio = QRadioButton(ratio)
            radio.setObjectName("ratio")
            radio.setCursor(Qt.PointingHandCursor)
            radio.setChecked(i == 0)
            self.ratio_group.addButton(radio)
            ratio_grid.addWidget(radio, 0, i)
        self._field(layout, "圖片比例", ratio_widget)

        # 解析度、生成數量
        self.resolution_combo = QComboBox()
        self.resolution_combo.setObjectName("field")
        for res in (512, 768, 1024):
            self.resolution_combo.addItem(str(res), res)
        self.resolution_combo.setCurrentIndex(1)  # 預設 768

        self.count_combo = QComboBox()
        self.count_combo.setObjectName("field")
        for n in range(1, 5):
            self.count_combo.addItem(f"{n} 張", n)

        grid = QHBoxLayout()
        grid.setSpacing(12)
        for label_text, combo in (("解析度", self.resolution_combo),
                                  ("生成數量", self.count_combo)):
            col = QVBoxLayout()
            col.setSpacing(8)
            col.addWidget(make_label(label_text, "fieldLabel"))
            col.addWidget(combo)
            grid.addLayout(col, 1)
        layout.addLayout(grid)
        layout.addSpacing(20)

        # 參考圖片
        layout.addWidget(make_label("參考圖片（選填）", "fieldLabel"))
        layout.addSpacing(8)
        layout.addWidget(self._build_dropzone())
        layout.addWidget(self._build_reference_preview())
        layout.addSpacing(20)

        # 生成按鈕
        btn_generate = QPushButton("✨  生成圖片")
        btn_generate.setObjectName("btnGenerate")
        btn_generate.setCursor(Qt.PointingHandCursor)
        btn_generate.clicked.connect(self.handle_generate)
        layout.addWidget(btn_generate)

        # 狀態提示
        self.status_label = make_label("目前尚未連接圖片生成模型。", "statusText", wrap=True)
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addSpacing(12)
        layout.addWidget(self.status_label)
        layout.addStretch(1)
        return panel

    def _build_dropzone(self) -> ClickableCard:
        zone = ClickableCard("dropzone")
        zone.setMinimumHeight(120)
        col = QVBoxLayout(zone)
        col.setContentsMargins(16, 16, 16, 16)
        col.setSpacing(7)
        col.addStretch(1)
        for text, name in (("☁", "dropzoneIcon"),
                           ("點擊上傳圖片", "dropzoneTitle"),
                           ("支援常見圖片格式", "dropzoneDesc")):
            label = zone.add_child(make_label(text, name))
            label.setAlignment(Qt.AlignCenter)
            col.addWidget(label)
        col.addStretch(1)
        zone.clicked.connect(self.choose_reference_image)
        return zone

    def _build_reference_preview(self) -> QFrame:
        self.ref_wrap = QFrame()
        self.ref_wrap.setObjectName("refPreview")
        row = QHBoxLayout(self.ref_wrap)
        row.setContentsMargins(10, 10, 10, 10)
        row.setSpacing(12)

        self.ref_thumb = QLabel()
        self.ref_thumb.setFixedSize(58, 58)
        row.addWidget(self.ref_thumb)

        info = QVBoxLayout()
        info.setSpacing(6)
        self.ref_name = ElidedLabel()
        self.ref_name.setObjectName("refName")
        btn_remove = QPushButton("移除")
        btn_remove.setObjectName("btnRemoveRef")
        btn_remove.setCursor(Qt.PointingHandCursor)
        btn_remove.clicked.connect(self.remove_reference_image)
        info.addWidget(self.ref_name)
        info.addWidget(btn_remove, 0, Qt.AlignLeft)
        row.addLayout(info, 1)

        self.ref_wrap.hide()
        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 10, 0, 0)
        wrapper_layout.addWidget(self.ref_wrap)
        return wrapper

    # ---------- 右側：圖片預覽 ----------

    def _build_preview_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(0)

        heading = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(6)
        titles.addWidget(make_label("OUTPUT", "eyebrow"))
        titles.addWidget(make_label("圖片預覽", "panelTitle"))
        heading.addLayout(titles, 1)
        self.preview_state = make_label("尚未生成", "previewState")
        heading.addWidget(self.preview_state, 0, Qt.AlignVCenter)
        layout.addLayout(heading)
        layout.addSpacing(18)

        # 預覽畫布
        self.preview_canvas = QFrame()
        self.preview_canvas.setObjectName("previewCanvas")
        self.preview_canvas.setMinimumHeight(420)
        canvas_layout = QVBoxLayout(self.preview_canvas)
        canvas_layout.setContentsMargins(24, 24, 24, 24)
        canvas_layout.addStretch(1)

        empty = QWidget()
        empty.setMaximumWidth(350)
        empty_layout = QVBoxLayout(empty)
        empty_layout.setContentsMargins(0, 0, 0, 0)
        empty_layout.setSpacing(9)
        icon = make_label("🖼", "previewIcon")
        icon.setFixedSize(76, 76)
        icon.setAlignment(Qt.AlignCenter)
        empty_layout.addWidget(icon, 0, Qt.AlignHCenter)
        empty_layout.addSpacing(9)
        for text, name in (("圖片將顯示於此", "emptyTitle"),
                           ("請先在左側輸入提示詞，並調整圖片生成設定。", "emptyDesc")):
            label = make_label(text, name, wrap=True)
            label.setAlignment(Qt.AlignCenter)
            empty_layout.addWidget(label)

        canvas_layout.addWidget(empty, 0, Qt.AlignHCenter)
        canvas_layout.addStretch(1)
        layout.addWidget(self.preview_canvas, 1)

        # 底部說明
        layout.addSpacing(16)
        layout.addWidget(make_label("ⓘ  目前為介面原型，尚未連接圖片生成模型。",
                                    "previewFooter", wrap=True))
        return panel

    # ---------- 參考圖片 ----------

    def choose_reference_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇參考圖片", "",
            "圖片檔案 (*.png *.jpg *.jpeg *.bmp *.gif *.webp);;所有檔案 (*)",
        )
        if not path:
            return

        pixmap = QPixmap(path)
        if pixmap.isNull():  # 確認是否為圖片
            QMessageBox.warning(self, "格式錯誤", "請選擇圖片檔案！")
            return

        # 裁成 58x58 正方形縮圖（對應 object-fit: cover）
        scaled = pixmap.scaled(58, 58, Qt.KeepAspectRatioByExpanding,
                               Qt.SmoothTransformation)
        x = (scaled.width() - 58) // 2
        y = (scaled.height() - 58) // 2
        self.ref_thumb.setPixmap(scaled.copy(QRect(x, y, 58, 58)))

        self.reference_image_path = path
        self.ref_name.set_full_text(Path(path).name)
        self.ref_wrap.show()

    def remove_reference_image(self) -> None:
        self.reference_image_path = None
        self.ref_thumb.clear()
        self.ref_name.set_full_text("")
        self.ref_wrap.hide()

    # ---------- 生成 ----------

    def collect_settings(self) -> dict:
        """整理目前所有設定，之後串接圖片生成後端時可直接使用"""
        checked = self.ratio_group.checkedButton()
        return {
            "prompt": self.prompt_edit.toPlainText().strip(),
            "negative_prompt": self.negative_edit.toPlainText().strip(),
            "style": self.style_combo.currentData(),
            "style_label": self.style_combo.currentText(),
            "aspect_ratio": checked.text() if checked else "1:1",
            "resolution": self.resolution_combo.currentData(),
            "count": self.count_combo.currentData(),
            "reference_image": self.reference_image_path,
        }

    def handle_generate(self) -> None:
        settings = self.collect_settings()

        if not settings["prompt"]:
            self.status_label.setText("請先輸入圖片提示詞。")
            self.prompt_edit.setFocus()
            return

        self.status_label.setText(
            f"提示詞已讀取。風格：{settings['style_label']}；"
            f"比例：{settings['aspect_ratio']}；"
            f"解析度：{settings['resolution']}；"
            f"數量：{settings['count']} 張。"
            " 尚未連接圖片生成模型，尚未實際生成圖片。"
        )
        self.preview_state.setText("等待模型連線")

        # 目前先保留負面提示詞資料供後續模型串接使用
        print("負面提示詞：", settings["negative_prompt"])

    # ---------- 響應式：視窗變窄時改為上下排列 ----------

    def resizeEvent(self, event):
        super().resizeEvent(event)
        narrow = self.window().width() <= NARROW_BREAKPOINT
        if narrow == self._narrow:
            return
        self._narrow = narrow
        if narrow:
            self.workspace_layout.setDirection(QBoxLayout.TopToBottom)
            self.settings_panel.setMaximumWidth(16777215)
            self.preview_canvas.setMinimumHeight(350)
        else:
            self.workspace_layout.setDirection(QBoxLayout.LeftToRight)
            self.settings_panel.setMaximumWidth(370)
            self.preview_canvas.setMinimumHeight(420)


# ==================== 主視窗 ====================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Web Chat Assistant")
        self.resize(1280, 800)
        self.setMinimumSize(820, 560)

        self.store = SessionStore(DATA_FILE)
        self.ai = AIBridge(self)
        self.ai.succeeded.connect(self._on_ai_success)
        self.ai.failed.connect(self._on_ai_failed)
        self._waiting = False
        self._sidebar_collapsed = False
        self._sidebar_anim: QPropertyAnimation | None = None

        central = QWidget()
        central.setObjectName("central")
        row = QHBoxLayout(central)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        self.sidebar = Sidebar()
        self.chat_view = ChatView()
        self.single_view = SingleCreateView()
        self.stack = QStackedWidget()
        self.stack.addWidget(self.chat_view)
        self.stack.addWidget(self.single_view)

        row.addWidget(self.sidebar)
        row.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        # ---- 事件綁定 ----
        self.sidebar.single_create_clicked.connect(self._new_single)
        self.sidebar.interactive_create_clicked.connect(self._new_interactive)
        self.chat_view.toggle_sidebar_clicked.connect(self.toggle_sidebar)
        self.single_view.toggle_sidebar_clicked.connect(self.toggle_sidebar)
        self.single_view.back_clicked.connect(self.show_chat_view)
        self.chat_view.input.send_requested.connect(self.handle_send_message)
        self.chat_view.btn_send.clicked.connect(lambda: self.handle_send_message())

        self.init_app()

    # ---------- 初始化 ----------

    def init_app(self) -> None:
        self.store.migrate()

        if not self.store.sessions:
            # 沒有任何歷史紀錄時，自動建立互動式生成圖片
            self.store.create("interactive")
        elif self.store.current() is None:
            self.store.current_id = self.store.sessions[0]["id"]

        self.render_sidebar()
        self.render_messages()
        self.show_session_view(self.store.current())

    # ---------- 畫面切換 ----------

    def show_single_create_view(self) -> None:
        self.stack.setCurrentWidget(self.single_view)
        self.sidebar.hide_menu()

    def show_chat_view(self) -> None:
        self.stack.setCurrentWidget(self.chat_view)
        self.render_messages()

    def show_session_view(self, session: dict | None) -> None:
        """根據歷史紀錄類型切換畫面"""
        if session and session.get("mode") == "single":
            self.show_single_create_view()
        else:
            self.show_chat_view()

    def toggle_sidebar(self) -> None:
        """側邊欄收合 / 展開（含 0.3 秒動畫）"""
        self._sidebar_collapsed = not self._sidebar_collapsed
        anim = QPropertyAnimation(self.sidebar, b"maximumWidth", self)
        anim.setDuration(300)
        anim.setStartValue(self.sidebar.width())
        anim.setEndValue(0 if self._sidebar_collapsed else SIDEBAR_WIDTH)
        anim.setEasingCurve(QEasingCurve.InOutQuad)
        anim.start()
        self._sidebar_anim = anim

    # ---------- 新增創作 ----------

    def _new_single(self) -> None:
        self.store.create("single")
        self.render_sidebar()
        self.render_messages()
        self.show_single_create_view()

    def _new_interactive(self) -> None:
        self.store.create("interactive")
        self.render_sidebar()
        self.sidebar.hide_menu()
        self.show_chat_view()

    # ---------- 側邊欄歷史紀錄 ----------

    def render_sidebar(self) -> None:
        layout = self.sidebar.history_layout
        clear_layout(layout)

        for session in self.store.sessions:
            item = HistoryItem(session, session["id"] == self.store.current_id)
            item.clicked.connect(lambda s=session: self._select_session(s))
            item.rename_requested.connect(lambda s=session: self._rename_session(s))
            item.delete_requested.connect(lambda s=session: self._delete_session(s["id"]))
            layout.addWidget(item)
        layout.addStretch(1)

    def _select_session(self, session: dict) -> None:
        self.store.current_id = session["id"]
        self.render_sidebar()
        self.show_session_view(session)

    def _rename_session(self, session: dict) -> None:
        new_title, ok = QInputDialog.getText(
            self, "重新命名", "請輸入新的創作名稱：", text=session["title"]
        )
        if not ok:
            return
        new_title = new_title.strip()
        if not new_title:
            QMessageBox.warning(self, "提示", "名稱不能空白！")
            return
        session["title"] = new_title
        session["titleEdited"] = True
        self.store.save()
        self.render_sidebar()

    def _delete_session(self, session_id: str) -> None:
        self.store.delete(session_id)

        # 已經沒有紀錄：自動建立互動式生成圖片
        if not self.store.sessions:
            self.store.create("interactive")

        self.store.save()
        self.render_sidebar()
        self.show_session_view(self.store.current())

    # ---------- 對話 ----------

    def render_messages(self) -> None:
        self.chat_view.render_messages(self.store.current())

    def handle_send_message(self) -> None:
        if self._waiting:
            return

        chat_input = self.chat_view.input
        text = chat_input.toPlainText().strip()
        if not text:
            return

        session = self.store.current()
        if session is None:
            return

        # 追加使用者訊息
        session["messages"].append({"role": "user", "content": text})

        # 若為首條訊息，自動設為對話標題（條件與原版相同）
        if (
            len(session["messages"]) == 1
            and not session.get("titleEdited")
            and session.get("mode") in (None, "chat")
        ):
            session["title"] = text[:15] + ("..." if len(text) > 15 else "")

        # 重置輸入框
        chat_input.clear()
        self._set_waiting(True)

        self.render_sidebar()
        self.render_messages()

        # 插入「思考中」佔位
        self.chat_view.show_loading()

        # 背景發送 API 請求
        self.ai.request(session)

    def _set_waiting(self, waiting: bool) -> None:
        self._waiting = waiting
        self.chat_view.btn_send.setDisabled(waiting)

    def _on_ai_success(self, session: dict, reply: str) -> None:
        session["messages"].append({"role": "ai", "content": reply})
        self.store.save()
        self.render_messages()
        self._set_waiting(False)

    def _on_ai_failed(self, session: dict, error: str) -> None:
        QMessageBox.critical(self, "錯誤", f"請求失敗: {error}")
        if session["messages"]:
            session["messages"].pop()  # 請求失敗時復原最後一筆發送
        self.render_messages()
        self._set_waiting(False)


# ==================== 啟動主流程 ====================

def main() -> None:
    app = QApplication(sys.argv)

    font = QFont()
    font.setFamilies(
        ["Microsoft JhengHei UI", "PingFang TC", "Noto Sans CJK TC", "Segoe UI", "Arial"]
    )
    app.setFont(font)
    app.setStyleSheet(build_style_sheet())

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()