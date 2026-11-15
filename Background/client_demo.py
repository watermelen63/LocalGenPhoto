"""
PySide6 前端範例：呼叫本地端生圖後端（server.py）

先啟動後端：uvicorn server:app --port 8000
再執行：python client_demo.py

重點：用 QNetworkAccessManager 非同步送請求，生圖時介面不會卡住；
      用 QTimer 每 0.3 秒查 /progress 更新進度條。
你的 APP 可以把 ImageGenClient 這個類別直接搬進去用。
"""

import json
import sys

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PySide6.QtWidgets import (
    QApplication, QComboBox, QHBoxLayout, QLabel, QLineEdit, QProgressBar,
    QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

BACKEND = "http://127.0.0.1:8000"


class ImageGenClient(QObject):
    """封裝和後端的溝通，用 Signal 把結果丟回介面。"""

    models_loaded = Signal(list)          # [{"id":..., "label":...}, ...]
    progress = Signal(int, int)           # step, total
    finished = Signal(QPixmap, dict)      # 圖片, 資訊（seed、耗時…）
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.net = QNetworkAccessManager(self)
        self.timer = QTimer(self)
        self.timer.setInterval(300)
        self.timer.timeout.connect(self._poll_progress)

    # ---- 取得模型清單 ----
    def fetch_models(self):
        reply = self.net.get(QNetworkRequest(QUrl(f"{BACKEND}/models")))
        reply.finished.connect(lambda: self._on_models(reply))

    def _on_models(self, reply: QNetworkReply):
        if reply.error() != QNetworkReply.NetworkError.NoError:
            self.failed.emit(f"連不到後端：{reply.errorString()}")
        else:
            data = json.loads(bytes(reply.readAll()).decode())
            self.models_loaded.emit(data["models"])
        reply.deleteLater()

    # ---- 生圖 ----
    def generate(self, prompt: str, model: str, width=1024, height=1024, seed=-1):
        req = QNetworkRequest(QUrl(f"{BACKEND}/generate"))
        req.setHeader(QNetworkRequest.KnownHeaders.ContentTypeHeader, "application/json")
        req.setTransferTimeout(30 * 60 * 1000)  # 第一次要下載模型，給長一點
        body = json.dumps({
            "prompt": prompt, "model": model,
            "width": width, "height": height, "seed": seed,
        }).encode()
        reply = self.net.post(req, body)
        reply.finished.connect(lambda: self._on_generated(reply))
        self.timer.start()

    def _on_generated(self, reply: QNetworkReply):
        self.timer.stop()
        data = bytes(reply.readAll())
        if reply.error() != QNetworkReply.NetworkError.NoError:
            try:
                msg = json.loads(data.decode())["detail"]
            except Exception:
                msg = reply.errorString()
            self.failed.emit(msg)
        else:
            pix = QPixmap()
            pix.loadFromData(data, "PNG")
            info = {
                "seed": bytes(reply.rawHeader(b"X-Seed")).decode(),
                "elapsed": bytes(reply.rawHeader(b"X-Elapsed")).decode(),
                "model": bytes(reply.rawHeader(b"X-Model")).decode(),
                "png": data,  # 原始 PNG，要存檔可以直接寫出去
            }
            self.finished.emit(pix, info)
        reply.deleteLater()

    # ---- 進度 ----
    def _poll_progress(self):
        reply = self.net.get(QNetworkRequest(QUrl(f"{BACKEND}/progress")))
        reply.finished.connect(lambda: self._on_progress(reply))

    def _on_progress(self, reply: QNetworkReply):
        if reply.error() == QNetworkReply.NetworkError.NoError:
            p = json.loads(bytes(reply.readAll()).decode())
            self.progress.emit(p.get("step", 0), p.get("total", 0))
        reply.deleteLater()


class DemoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("本地生圖 Demo")
        self.resize(720, 820)

        self.prompt = QLineEdit(placeholderText="輸入提示詞，例如：一隻在台中街頭的橘貓")
        self.model = QComboBox()
        self.seed = QSpinBox(minimum=-1, maximum=2**31 - 1, value=-1)
        self.seed.setPrefix("seed ")
        self.btn = QPushButton("生成")
        self.bar = QProgressBar()
        self.status = QLabel("連線中…")
        self.image = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(512, 512)
        self.image.setStyleSheet("background:#222; border-radius:8px;")

        row = QHBoxLayout()
        row.addWidget(self.model, 2)
        row.addWidget(self.seed, 1)
        row.addWidget(self.btn, 1)

        lay = QVBoxLayout(self)
        lay.addWidget(self.prompt)
        lay.addLayout(row)
        lay.addWidget(self.bar)
        lay.addWidget(self.status)
        lay.addWidget(self.image, 1)

        self.client = ImageGenClient(self)
        self.client.models_loaded.connect(self.on_models)
        self.client.progress.connect(self.on_progress)
        self.client.finished.connect(self.on_finished)
        self.client.failed.connect(self.on_failed)
        self.btn.clicked.connect(self.on_click)
        self.prompt.returnPressed.connect(self.on_click)

        self.client.fetch_models()

    def on_models(self, models):
        for m in models:
            self.model.addItem(m["label"], m["id"])
        self.status.setText("後端已連線")

    def on_click(self):
        text = self.prompt.text().strip()
        if not text:
            return
        self.btn.setEnabled(False)
        self.bar.setValue(0)
        self.status.setText("生成中…（第一次使用某個模型會先下載，請耐心等）")
        self.client.generate(text, self.model.currentData(), seed=self.seed.value())

    def on_progress(self, step, total):
        if total:
            self.bar.setMaximum(total)
            self.bar.setValue(step)

    def on_finished(self, pix: QPixmap, info: dict):
        self.btn.setEnabled(True)
        self.bar.setValue(self.bar.maximum())
        self.image.setPixmap(pix.scaled(self.image.size(),
                                        Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation))
        self.status.setText(f"完成：{info['model']}，seed={info['seed']}，耗時 {info['elapsed']} 秒")
        with open("last_output.png", "wb") as f:
            f.write(info["png"])

    def on_failed(self, msg):
        self.btn.setEnabled(True)
        self.status.setText(f"錯誤：{msg}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = DemoWindow()
    w.show()
    sys.exit(app.exec())
