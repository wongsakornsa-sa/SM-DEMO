"""Button-driven GUI for the eight-view object demo. Run: python gui.py"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from demo import (
    TARGET_NAMES, VIEW_COUNT, load_model, new_tracker, process_frame,
    require_packages, write_csv,
)
from camera_sources import BACKENDS, SOURCE_TYPES, CameraSource, probe_devices
from gui_layout import build_ui, COLORS
from font_setup import register_font
from panorama import make_grid, make_overview, make_panorama


class DemoGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Object 360 Studio")
        # Loading Torch/Ultralytics at startup keeps the window hidden for
        # several seconds. Import them only when the experiment begins.
        self.cv2 = self.YOLO = self.DeepSort = None
        self.cap = None
        self.connected_description = ""
        self.latest_frame = None
        self.model = None
        self.class_ids = set()
        self.tracker_online = None
        self.frames = []
        self.run_dir = None
        self.first_capture = None
        self.capture_elapsed = 0.0
        self.online_processing_history = 0.0
        self.retakes = 0
        self.online = {"detections": [], "tracks": [], "seconds": []}
        self.batch = None
        self.latest_summary = None
        self.selected_capture_view = None
        self.selected_review_view = 0
        self.events = queue.Queue()
        self.busy = False
        self.loaded_weights = None
        self.closed = False
        self.output_root = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent

        build_ui(self)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(40, self.update_preview)
        self.root.after(100, self.poll_events)

    def log(self, text: str):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def show_frame(self, label: tk.Label, frame, size=None):
        if size is None:
            size = (max(100, label.winfo_width() - 12), max(100, label.winfo_height() - 12))
        h, w = frame.shape[:2]
        scale = min(1.0, size[0] / w, size[1] / h)
        if scale < 1:
            frame = self.cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=self.cv2.INTER_AREA)
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail(size)
        photo = ImageTk.PhotoImage(img)
        label.configure(image=photo, text="")
        label.image = photo

    def refresh_progress(self):
        completed = len(self.frames)
        self.progress.set(completed / VIEW_COUNT)
        self.progress_var.set(f"{completed} / {VIEW_COUNT} มุม")
        for index, label in enumerate(self.step_labels):
            if index < completed:
                label.configure(text=f"✓   มุมที่ {index + 1}", fg=COLORS["green"], bg=COLORS["panel"])
            elif index == completed and completed < VIEW_COUNT:
                label.configure(text=f"●   มุมที่ {index + 1}", fg=COLORS["text"], bg=COLORS["raised"])
            else:
                label.configure(text=f"○   มุมที่ {index + 1}", fg=COLORS["muted"], bg=COLORS["panel"])
        self.capture_button.configure(text=f"ถ่ายภาพมุมที่ {min(completed + 1, VIEW_COUNT)}" if completed < VIEW_COUNT else "ถ่ายครบ 8 มุม")
        self.retake_button.configure(
            state="normal" if self.selected_capture_view is not None
            and self.selected_capture_view < completed and not self.busy else "disabled"
        )

    def show_thumbnail(self, index, frame):
        label = self.view_cards[index]
        self.show_frame(label, frame, (max(85, label.winfo_width() - 10), 72))
        label.configure(text=f"มุม {index + 1}", compound="top", height=95,
                        fg=COLORS["text"], bg=COLORS["raised"])

    def select_capture_view(self, index):
        if index >= len(self.frames):
            return
        self.selected_capture_view = index
        for i, tile in enumerate(self.view_cards):
            tile.configure(highlightbackground=COLORS["blue"] if i == index else COLORS["empty"])
        self.retake_button.configure(state="disabled" if self.busy else "normal")
        self.result_title_var.set(f"ผลตรวจจับมุมที่ {index + 1} · เลือกถ่ายใหม่ได้")
        image_path = self.run_dir / "online" / f"view_{index + 1:02d}.jpg"
        image = self.cv2.imread(str(image_path)) if image_path.exists() else self.frames[index]
        if image is not None:
            self.show_frame(self.result_label, image)

    def retake_selected(self):
        index = self.selected_capture_view
        if index is None or index >= len(self.frames) or self.busy or self.cap is None or self.latest_frame is None:
            return
        replacement = self.latest_frame.copy()
        self.busy = True
        self.capture_button.configure(state="disabled")
        self.retake_button.configure(state="disabled")
        self.start_button.configure(state="disabled")
        self.connect_button.configure(state="disabled")
        self.scan_button.configure(state="disabled")
        self.status_var.set(f"กำลังถ่ายใหม่และคำนวณผลตั้งแต่มุมที่ {index + 1}...")
        threading.Thread(target=self.reprocess_retake, args=(index, replacement), daemon=True).start()

    def reprocess_retake(self, index, replacement):
        try:
            frames = list(self.frames)
            frames[index] = replacement
            tracker = new_tracker(self.DeepSort)
            online = {"detections": [], "tracks": [], "seconds": []}
            annotated = []
            for view, frame in enumerate(frames, 1):
                image, detections, tracks, seconds = process_frame(
                    frame, view, "online", self.model, self.class_ids, tracker, self.cv2, 0.35
                )
                annotated.append(image)
                online["detections"].extend(detections)
                online["tracks"].extend(tracks)
                online["seconds"].append(seconds)
            self.events.put(("retake_ready", index, frames, annotated, online, tracker))
        except Exception as exc:
            self.events.put(("retake_error", str(exc)))

    def connect(self):
        if self.busy:
            return
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.latest_frame = None
            self.camera_state.set("● กล้อง: ยังไม่เชื่อมต่อ")
            self.start_button.configure(state="disabled", fg_color=COLORS["raised"])
        source = self.source_var.get().strip()
        if not source:
            messagebox.showerror("กล้อง", "ใส่หมายเลขกล้องหรือ URL สตรีม")
            return
        self.busy = True
        self.connect_button.configure(state="disabled")
        self.scan_button.configure(state="disabled")
        kind, backend = self.kind_var.get(), self.backend_var.get()
        self.status_var.set("กำลังเชื่อมต่อกล้อง...")
        threading.Thread(target=self.connect_worker, args=(kind, source, backend), daemon=True).start()

    def connect_worker(self, kind, source, backend):
        try:
            if self.cv2 is None:
                import cv2
                self.cv2 = cv2
            cap = CameraSource(self.cv2, kind, source, backend)
            if self.closed:
                cap.release()
                return
            self.events.put(("connected", cap, kind, source, backend))
        except Exception as exc:
            self.events.put(("connect_error", str(exc)))

    def finish_connect(self, cap, kind, source, backend):
        self.busy = False
        self.connect_button.configure(state="normal")
        self.scan_button.configure(state="normal")
        self.cap = cap
        self.connected_description = f"{kind}: {source} ({backend})"
        self.camera_state.set("● กล้อง: เชื่อมต่อแล้ว")
        self.start_button.configure(state="normal", fg_color=COLORS["blue"])
        self.status_var.set(f"เชื่อมต่อแล้ว: {source} | กดเริ่มรอบทดลอง")
        self.log(f"เชื่อมต่อกล้อง {source}")

    def scan(self):
        if self.busy:
            return
        if self.cv2 is None:
            try:
                import cv2
                self.cv2 = cv2
            except ImportError as exc:
                messagebox.showerror("กล้อง", str(exc))
                return
        self.status_var.set("กำลังค้นหากล้องที่คอมพิวเตอร์มองเห็น...")
        self.scan_button.configure(state="disabled")
        threading.Thread(target=self.scan_worker, daemon=True).start()

    def scan_worker(self):
        try:
            self.events.put(("scan_done", probe_devices(self.cv2)))
        except Exception as exc:
            self.events.put(("error", f"ค้นหากล้องไม่สำเร็จ: {exc}"))

    def choose_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
        if path:
            self.kind_var.set("Video file")
            self.source_var.set(path)

    def show_summary(self):
        self.tabs.set("สรุปผล")
        self.refresh_review()
        self.refresh_panorama()

    @staticmethod
    def set_readonly_text(widget, content):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.configure(state="disabled")

    def switch_summary_mode(self, selected):
        self.per_view_frame.pack_forget()
        self.panorama_frame.pack_forget()
        if selected == "ภาพรายมุม":
            self.per_view_frame.pack(fill="x")
            self.refresh_review()
        else:
            self.panorama_frame.pack(fill="x")
            self.refresh_panorama()

    def change_review_method(self, _selected):
        self.refresh_review()

    def select_review_view(self, index):
        self.selected_review_view = index
        self.refresh_review()

    def refresh_review(self):
        index = self.selected_review_view
        for i, button in enumerate(self.review_buttons):
            button.configure(fg_color=COLORS["blue"] if i == index else COLORS["raised"])
        if index >= len(self.frames) or self.run_dir is None:
            self.review_title.set(f"มุมที่ {index + 1} · ยังไม่ได้ถ่ายภาพ")
            self.review_image.configure(image="", text="ยังไม่มีภาพสำหรับมุมนี้")
            self.review_image.image = None
            self.set_readonly_text(self.review_findings, "ถ่ายภาพมุมนี้ก่อน จึงจะดูผลตรวจจับได้")
            return
        mode = "online" if self.review_method.get() == "วิธีที่ 1" else "batch"
        data = self.online if mode == "online" else self.batch
        if data is None:
            self.review_title.set(f"มุมที่ {index + 1} · รอวิธีที่ 2 ประมวลผลครบ 8 มุม")
            self.review_image.configure(image="", text="ผลวิธีที่ 2 ยังไม่พร้อม")
            self.review_image.image = None
            self.set_readonly_text(self.review_findings, "วิธีที่ 2 จะเริ่มหลังถ่ายครบทั้ง 8 มุม")
            return
        image = self.cv2.imread(str(self.run_dir / mode / f"view_{index + 1:02d}.jpg"))
        if image is not None:
            self.show_frame(self.review_image, image, (1050, 315))
        detections = [row for row in data["detections"] if row["view"] == index + 1]
        tracks = [row for row in data["tracks"] if row["view"] == index + 1]
        self.review_title.set(f"มุมที่ {index + 1} · ตรวจพบ {len(detections)} รายการ · ID {len(tracks)} รายการ")
        lines = [f"YOLO ตรวจพบ {len(detections)} รายการในภาพนี้"]
        lines.extend(f"{number}. {row['class']}  ความมั่นใจ {row['confidence']:.2f}  "
                     f"กรอบ ({row['x1']}, {row['y1']})–({row['x2']}, {row['y2']})"
                     for number, row in enumerate(detections, 1))
        if not detections:
            lines.append("ไม่พบขวด แก้ว หรือเก้าอี้เหนือเกณฑ์ความมั่นใจ 0.35")
        lines.append(f"\nDeep SORT: {len(tracks)} ID ในภาพนี้")
        lines.extend(f"• {row['class']}  ID {row['track_id']} ({row['state']})" for row in tracks)
        if not tracks:
            lines.append("ยังไม่มี ID ที่แสดงในภาพนี้")
        self.set_readonly_text(self.review_findings, "\n".join(lines))

    def refresh_panorama(self):
        summary = self.latest_summary
        if summary is None or self.run_dir is None:
            self.panorama_title.set("พาโนรามาหรือภาพรวม 8 มุม")
            self.summary_image.configure(image="", text="ภาพรวมจะแสดงหลังถ่ายครบ 8 มุม")
            self.summary_image.image = None
            self.set_readonly_text(self.panorama_findings, "ถ่ายครบ 8 มุมก่อนจึงจะเห็นภาพรวมและผลตรวจจับ")
            return
        image = self.cv2.imread(str(self.run_dir / summary["summary_image"]))
        if image is not None:
            self.show_frame(self.summary_image, image, (1080, 335))
        if summary["panorama_stitched"]:
            self.panorama_title.set("พาโนรามา · ผล YOLO บนภาพที่ต่อแล้ว")
            detections = summary["panorama_detections"]
            lines = [f"YOLO ตรวจบนภาพพาโนรามา: {len(detections)} รายการ"]
            lines.extend(f"{number}. {row['class']}  ความมั่นใจ {row['confidence']:.2f}  "
                         f"กรอบ ({row['x1']}, {row['y1']})–({row['x2']}, {row['y2']})"
                         for number, row in enumerate(detections, 1))
            if summary["panorama_detection_error"]:
                lines.append("ตรวจภาพพาโนรามาไม่สำเร็จ: " + summary["panorama_detection_error"])
            elif not detections:
                lines.append("ไม่พบขวด แก้ว หรือเก้าอี้เหนือเกณฑ์ความมั่นใจ 0.35 บนภาพที่ต่อแล้ว")
        else:
            self.panorama_title.set("ภาพรวม 8 มุม · กรอบตรวจจับจากแต่ละภาพ")
            lines = ["ต่อพาโนรามาไม่สำเร็จ จึงแสดงภาพตารางพร้อมกรอบที่ตรวจจากแต่ละมุม (วิธีที่ 1)"]
            for view in range(1, VIEW_COUNT + 1):
                counts = Counter(row["class"] for row in self.online["detections"] if row["view"] == view)
                description = ", ".join(f"{name} {count}" for name, count in sorted(counts.items())) or "ไม่พบวัตถุ"
                lines.append(f"มุม {view}: {description}")
            lines.append("ตัวเลขรวมอาจนับวัตถุชิ้นเดิมหลายครั้งเมื่อปรากฏหลายมุม")
        self.set_readonly_text(self.panorama_findings, "\n".join(lines))

    def update_preview(self):
        if self.cap is not None and self.cap.isOpened():
            ok, frame = self.cap.read()
            if ok:
                self.latest_frame = frame
                self.show_frame(self.live_label, frame)
            else:
                self.latest_frame = None
        self.root.after(40, self.update_preview)

    def start_session(self):
        if self.cap is None or self.latest_frame is None:
            messagebox.showwarning("กล้อง", "รอให้ภาพสดจากกล้องปรากฏก่อน")
            return
        if self.busy:
            return
        self.busy = True
        self.start_button.configure(state="disabled")
        self.connect_button.configure(state="disabled")
        self.scan_button.configure(state="disabled")
        weights = self.weights_var.get().strip()
        self.status_var.set("กำลังเตรียมโมเดล กรุณารอสักครู่...")
        threading.Thread(target=self.prepare_session, args=(weights,), daemon=True).start()

    def prepare_session(self, weights):
        try:
            self.cv2, self.YOLO, self.DeepSort = require_packages()
            if getattr(sys, "frozen", False) and weights == "yolo11n.pt":
                weights = str(Path(sys._MEIPASS) / weights)
            if self.model is None or self.loaded_weights != weights:
                self.model, self.class_ids = load_model(self.YOLO, weights, list(TARGET_NAMES))
                # Warm up outside measured capture time; later rounds reuse the model.
                import numpy as np
                self.model.predict(np.zeros((480, 640, 3), dtype=np.uint8),
                                   classes=sorted(self.class_ids), verbose=False)
                self.loaded_weights = weights
            self.tracker_online = new_tracker(self.DeepSort)
            self.events.put(("session_ready",))
        except (Exception, SystemExit) as exc:
            self.events.put(("session_error", str(exc)))

    def begin_session(self):
        self.busy = False
        self.model_state.set("● YOLO: พร้อม")
        self.tracker_state.set("● Deep SORT: พร้อม")
        self.run_dir = self.output_root / "runs" / datetime.now().strftime("demo_%Y%m%d_%H%M%S")
        for sub in ("raw", "online", "batch"):
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)
        self.frames = []
        self.first_capture = None
        self.capture_elapsed = 0.0
        self.online_processing_history = 0.0
        self.retakes = 0
        self.online = {"detections": [], "tracks": [], "seconds": []}
        self.batch = None
        self.latest_summary = None
        self.selected_capture_view = None
        self.selected_review_view = 0
        for index, tile in enumerate(self.view_cards):
            tile.configure(image="", text=f"{index + 1}\nรอถ่าย", compound="none", height=5,
                           fg=COLORS["muted"], bg=COLORS["empty"],
                           highlightbackground=COLORS["empty"])
            tile.image = None
        self.result_title_var.set("ผลตรวจจับล่าสุด")
        self.result_label.configure(image="", text="ผล YOLO และ Deep SORT จะแสดงที่นี่")
        self.result_label.image = None
        self.summary_image.configure(image="", text="ภาพรวมจะแสดงหลังถ่ายครบ 8 มุม")
        self.summary_image.image = None
        self.summary_status.set("กำลังทดลอง กดถ่ายภาพให้ครบ 8 มุม")
        self.online_metric_var.set("ยังไม่มีผล")
        self.batch_metric_var.set("ยังไม่มีผล")
        self.set_readonly_text(self.summary_details, "ยังไม่มีผลการทดลอง")
        self.refresh_review()
        self.refresh_panorama()
        self.refresh_progress()
        self.start_button.configure(state="disabled", fg_color=COLORS["raised"])
        self.connect_button.configure(state="disabled")
        self.scan_button.configure(state="disabled")
        self.capture_button.configure(state="normal", fg_color=COLORS["blue"])
        self.open_button.configure(state="normal")
        self.status_var.set("พร้อมถ่ายภาพมุมที่ 1 จาก 8 | หันกล้องด้วยมือแล้วกด ถ่ายภาพมุมนี้")
        self.log(f"เริ่มรอบทดลองใหม่: {self.run_dir}")

    def capture(self):
        if self.latest_frame is None or self.busy or len(self.frames) >= VIEW_COUNT:
            return
        if self.first_capture is None:
            self.first_capture = time.perf_counter()
        view = len(self.frames) + 1
        frame = self.latest_frame.copy()
        self.busy = True
        self.capture_button.configure(state="disabled", fg_color=COLORS["raised"])
        self.retake_button.configure(state="disabled")
        self.status_var.set(f"กำลังประมวลผลมุมที่ {view}...")
        threading.Thread(target=self.capture_worker, args=(frame, view), daemon=True).start()

    def capture_worker(self, frame, view):
        try:
            result = process_frame(frame, view, "online", self.model, self.class_ids,
                                   self.tracker_online, self.cv2, 0.35)
            if not self.cv2.imwrite(str(self.run_dir / "raw" / f"view_{view:02d}.jpg"), frame):
                raise OSError("บันทึกภาพต้นฉบับไม่ได้")
            if not self.cv2.imwrite(str(self.run_dir / "online" / f"view_{view:02d}.jpg"), result[0]):
                raise OSError("บันทึกภาพผลตรวจจับไม่ได้")
            self.events.put(("capture_ready", frame, view, result))
        except Exception as exc:
            self.events.put(("capture_error", str(exc)))

    def finish_capture(self, frame, view, result):
        image, detections, tracks, seconds = result
        self.frames.append(frame)
        self.busy = False
        self.online["detections"].extend(detections)
        self.online["tracks"].extend(tracks)
        self.online["seconds"].append(seconds)
        self.online_processing_history += seconds
        self.show_frame(self.result_label, image)
        self.show_thumbnail(view - 1, frame)
        self.capture_elapsed = time.perf_counter() - self.first_capture
        self.refresh_progress()
        self.select_capture_view(view - 1)
        self.select_review_view(view - 1)
        self.log(f"มุม {view}/8: ตรวจพบ {len(detections)} รายการ, ID {len(tracks)} รายการ, {seconds:.3f} วินาที")
        if view == VIEW_COUNT:
            self.busy = True
            self.retake_button.configure(state="disabled")
            self.status_var.set("ถ่ายครบ 8 มุม กำลังประมวลผลวิธีที่ 2...")
            threading.Thread(target=self.run_batch, daemon=True).start()
        else:
            self.capture_button.configure(state="normal", fg_color=COLORS["blue"])
            self.status_var.set(f"พร้อมถ่ายภาพมุมที่ {view + 1} จาก 8")

    def run_batch(self):
        try:
            tracker = new_tracker(self.DeepSort)
            batch = {"detections": [], "tracks": [], "seconds": []}
            for view, frame in enumerate(self.frames, 1):
                image, detections, tracks, seconds = process_frame(
                    frame, view, "batch", self.model, self.class_ids, tracker, self.cv2, 0.35
                )
                self.cv2.imwrite(str(self.run_dir / "batch" / f"view_{view:02d}.jpg"), image)
                batch["detections"].extend(detections)
                batch["tracks"].extend(tracks)
                batch["seconds"].append(seconds)
                self.events.put(("progress", view, seconds))
            for mode, data in (("online", self.online), ("batch", batch)):
                write_csv(self.run_dir / f"{mode}_detections.csv", data["detections"],
                          ["view", "mode", "class", "confidence", "x1", "y1", "x2", "y2"])
                write_csv(self.run_dir / f"{mode}_tracks.csv", data["tracks"],
                          ["view", "mode", "class", "track_id", "state", "x1", "y1", "x2", "y2"])
            online_seconds = self.online_processing_history
            batch_seconds = sum(batch["seconds"])
            manual_seconds = max(0.0, self.capture_elapsed - online_seconds)
            make_overview(self.cv2, self.frames, self.run_dir / "overview_8_views.jpg")
            make_grid(self.cv2, self.frames, self.run_dir / "grid_8_views.jpg")
            stitched, stitch_note = make_panorama(self.cv2, self.frames, self.run_dir / "panorama.jpg")
            panorama_detections = []
            panorama_detection_error = ""
            if stitched:
                summary_image = "panorama.jpg"
                try:
                    panorama_detections = self.detect_panorama(
                        self.run_dir / "panorama.jpg", self.run_dir / "panorama_detected.jpg"
                    )
                    summary_image = "panorama_detected.jpg"
                except Exception as exc:
                    panorama_detection_error = str(exc)
            else:
                annotated = [self.cv2.imread(str(self.run_dir / "online" / f"view_{view:02d}.jpg"))
                             for view in range(1, VIEW_COUNT + 1)]
                make_grid(self.cv2, annotated, self.run_dir / "grid_8_detected.jpg")
                summary_image = "grid_8_detected.jpg"
            summary = {
                "views": VIEW_COUNT, "classes": list(TARGET_NAMES),
                "source": self.connected_description,
                "capture_phase_seconds": round(self.capture_elapsed, 3),
                "estimated_manual_capture_seconds": round(manual_seconds, 3),
                "online_processing_seconds": round(online_seconds, 3),
                "batch_processing_seconds": round(batch_seconds, 3),
                "online_total_seconds": round(self.capture_elapsed, 3),
                "estimated_batch_total_seconds": round(manual_seconds + batch_seconds, 3),
                "timing_note": "Batch total estimates the same capture phase plus batch processing. "
                               "After retakes, online time also includes earlier attempts and is not a controlled comparison.",
                "retakes": self.retakes,
                "detections_online": len(self.online["detections"]),
                "detections_batch": len(batch["detections"]),
                "panorama_stitched": stitched,
                "panorama_note": stitch_note,
                "panorama_detections": panorama_detections,
                "panorama_detection_error": panorama_detection_error,
                "summary_image": summary_image,
            }
            (self.run_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self.events.put(("done", summary, batch))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def detect_panorama(self, source_path, output_path):
        image = self.cv2.imread(str(source_path))
        if image is None:
            raise ValueError("อ่านภาพพาโนรามาไม่ได้")
        result = self.model.predict(image, conf=0.35, classes=sorted(self.class_ids), verbose=False)[0]
        annotated = image.copy()
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = (round(float(value)) for value in box.xyxy[0].cpu().tolist())
            label = str(result.names[int(box.cls[0].item())])
            confidence = round(float(box.conf[0].item()), 4)
            self.cv2.rectangle(annotated, (x1, y1), (x2, y2), (25, 210, 50), 3)
            self.cv2.putText(annotated, f"{label} {confidence:.2f}", (x1, max(22, y1 - 8)),
                             self.cv2.FONT_HERSHEY_SIMPLEX, 0.75, (25, 210, 50), 2)
            detections.append({"class": label, "confidence": confidence,
                               "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        if not self.cv2.imwrite(str(output_path), annotated):
            raise OSError("บันทึกภาพพาโนรามาที่วาดกรอบไม่ได้")
        return detections

    def poll_events(self):
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "connected":
                    self.finish_connect(*event[1:])
                elif event[0] == "connect_error":
                    self.busy = False
                    self.connect_button.configure(state="normal")
                    self.scan_button.configure(state="normal")
                    self.status_var.set("เชื่อมต่อไม่สำเร็จ ตรวจแหล่งภาพแล้วลองอีกครั้ง")
                    messagebox.showerror("กล้อง", event[1])
                elif event[0] == "session_ready":
                    self.begin_session()
                elif event[0] == "capture_ready":
                    self.finish_capture(*event[1:])
                elif event[0] in ("session_error", "capture_error"):
                    self.busy = False
                    if event[0] == "session_error":
                        self.start_button.configure(state="normal", fg_color=COLORS["blue"])
                        self.connect_button.configure(state="normal")
                        self.scan_button.configure(state="normal")
                    else:
                        self.capture_button.configure(state="normal", fg_color=COLORS["blue"])
                    self.refresh_progress()
                    self.status_var.set("ดำเนินการไม่สำเร็จ ลองอีกครั้งได้")
                    messagebox.showerror("โปรแกรม", event[1])
                elif event[0] == "progress":
                    self.log(f"วิธีที่ 2: มุม {event[1]}/8 ใช้เวลา {event[2]:.3f} วินาที")
                elif event[0] == "retake_ready":
                    _, index, frames, annotated, online, tracker = event
                    self.frames = frames
                    self.online = online
                    self.online_processing_history += sum(online["seconds"])
                    self.retakes += 1
                    self.tracker_online = tracker
                    self.batch = None
                    self.latest_summary = None
                    self.refresh_panorama()
                    self.cv2.imwrite(str(self.run_dir / "raw" / f"view_{index + 1:02d}.jpg"), frames[index])
                    for view, image in enumerate(annotated, 1):
                        self.cv2.imwrite(str(self.run_dir / "online" / f"view_{view:02d}.jpg"), image)
                    self.capture_elapsed = time.perf_counter() - self.first_capture
                    self.show_thumbnail(index, frames[index])
                    self.select_capture_view(index)
                    self.select_review_view(index)
                    self.log(f"ถ่ายใหม่มุม {index + 1} และคำนวณผลวิธีที่ 1 ใหม่ครบ {len(frames)} มุม")
                    if len(frames) == VIEW_COUNT:
                        self.summary_status.set("กำลังคำนวณผลใหม่หลังถ่ายมุมเดิมซ้ำ")
                        self.status_var.set("กำลังคำนวณผลวิธีที่ 2 และภาพรวมใหม่...")
                        threading.Thread(target=self.run_batch, daemon=True).start()
                    else:
                        self.busy = False
                        self.capture_button.configure(state="normal", fg_color=COLORS["blue"])
                        self.refresh_progress()
                        self.status_var.set(f"ถ่ายใหม่มุม {index + 1} แล้ว | พร้อมถ่ายมุมที่ {len(frames) + 1}")
                elif event[0] == "retake_error":
                    self.busy = False
                    self.refresh_progress()
                    if len(self.frames) < VIEW_COUNT:
                        self.capture_button.configure(state="normal", fg_color=COLORS["blue"])
                    else:
                        self.start_button.configure(state="normal", fg_color=COLORS["blue"])
                        self.connect_button.configure(state="normal")
                        self.scan_button.configure(state="normal")
                    self.status_var.set("ถ่ายใหม่ไม่สำเร็จ ภาพเดิมยังอยู่")
                    messagebox.showerror("ถ่ายใหม่", event[1])
                elif event[0] == "done":
                    self.busy = False
                    summary = event[1]
                    self.batch = event[2]
                    self.latest_summary = summary
                    self.status_var.set(
                        "เสร็จแล้ว | วิธีที่ 1: %.2f วินาที | วิธีที่ 2 (ประมาณ): %.2f วินาที" %
                        (summary["online_total_seconds"], summary["estimated_batch_total_seconds"])
                    )
                    self.log(f"บันทึกผลครบแล้วที่ {self.run_dir}")
                    self.start_button.configure(state="normal", fg_color=COLORS["blue"])
                    self.connect_button.configure(state="normal")
                    self.scan_button.configure(state="normal")
                    self.refresh_progress()
                    self.refresh_review()
                    self.refresh_panorama()
                    self.summary_status.set(
                        "พร้อมดูผลรายมุมและพาโนรามา" if summary["panorama_stitched"]
                        else "พร้อมดูผลรายมุมและภาพรวม 8 มุม (ต่อพาโนรามาไม่สำเร็จ)"
                    )
                    self.online_metric_var.set(
                        f"ตรวจพบ {summary['detections_online']} รายการ  ·  รวม {summary['online_total_seconds']:.2f} วินาที"
                    )
                    self.batch_metric_var.set(
                        f"ตรวจพบ {summary['detections_batch']} รายการ  ·  รวมประมาณ {summary['estimated_batch_total_seconds']:.2f} วินาที"
                    )
                    details = (
                        f"ภาพทั้งหมด: {summary['views']} มุม    "
                        f"วัตถุที่ตรวจพบ: วิธีที่ 1 = {summary['detections_online']}, "
                        f"วิธีที่ 2 = {summary['detections_batch']}\n"
                        f"เวลาประมวลผล: วิธีที่ 1 = {summary['online_processing_seconds']:.2f} วินาที, "
                        f"วิธีที่ 2 = {summary['batch_processing_seconds']:.2f} วินาที\n"
                        f"เวลารวม: วิธีที่ 1 = {summary['online_total_seconds']:.2f} วินาที, "
                        f"วิธีที่ 2 (ค่าประมาณ) = {summary['estimated_batch_total_seconds']:.2f} วินาที\n\n"
                        "ไฟล์ที่บันทึก: ภาพรายมุม, ภาพรวมพร้อมกรอบ, CSV และ summary.json\n"
                        "มุมถ่ายเป็นค่าประมาณจากการหันกล้องด้วยมือ ภาพรวม 8 มุมไม่ใช่ภาพ 360 องศาที่มองได้พร้อมกัน"
                    )
                    if summary["retakes"]:
                        details += (f"\nถ่ายใหม่ {summary['retakes']} ครั้ง: เวลาวิธีที่ 1 รวมการประมวลผลซ้ำ "
                                    "จึงไม่ควรนำเวลารอบนี้ไปเปรียบเทียบสองวิธีแบบควบคุม")
                    self.set_readonly_text(self.summary_details, details)
                    self.show_summary()
                elif event[0] == "scan_done":
                    self.scan_button.configure(state="normal")
                    found = event[1]
                    if found:
                        self.kind_var.set("USB / virtual camera")
                        self.source_var.set(str(found[0][0]))
                        self.backend_var.set(found[0][1])
                        found_text = ", ".join(f"{i} ({b})" for i, b in found)
                        self.status_var.set(f"พบกล้อง: {found_text}")
                        self.log(f"พบกล้อง: {found_text}")
                    else:
                        self.status_var.set("ไม่พบกล้อง 0–5; ลอง URL สตรีมหรือ HTTP snapshot")
                elif event[0] == "error":
                    self.busy = False
                    self.refresh_progress()
                    self.scan_button.configure(state="normal")
                    self.connect_button.configure(state="normal")
                    self.start_button.configure(state="normal", fg_color=COLORS["blue"])
                    self.status_var.set("ประมวลผลวิธีที่ 2 ไม่สำเร็จ")
                    messagebox.showerror("ประมวลผล", event[1])
        except queue.Empty:
            pass
        self.root.after(100, self.poll_events)

    def open_output(self):
        if self.run_dir and self.run_dir.exists():
            os.startfile(str(self.run_dir.resolve()))

    def close(self):
        self.closed = True
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


def main():
    register_font()
    root = tk.Tk()
    try:
        DemoGUI(root)
    except SystemExit as exc:
        root.destroy()
        raise SystemExit(exc) from exc
    root.mainloop()


if __name__ == "__main__":
    main()
