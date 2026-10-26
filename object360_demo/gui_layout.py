"""The dark research-workbench interface for Object 360 Studio."""

import tkinter as tk

import customtkinter as ctk

from camera_sources import BACKENDS, SOURCE_TYPES
from demo import VIEW_COUNT
from font_setup import FONT_FAMILY


COLORS = {
    "bg": "#0B1120",
    "panel": "#111E32",
    "raised": "#1C2E49",
    "border": "#304563",
    "text": "#f0f6ff",
    "muted": "#aabed8",
    "blue": "#367BF5",
    "blue_hover": "#3490ff",
    "green": "#2bd28a",
    "empty": "#1b2c45",
}
FONT = FONT_FAMILY


def _label(parent, text, size=12, weight="normal", color=None, **kwargs):
    return ctk.CTkLabel(parent, text=text, font=(FONT, size + 2, weight),
                        text_color=color or COLORS["text"], **kwargs)


def _button(parent, text, command, primary=False, height=38, **kwargs):
    c = COLORS
    return ctk.CTkButton(
        parent, text=text, command=command, height=height, corner_radius=12,
        fg_color=c["blue"] if primary else c["raised"],
        hover_color=c["blue_hover"] if primary else c["border"],
        text_color="white" if primary else c["text"],
        border_width=0 if primary else 1, border_color=c["border"],
        font=(FONT, 15, "bold" if primary else "normal"), **kwargs,
    )


def _field_label(parent, text):
    _label(parent, text, 11, color=COLORS["muted"], anchor="w").pack(fill="x", pady=(0, 3))


def build_ui(app):
    ctk.set_appearance_mode("dark")
    c = COLORS
    root = app.root
    width = min(1440, root.winfo_screenwidth() - 60)
    height = min(900, root.winfo_screenheight() - 90)
    root.geometry(f"{width}x{height}")
    root.minsize(1060, 640)
    root.configure(bg=c["bg"])

    header = ctk.CTkFrame(root, fg_color=c["bg"], corner_radius=0)
    header.pack(fill="x", padx=20, pady=(12, 4))
    heading = ctk.CTkFrame(header, fg_color="transparent")
    heading.pack(side="left")
    _label(heading, "Object 360 Studio", 26, "bold", anchor="w").pack(fill="x")
    _label(heading, "ตรวจจับและติดตามวัตถุจากภาพ 8 ทิศทาง", 12, color=c["muted"], anchor="w").pack(fill="x")
    health = ctk.CTkFrame(header, fg_color="transparent")
    health.pack(side="right")
    app.camera_state = tk.StringVar(value="● กล้อง: ยังไม่เชื่อมต่อ")
    app.model_state = tk.StringVar(value="● YOLO: ยังไม่พร้อม")
    app.tracker_state = tk.StringVar(value="● Deep SORT: ยังไม่พร้อม")
    for var in (app.camera_state, app.model_state, app.tracker_state):
        _label(health, "", 12, color=c["muted"], textvariable=var).pack(side="left", padx=9)

    app.tabs = ctk.CTkTabview(
        root, fg_color=c["bg"], corner_radius=10,
        segmented_button_fg_color=c["panel"],
        segmented_button_selected_color=c["blue"],
        segmented_button_selected_hover_color=c["blue_hover"],
        segmented_button_unselected_color=c["raised"],
        segmented_button_unselected_hover_color=c["border"],
        text_color=c["text"], anchor="w",
    )
    app.tabs.pack(fill="both", expand=True, padx=8, pady=(0, 8))
    app.capture_page = app.tabs.add("ถ่ายภาพ 8 มุม")
    app.summary_page = app.tabs.add("สรุปผล")
    app.tabs._segmented_button.configure(font=(FONT, 14, "bold"), height=34)

    page = app.capture_page
    page.grid_columnconfigure(0, minsize=250)
    page.grid_columnconfigure(1, weight=1)
    page.grid_columnconfigure(2, minsize=278)
    page.grid_rowconfigure(0, weight=1)
    left = ctk.CTkScrollableFrame(page, width=230, fg_color=c["panel"], corner_radius=12, scrollbar_button_color=c["border"])
    left.grid(row=0, column=0, sticky="nsew", padx=(8, 5), pady=(4, 6))
    _label(left, "01  ตั้งค่ากล้อง", 18, "bold", anchor="w").pack(fill="x", padx=16, pady=(13, 12))
    _label(left, "เชื่อมต่อ → เริ่มรอบ → ถ่ายครบ 8 มุม", 10, color=c["muted"], anchor="w").pack(fill="x", padx=16, pady=(0, 10))
    fields = ctk.CTkFrame(left, fg_color="transparent")
    fields.pack(fill="x", padx=16)
    app.kind_var = tk.StringVar(value=SOURCE_TYPES[0])
    _field_label(fields, "ชนิดแหล่งภาพ")
    ctk.CTkComboBox(fields, variable=app.kind_var, values=list(SOURCE_TYPES), height=36,
                    fg_color=c["raised"], border_color=c["border"], button_color=c["blue"],
                    dropdown_fg_color=c["panel"], text_color=c["text"], font=(FONT, 14)).pack(fill="x", pady=(0, 9))
    app.source_var = tk.StringVar(value="0")
    _field_label(fields, "หมายเลขกล้อง / URL / พาธไฟล์")
    ctk.CTkEntry(fields, textvariable=app.source_var, height=36, corner_radius=8,
                 fg_color=c["raised"], border_color=c["border"], text_color=c["text"],
                 font=(FONT, 14)).pack(fill="x", pady=(0, 9))
    app.backend_var = tk.StringVar(value="Auto")
    _field_label(fields, "ระบบอ่านภาพ")
    ctk.CTkComboBox(fields, variable=app.backend_var, values=list(BACKENDS), height=36,
                    fg_color=c["raised"], border_color=c["border"], button_color=c["blue"],
                    dropdown_fg_color=c["panel"], text_color=c["text"], font=(FONT, 14)).pack(fill="x", pady=(0, 9))
    app.weights_var = tk.StringVar(value="yolo11n.pt")
    _field_label(fields, "โมเดล YOLO")
    ctk.CTkEntry(fields, textvariable=app.weights_var, height=36, corner_radius=8,
                 fg_color=c["raised"], border_color=c["border"], text_color=c["text"],
                 font=(FONT, 14)).pack(fill="x", pady=(0, 9))
    small = ctk.CTkFrame(left, fg_color="transparent")
    small.pack(fill="x", padx=16, pady=(3, 7))
    small.grid_columnconfigure((0, 1), weight=1)
    app.scan_button = _button(small, "ค้นหากล้อง", app.scan, height=36)
    app.scan_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
    app.file_button = _button(small, "เลือกวิดีโอ", app.choose_video, height=36)
    app.file_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))
    app.connect_button = _button(left, "เชื่อมต่อกล้อง", app.connect, height=38)
    app.connect_button.pack(fill="x", padx=16, pady=(0, 7))
    app.start_button = _button(left, "เริ่มรอบทดลอง", app.start_session, primary=True, height=40)
    app.start_button.pack(fill="x", padx=16, pady=(0, 13))
    app.start_button.configure(state="disabled", fg_color=c["raised"], text_color_disabled=c["muted"])

    center = ctk.CTkFrame(page, fg_color="transparent")
    center.grid(row=0, column=1, sticky="nsew", padx=5, pady=(4, 6))
    center.grid_columnconfigure((0, 1), weight=1, uniform="preview")
    center.grid_rowconfigure(0, weight=1)
    preview_cards = []
    app.result_title_var = tk.StringVar(value="ผลตรวจจับล่าสุด")
    for column, title in enumerate(("ภาพสดจากกล้อง", "")):
        card = ctk.CTkFrame(center, fg_color=c["panel"], corner_radius=12,
                            border_width=1, border_color=c["border"])
        card.grid(row=0, column=column, sticky="nsew", padx=(0, 5) if column == 0 else (5, 0))
        _label(card, title, 16, "bold", anchor="w",
               **({"textvariable": app.result_title_var} if column == 1 else {})).pack(
                   fill="x", padx=13, pady=(11, 8))
        preview_cards.append(card)
    app.live_label = tk.Label(preview_cards[0], text="◉\nเชื่อมต่อกล้องเพื่อดูภาพสด", bg=c["empty"],
                              fg=c["muted"], font=(FONT, 15), justify="center")
    app.live_label.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    app.result_label = tk.Label(preview_cards[1], text="◎\nผล YOLO และ Deep SORT จะแสดงที่นี่", bg=c["empty"],
                                fg=c["muted"], font=(FONT, 15), justify="center")
    app.result_label.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    right = ctk.CTkScrollableFrame(page, width=250, fg_color=c["panel"], corner_radius=12,
                                   scrollbar_button_color=c["border"])
    right.grid(row=0, column=2, sticky="nsew", padx=(5, 8), pady=(4, 6))
    _label(right, "02  บันทึกภาพรอบตัว", 17, "bold", anchor="w").pack(fill="x", padx=15, pady=(13, 0))
    app.progress_var = tk.StringVar(value="0 / 8 มุม")
    _label(right, "", 12, color=c["muted"], textvariable=app.progress_var, anchor="w").pack(fill="x", padx=15, pady=(0, 7))
    app.progress = ctk.CTkProgressBar(right, height=9, corner_radius=5, fg_color=c["empty"], progress_color=c["blue"])
    app.progress.pack(fill="x", padx=15, pady=(0, 10))
    app.progress.set(0)
    _label(right, "หันกล้องครั้งละประมาณ 45°\nรอภาพนิ่งแล้วกดถ่ายแต่ละมุม", 11, color=c["muted"], justify="left", anchor="w").pack(fill="x", padx=15, pady=(0, 10))
    steps = ctk.CTkFrame(right, fg_color="transparent")
    steps.pack(fill="x", padx=15)
    app.step_labels = []
    for i in range(VIEW_COUNT):
        label = tk.Label(steps, text=f"○   มุมที่ {i + 1}", anchor="w", bg=c["panel"], fg=c["muted"],
                         font=(FONT, 13), padx=8, pady=3)
        label.pack(fill="x")
        app.step_labels.append(label)
    app.capture_button = _button(right, "ถ่ายภาพมุมที่ 1", app.capture, primary=True, height=48)
    app.capture_button.pack(fill="x", padx=15, pady=(11, 7))
    app.capture_button.configure(state="disabled", fg_color=c["raised"], text_color_disabled=c["muted"])
    app.summary_button = _button(right, "ไปหน้าสรุปผล", app.show_summary, height=36)
    app.summary_button.pack(fill="x", padx=15)
    _label(right, "บันทึกการทำงาน", 14, "bold", anchor="w").pack(fill="x", padx=15, pady=(10, 4))
    app.log_box = ctk.CTkTextbox(right, height=80, fg_color=c["empty"], text_color=c["text"],
                                 font=(FONT, 13), corner_radius=7, border_width=0, wrap="word")
    app.log_box.pack(fill="both", expand=True, padx=15, pady=(0, 13))
    app.log_box.configure(state="disabled")

    bottom = ctk.CTkFrame(page, fg_color=c["panel"], corner_radius=12,
                          border_width=1, border_color=c["border"])
    bottom.grid(row=1, column=0, columnspan=3, sticky="ew", padx=8, pady=(0, 6))
    bar = ctk.CTkFrame(bottom, fg_color="transparent")
    bar.pack(fill="x", padx=14, pady=(8, 6))
    _label(bar, "ภาพที่ถ่ายแล้ว · คลิกภาพเพื่อเลือกดูหรือถ่ายใหม่", 15, "bold").pack(side="left")
    app.open_button = _button(bar, "เปิดโฟลเดอร์ผล", app.open_output, height=32)
    app.open_button.pack(side="right")
    app.open_button.configure(state="disabled")
    app.retake_button = _button(bar, "ถ่ายใหม่มุมที่เลือก", app.retake_selected, height=32)
    app.retake_button.pack(side="right", padx=(0, 8))
    app.retake_button.configure(state="disabled", fg_color=c["raised"], text_color_disabled=c["muted"])
    tiles = ctk.CTkFrame(bottom, fg_color="transparent")
    tiles.pack(fill="x", padx=14, pady=(0, 9))
    app.view_cards = []
    for i in range(VIEW_COUNT):
        tiles.grid_columnconfigure(i, weight=1, uniform="tiles")
        tile = tk.Label(tiles, text=f"{i + 1}\nรอถ่าย", bg=c["empty"], fg=c["muted"],
                        font=(FONT, 13), height=5, relief="solid", bd=1,
                        cursor="hand2", highlightthickness=2, highlightbackground=c["empty"])
        tile.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 4, 0))
        tile.bind("<Button-1>", lambda _event, index=i: app.select_capture_view(index))
        app.view_cards.append(tile)
    app.status_var = tk.StringVar(value="เลือกแหล่งภาพและกดเชื่อมต่อกล้อง")
    _label(page, "", 11, color=c["muted"], textvariable=app.status_var, anchor="w").grid(
        row=2, column=0, columnspan=3, sticky="ew", padx=14, pady=(0, 2))

    summary = ctk.CTkScrollableFrame(app.summary_page, fg_color=c["bg"],
                                     scrollbar_button_color=c["border"],
                                     scrollbar_button_hover_color=c["blue"])
    summary.pack(fill="both", expand=True)
    app.summary_scroll = summary
    top = ctk.CTkFrame(summary, fg_color="transparent")
    top.pack(fill="x", padx=13, pady=(9, 5))
    _label(top, "สรุปผลการทดลอง", 21, "bold").pack(side="left")
    _button(top, "เปิดโฟลเดอร์ผล", app.open_output, height=34).pack(side="right")
    app.summary_status = tk.StringVar(value="ยังไม่มีผลการทดลอง กดถ่ายภาพให้ครบ 8 มุมก่อน")
    _label(summary, "", 12, color=c["muted"], textvariable=app.summary_status,
           anchor="w").pack(fill="x", padx=14, pady=(0, 6))

    app.summary_modes = ctk.CTkSegmentedButton(
        summary, values=["ภาพรายมุม", "พาโนรามา / ภาพรวม"], command=app.switch_summary_mode,
        selected_color=c["blue"], selected_hover_color=c["blue_hover"],
        unselected_color=c["raised"], unselected_hover_color=c["border"],
        font=(FONT, 15, "bold"), height=38,
    )
    app.summary_modes.pack(fill="x", padx=13, pady=(0, 8))
    body = ctk.CTkFrame(summary, fg_color="transparent")
    body.pack(fill="x", padx=13, pady=(0, 9))

    app.per_view_frame = ctk.CTkFrame(body, fg_color=c["panel"], corner_radius=12,
                                      border_width=1, border_color=c["border"])
    _label(app.per_view_frame, "ตรวจจับในแต่ละภาพ", 17, "bold", anchor="w").pack(
        fill="x", padx=14, pady=(10, 5))
    app.review_method = ctk.CTkSegmentedButton(
        app.per_view_frame, values=["วิธีที่ 1", "วิธีที่ 2"], command=app.change_review_method,
        selected_color=c["blue"], selected_hover_color=c["blue_hover"],
        unselected_color=c["raised"], unselected_hover_color=c["border"],
        font=(FONT, 13), height=31,
    )
    app.review_method.pack(fill="x", padx=14, pady=(0, 8))
    review_buttons = ctk.CTkFrame(app.per_view_frame, fg_color="transparent")
    review_buttons.pack(fill="x", padx=14, pady=(0, 7))
    app.review_buttons = []
    for i in range(VIEW_COUNT):
        review_buttons.grid_columnconfigure(i, weight=1, uniform="review")
        button = _button(review_buttons, f"มุม {i + 1}", lambda index=i: app.select_review_view(index), height=31)
        button.grid(row=0, column=i, sticky="ew", padx=3)
        app.review_buttons.append(button)
    app.review_title = tk.StringVar(value="เลือกมุมเพื่อดูภาพและวัตถุที่ตรวจพบ")
    _label(app.per_view_frame, "", 13, color=c["muted"], textvariable=app.review_title,
           anchor="w").pack(fill="x", padx=14, pady=(0, 5))
    review_image_box = ctk.CTkFrame(app.per_view_frame, fg_color=c["empty"], height=330)
    review_image_box.pack(fill="x", padx=14)
    review_image_box.pack_propagate(False)
    app.review_image = tk.Label(review_image_box, text="ยังไม่มีภาพที่ถ่าย", bg=c["empty"],
                                fg=c["muted"], font=(FONT, 15))
    app.review_image.pack(fill="both", expand=True)
    app.review_findings = ctk.CTkTextbox(app.per_view_frame, height=112, fg_color=c["empty"],
                                          text_color=c["text"], font=(FONT, 13),
                                          corner_radius=8, wrap="word")
    app.review_findings.pack(fill="x", padx=14, pady=(8, 13))
    app.review_findings.configure(state="disabled")

    app.panorama_frame = ctk.CTkFrame(body, fg_color=c["panel"], corner_radius=12,
                                      border_width=1, border_color=c["border"])
    app.panorama_title = tk.StringVar(value="พาโนรามาหรือภาพรวม 8 มุม")
    _label(app.panorama_frame, "", 17, "bold", textvariable=app.panorama_title,
           anchor="w").pack(fill="x", padx=14, pady=(10, 6))
    panorama_image_box = ctk.CTkFrame(app.panorama_frame, fg_color=c["empty"], height=350)
    panorama_image_box.pack(fill="x", padx=14)
    panorama_image_box.pack_propagate(False)
    app.summary_image = tk.Label(panorama_image_box, text="ภาพรวมจะแสดงหลังถ่ายครบ 8 มุม", bg=c["empty"],
                                 fg=c["muted"], font=(FONT, 15))
    app.summary_image.pack(fill="both", expand=True)
    app.panorama_findings = ctk.CTkTextbox(app.panorama_frame, height=115, fg_color=c["empty"],
                                            text_color=c["text"], font=(FONT, 13),
                                            corner_radius=8, wrap="word")
    app.panorama_findings.pack(fill="x", padx=14, pady=(8, 13))
    app.panorama_findings.configure(state="disabled")

    app.summary_body = body
    app.review_method.set("วิธีที่ 1")
    app.summary_modes.set("ภาพรายมุม")
    app.per_view_frame.pack(fill="x")
    details_card = ctk.CTkFrame(summary, fg_color=c["panel"], corner_radius=12,
                                border_width=1, border_color=c["border"])
    details_card.pack(fill="x", padx=13, pady=(0, 9))
    _label(details_card, "เปรียบเทียบผลการประมวลผล", 17, "bold", anchor="w").pack(fill="x", padx=14, pady=(10, 5))
    metrics = ctk.CTkFrame(details_card, fg_color="transparent")
    metrics.pack(fill="x", padx=12, pady=(0, 7))
    metrics.grid_columnconfigure((0, 1), weight=1, uniform="methods")
    app.online_metric_var = tk.StringVar(value="ยังไม่มีผล")
    app.batch_metric_var = tk.StringVar(value="ยังไม่มีผล")
    for column, title, var in ((0, "วิธีที่ 1 · ประมวลผลทีละภาพ", app.online_metric_var),
                               (1, "วิธีที่ 2 · ประมวลผลหลังถ่ายครบ", app.batch_metric_var)):
        metric_card = ctk.CTkFrame(metrics, fg_color=c["empty"], corner_radius=9)
        metric_card.grid(row=0, column=column, sticky="ew", padx=(0, 4) if column == 0 else (4, 0))
        _label(metric_card, title, 13, "bold", anchor="w").pack(fill="x", padx=12, pady=(8, 0))
        _label(metric_card, "", 12, color=c["muted"], textvariable=var, anchor="w").pack(fill="x", padx=12, pady=(0, 7))
    app.summary_details = ctk.CTkTextbox(details_card, height=95, fg_color=c["empty"], text_color=c["text"],
                                         font=(FONT, 14), corner_radius=8, wrap="word")
    app.summary_details.pack(fill="x", padx=12, pady=(0, 12))
    app.summary_details.configure(state="disabled")
