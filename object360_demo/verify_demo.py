"""Integration checks using generated frames; no personal camera data."""
import json, tempfile, time, threading
from pathlib import Path
import tkinter as tk
import numpy as np
import cv2
from gui import DemoGUI
from camera_sources import CameraSource
from font_setup import register_font
register_font()
root=tk.Tk(); app=DemoGUI(root); app.cv2=cv2
errors=[]
root.report_callback_exception=lambda *args: errors.append(str(args))
beats=[]
def tick():
    beats.append(time.monotonic()); root.after(20,tick)
root.after(20,tick)
def wait(predicate, timeout=120):
    end=time.monotonic()+timeout
    while not predicate():
        root.update(); time.sleep(.01)
        if time.monotonic()>end: raise AssertionError('Timed out: '+app.status_var.get())
    assert not errors, errors
class FakeCamera:
    def isOpened(self): return True
    def read(self): return True, np.full((480,640,3),110,np.uint8)
    def release(self): pass
with tempfile.TemporaryDirectory(prefix='object360_verify_', dir=Path.cwd()) as folder:
    app.output_root=Path(folder); app.cap=FakeCamera(); app.latest_frame=app.cap.read()[1]
    app.connected_description='Generated test frames'
    started=time.monotonic(); app.start_session(); assert time.monotonic()-started<.3
    wait(lambda: app.run_dir is not None and not app.busy)
    loaded=app.model
    for i in range(8):
        app.latest_frame=np.full((480,640,3),100+i*5,np.uint8)
        started=time.monotonic(); app.capture(); assert time.monotonic()-started<.3
        wait(lambda: len(app.frames)==i+1)
    wait(lambda: app.latest_summary is not None)
    assert app.latest_summary['views']==8
    assert len(list((app.run_dir/'raw').glob('*.jpg')))==8
    app.switch_summary_mode('พาโนรามา / ภาพรวม'); root.update()
    app.switch_summary_mode('ภาพรายมุม'); app.select_review_view(7); root.update()
    app.select_capture_view(3); app.retake_selected()
    wait(lambda: app.retakes==1 and not app.busy and app.latest_summary is not None)
    assert app.latest_summary['retakes']==1
    print('PASS eight captures, both processing methods, summary, retake')
    app.start_session(); wait(lambda: not app.busy)
    assert app.model is loaded
    print('PASS model reused; UI heartbeat count',len(beats))
    root.geometry('1100x680'); root.update()
    assert app.capture_button.winfo_exists()
    print('PASS compact layout',root.winfo_width(),root.winfo_height())
    # Real OpenCV video adapter, asynchronous reads.
    video=Path(folder)/'camera.avi'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),25,(320,240))
    for _ in range(50): writer.write(np.zeros((240,320,3),np.uint8))
    writer.release()
    cam=CameraSource(cv2,'Video file',str(video))
    wait(lambda: cam.read()[0],10)
    began=time.monotonic(); assert cam.read()[0]; assert time.monotonic()-began<.05
    cam.release(); print('PASS buffered camera adapter')
    app.close()
print('ALL CHECKS PASSED')

