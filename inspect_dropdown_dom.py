import gradio as gr
import subprocess
import time
import re

with gr.Blocks() as demo:
    with gr.Column(elem_classes=["sidebar-panel"]):
        dd1 = gr.Dropdown(label="Answer style", choices=["Detailed", "Short"], value="Detailed", elem_classes=["sidebar-dropdown"], elem_id="answer-style-dd")
        dd2 = gr.Dropdown(label="Summary length", choices=["Detailed", "Brief"], value="Detailed", elem_classes=["sidebar-dropdown"], elem_id="summary-length-dd")

demo.launch(server_port=7869, prevent_thread_lock=True)
time.sleep(2)

chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
res = subprocess.run(
    [chrome_path, "--headless=new", "--dump-dom", "--virtual-time-budget=3000", "http://127.0.0.1:7869/"],
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="ignore"
)

demo.close()
html = res.stdout

idx = html.find("answer-style-dd")
print("--- 600 CHARS BEFORE answer-style-dd ---")
print(html[max(0, idx-600):idx+50])
