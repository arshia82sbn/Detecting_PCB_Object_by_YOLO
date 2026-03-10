import json
import os
import queue
import shutil
import subprocess
import threading
import time
from tkinter import filedialog

import customtkinter as ctk


APP_TITLE = "PCB YOLO Pipeline"
APP_SUBTITLE = "Visual setup and one-click runs for dataset, training, export, and inference"
APP_VERSION = "v1.2"

ACCENT = "#2d7ff9"
ACCENT_DARK = "#1f5fc6"
DANGER = "#b23b3b"
CARD_BG = "#1b1f27"
CARD_BG_LIGHT = "#f4f6fb"
TEXT_MUTED = "#9aa4b2"


def find_python_exe(root):
    candidate = os.path.join(root, "image_process", "Scripts", "python.exe")
    if os.path.exists(candidate):
        return candidate
    return os.path.join(root, "pcb_yolo", "image_process", "Scripts", "python.exe")


def abs_path(path, root):
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(root, path))


class PipelineUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1260x760")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.repo_root = os.path.dirname(self.root_dir)
        self.state_file = os.path.join(self.repo_root, ".pcb_ui_state.json")

        self.proc = None
        self.last_exit_code = None
        self.log_q = queue.Queue()
        self.active_step = None
        self.active_steps = []
        self.completed_steps = 0

        self._build_layout()
        self._load_state()
        self._poll_logs()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        sidebar = ctk.CTkFrame(self, width=280, corner_radius=12)
        sidebar.grid(row=0, column=0, sticky="nsw")
        sidebar.grid_rowconfigure(18, weight=1)

        brand = ctk.CTkFrame(sidebar, corner_radius=12, fg_color=CARD_BG)
        brand.grid(row=0, column=0, padx=16, pady=(18, 10), sticky="ew")
        brand.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            brand, text="PCB YOLO", font=ctk.CTkFont(size=22, weight="bold")
        ).grid(row=0, column=0, padx=14, pady=(12, 2), sticky="w")
        ctk.CTkLabel(
            brand, text="Defect Detection Suite", font=ctk.CTkFont(size=12), text_color=TEXT_MUTED
        ).grid(row=1, column=0, padx=14, pady=(0, 6), sticky="w")
        ctk.CTkLabel(
            brand, text=f"{APP_SUBTITLE}  |  {APP_VERSION}", font=ctk.CTkFont(size=10), text_color=TEXT_MUTED, wraplength=220
        ).grid(row=2, column=0, padx=14, pady=(0, 12), sticky="w")

        self.btn_run_all = ctk.CTkButton(
            sidebar,
            text="Run Full Pipeline >>",
            fg_color=ACCENT,
            hover_color=ACCENT_DARK,
            command=self.run_full_pipeline,
        )
        self.btn_run_all.grid(row=1, column=0, padx=18, pady=(0, 8), sticky="ew")

        self.btn_stop = ctk.CTkButton(
            sidebar,
            text="Stop Current !!",
            fg_color=DANGER,
            hover_color="#8f2f2f",
            command=self.stop_proc,
        )
        self.btn_stop.grid(row=2, column=0, padx=18, pady=(0, 12), sticky="ew")

        self.btn_wizard = ctk.CTkButton(
            sidebar,
            text="Guided Setup >>",
            border_width=1,
            fg_color="transparent",
            text_color=ACCENT,
            hover_color="#1a2538",
            command=self.open_setup_wizard,
        )
        self.btn_wizard.grid(row=3, column=0, padx=18, pady=(0, 16), sticky="ew")

        ctk.CTkLabel(
            sidebar, text="Python", font=ctk.CTkFont(size=12, weight="bold")
        ).grid(row=4, column=0, padx=18, pady=(2, 4), sticky="w")
        self.python_entry = ctk.CTkEntry(sidebar, placeholder_text="Path to python.exe")
        self.python_entry.grid(row=5, column=0, padx=18, pady=(0, 8), sticky="ew")

        self.pipeline_progress = ctk.CTkProgressBar(sidebar)
        self.pipeline_progress.set(0)
        self.pipeline_progress.grid(row=6, column=0, padx=18, pady=(6, 2), sticky="ew")
        self.pipeline_progress_label = ctk.CTkLabel(
            sidebar, text="Pipeline progress: 0/0", font=ctk.CTkFont(size=11), text_color=TEXT_MUTED
        )
        self.pipeline_progress_label.grid(row=7, column=0, padx=18, pady=(0, 8), sticky="w")

        self.run_progress = ctk.CTkProgressBar(sidebar)
        self.run_progress.set(0)
        self.run_progress.grid(row=8, column=0, padx=18, pady=(2, 2), sticky="ew")
        self.run_progress_label = ctk.CTkLabel(
            sidebar, text="Step status: idle", font=ctk.CTkFont(size=11), text_color=TEXT_MUTED
        )
        self.run_progress_label.grid(row=9, column=0, padx=18, pady=(0, 6), sticky="w")

        self.theme_toggle = ctk.CTkSwitch(
            sidebar, text="Light Mode", command=self.toggle_theme
        )
        self.theme_toggle.grid(row=10, column=0, padx=18, pady=(2, 12), sticky="w")

        ctk.CTkLabel(
            sidebar, text="Logs", font=ctk.CTkFont(size=12, weight="bold")
        ).grid(row=11, column=0, padx=18, pady=(6, 4), sticky="w")
        self.log_box = ctk.CTkTextbox(sidebar, height=180)
        self.log_box.grid(row=12, column=0, padx=18, pady=(0, 12), sticky="ew")
        self.log_box.configure(state="disabled")

        self.status_label = ctk.CTkLabel(
            sidebar, text="Idle", font=ctk.CTkFont(size=12)
        )
        self.status_label.grid(row=13, column=0, padx=18, pady=(8, 16), sticky="w")

        main = ctk.CTkTabview(self, corner_radius=14)
        main.grid(row=0, column=1, sticky="nsew", padx=16, pady=16)

        self.tab_pipeline = main.add("Pipeline")
        self.tab_prepare = main.add("Prepare")
        self.tab_train = main.add("Train")
        self.tab_export = main.add("Export")
        self.tab_deploy = main.add("Deploy")
        self.tab_settings = main.add("Settings")

        self._build_pipeline_tab()
        self._build_prepare_tab()
        self._build_train_tab()
        self._build_export_tab()
        self._build_deploy_tab()
        self._build_settings_tab()

    def _build_pipeline_tab(self):
        frame = self.tab_pipeline
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            frame, text="Full Pipeline", font=ctk.CTkFont(size=20, weight="bold")
        ).grid(row=0, column=0, padx=18, pady=(18, 8), sticky="w")
        ctk.CTkLabel(
            frame,
            text="Select the steps you want to run. Steps execute in order.",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        ).grid(row=1, column=0, padx=18, pady=(0, 12), sticky="w")

        status_card = ctk.CTkFrame(frame, corner_radius=12, fg_color=CARD_BG)
        status_card.grid(row=2, column=0, padx=18, pady=(0, 12), sticky="ew")
        status_card.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            status_card, text="Status Board", font=ctk.CTkFont(size=13, weight="bold")
        ).grid(row=0, column=0, columnspan=2, padx=12, pady=(10, 8), sticky="w")

        self.step_status_labels = {}
        for idx, name in enumerate(["Prepare", "Train", "Export", "Deploy"]):
            ctk.CTkLabel(status_card, text=f"{name}:", text_color=TEXT_MUTED).grid(
                row=idx + 1, column=0, padx=12, pady=4, sticky="w"
            )
            label = ctk.CTkLabel(status_card, text="Idle")
            label.grid(row=idx + 1, column=1, padx=12, pady=4, sticky="w")
            self.step_status_labels[name.lower()] = label

        self.chk_prepare = ctk.CTkCheckBox(frame, text="1) Prepare dataset")
        self.chk_prepare.select()
        self.chk_prepare.grid(row=3, column=0, padx=18, pady=6, sticky="w")

        self.chk_train = ctk.CTkCheckBox(frame, text="2) Train model")
        self.chk_train.select()
        self.chk_train.grid(row=4, column=0, padx=18, pady=6, sticky="w")

        self.chk_export = ctk.CTkCheckBox(frame, text="3) Export model")
        self.chk_export.select()
        self.chk_export.grid(row=5, column=0, padx=18, pady=6, sticky="w")

        self.chk_deploy = ctk.CTkCheckBox(frame, text="4) Deploy / Inference")
        self.chk_deploy.select()
        self.chk_deploy.grid(row=6, column=0, padx=18, pady=6, sticky="w")

        self.auto_delete_after_train = ctk.CTkCheckBox(
            frame,
            text="Delete dataset after successful training (model must exist)",
        )
        self.auto_delete_after_train.grid(
            row=7, column=0, padx=18, pady=6, sticky="w"
        )

        ctk.CTkButton(
            frame, text="Run Selected Steps", command=self.run_full_pipeline
        ).grid(row=8, column=0, padx=18, pady=(12, 8), sticky="w")

    def _build_prepare_tab(self):
        frame = self.tab_prepare
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="Prepare Dataset", font=ctk.CTkFont(size=20, weight="bold")
        ).grid(row=0, column=0, columnspan=2, padx=18, pady=(18, 4), sticky="w")
        ctk.CTkLabel(
            frame,
            text="Use a local dataset or download from Roboflow with your API key.",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        ).grid(row=1, column=0, columnspan=2, padx=18, pady=(0, 12), sticky="w")

        ctk.CTkLabel(frame, text="Config").grid(
            row=2, column=0, padx=18, pady=6, sticky="w"
        )
        self.prepare_config = ctk.CTkEntry(frame, placeholder_text="pcb_yolo/configs/data_config.yaml")
        self.prepare_config.grid(row=2, column=1, padx=18, pady=6, sticky="ew")
        ctk.CTkButton(
            frame, text="Browse", command=lambda: self._browse_file(self.prepare_config)
        ).grid(row=2, column=2, padx=8, pady=6, sticky="w")

        self.prepare_download = ctk.CTkCheckBox(frame, text="Download from Roboflow")
        self.prepare_download.grid(row=3, column=1, padx=18, pady=6, sticky="w")

        ctk.CTkLabel(frame, text="Dataset dir").grid(
            row=4, column=0, padx=18, pady=6, sticky="w"
        )
        self.dataset_dir = ctk.CTkEntry(frame, placeholder_text="Detecting-the-PCB-object-3")
        self.dataset_dir.grid(row=4, column=1, padx=18, pady=6, sticky="ew")
        ctk.CTkButton(
            frame, text="Browse", command=lambda: self._browse_dir(self.dataset_dir)
        ).grid(row=4, column=2, padx=8, pady=6, sticky="w")

        self.prepare_status = ctk.CTkLabel(frame, text="Status: Idle", text_color=TEXT_MUTED)
        self.prepare_status.grid(row=5, column=0, padx=18, pady=(6, 6), sticky="w")

        ctk.CTkButton(frame, text="Run Prepare", command=self.run_prepare).grid(
            row=5, column=1, padx=18, pady=12, sticky="w"
        )

        ctk.CTkButton(
            frame, text="Delete Dataset", fg_color=DANGER, hover_color="#8f2f2f", command=self.delete_dataset
        ).grid(row=6, column=1, padx=18, pady=(0, 12), sticky="w")

    def _build_train_tab(self):
        frame = self.tab_train
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="Train Model", font=ctk.CTkFont(size=20, weight="bold")
        ).grid(row=0, column=0, columnspan=2, padx=18, pady=(18, 4), sticky="w")
        ctk.CTkLabel(
            frame,
            text="Run training with a config file and optional deterministic mode.",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        ).grid(row=1, column=0, columnspan=2, padx=18, pady=(0, 12), sticky="w")

        ctk.CTkLabel(frame, text="Train config").grid(
            row=2, column=0, padx=18, pady=6, sticky="w"
        )
        self.train_config = ctk.CTkEntry(frame, placeholder_text="pcb_yolo/configs/train_config.yaml")
        self.train_config.grid(row=2, column=1, padx=18, pady=6, sticky="ew")
        ctk.CTkButton(
            frame, text="Browse", command=lambda: self._browse_file(self.train_config)
        ).grid(row=2, column=2, padx=8, pady=6, sticky="w")

        ctk.CTkLabel(frame, text="Seed").grid(
            row=3, column=0, padx=18, pady=6, sticky="w"
        )
        self.train_seed = ctk.CTkEntry(frame, width=120)
        self.train_seed.grid(row=3, column=1, padx=18, pady=6, sticky="w")

        self.train_deterministic = ctk.CTkCheckBox(frame, text="Deterministic")
        self.train_deterministic.select()
        self.train_deterministic.grid(row=4, column=1, padx=18, pady=6, sticky="w")

        ctk.CTkLabel(frame, text="Expected model path").grid(
            row=5, column=0, padx=18, pady=6, sticky="w"
        )
        self.model_path = ctk.CTkEntry(frame, placeholder_text="pcb_yolo/experiments/pcb_train/weights/best.pt")
        self.model_path.grid(row=5, column=1, padx=18, pady=6, sticky="ew")
        ctk.CTkButton(
            frame, text="Browse", command=lambda: self._browse_file(self.model_path)
        ).grid(row=5, column=2, padx=8, pady=6, sticky="w")

        self.train_status = ctk.CTkLabel(frame, text="Status: Idle", text_color=TEXT_MUTED)
        self.train_status.grid(row=6, column=0, padx=18, pady=(6, 6), sticky="w")

        ctk.CTkButton(frame, text="Run Train", command=self.run_train).grid(
            row=6, column=1, padx=18, pady=12, sticky="w"
        )

    def _build_export_tab(self):
        frame = self.tab_export
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="Export Model", font=ctk.CTkFont(size=20, weight="bold")
        ).grid(row=0, column=0, columnspan=2, padx=18, pady=(18, 4), sticky="w")
        ctk.CTkLabel(
            frame,
            text="Convert the trained model for deployment formats.",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        ).grid(row=1, column=0, columnspan=2, padx=18, pady=(0, 12), sticky="w")

        ctk.CTkLabel(frame, text="Model path").grid(
            row=2, column=0, padx=18, pady=6, sticky="w"
        )
        self.export_model = ctk.CTkEntry(frame, placeholder_text="pcb_yolo/experiments/pcb_train/weights/best.pt")
        self.export_model.grid(row=2, column=1, padx=18, pady=6, sticky="ew")
        ctk.CTkButton(
            frame, text="Browse", command=lambda: self._browse_file(self.export_model)
        ).grid(row=2, column=2, padx=8, pady=6, sticky="w")

        ctk.CTkLabel(frame, text="Format").grid(
            row=3, column=0, padx=18, pady=6, sticky="w"
        )
        self.export_format = ctk.CTkComboBox(frame, values=["onnx", "torchscript"])
        self.export_format.set("onnx")
        self.export_format.grid(row=3, column=1, padx=18, pady=6, sticky="w")

        self.export_status = ctk.CTkLabel(frame, text="Status: Idle", text_color=TEXT_MUTED)
        self.export_status.grid(row=4, column=0, padx=18, pady=(6, 6), sticky="w")

        ctk.CTkButton(frame, text="Run Export", command=self.run_export).grid(
            row=4, column=1, padx=18, pady=12, sticky="w"
        )

    def _build_deploy_tab(self):
        frame = self.tab_deploy
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="Deploy / Inference", font=ctk.CTkFont(size=20, weight="bold")
        ).grid(row=0, column=0, columnspan=2, padx=18, pady=(18, 4), sticky="w")
        ctk.CTkLabel(
            frame,
            text="Run predictions on images or folders and save outputs.",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        ).grid(row=1, column=0, columnspan=2, padx=18, pady=(0, 12), sticky="w")

        ctk.CTkLabel(frame, text="Model path").grid(
            row=2, column=0, padx=18, pady=6, sticky="w"
        )
        self.deploy_model = ctk.CTkEntry(frame, placeholder_text="pcb_yolo/experiments/pcb_train/weights/best.pt")
        self.deploy_model.grid(row=2, column=1, padx=18, pady=6, sticky="ew")
        ctk.CTkButton(
            frame, text="Browse", command=lambda: self._browse_file(self.deploy_model)
        ).grid(row=2, column=2, padx=8, pady=6, sticky="w")

        ctk.CTkLabel(frame, text="Input").grid(
            row=3, column=0, padx=18, pady=6, sticky="w"
        )
        self.deploy_input = ctk.CTkEntry(frame, placeholder_text="data/mock/test/images")
        self.deploy_input.grid(row=3, column=1, padx=18, pady=6, sticky="ew")
        ctk.CTkButton(
            frame, text="Browse", command=lambda: self._browse_input(self.deploy_input)
        ).grid(row=3, column=2, padx=8, pady=6, sticky="w")

        ctk.CTkLabel(frame, text="Output dir").grid(
            row=4, column=0, padx=18, pady=6, sticky="w"
        )
        self.deploy_output = ctk.CTkEntry(frame, placeholder_text="pcb_yolo/experiments/predictions")
        self.deploy_output.grid(row=4, column=1, padx=18, pady=6, sticky="ew")
        ctk.CTkButton(
            frame, text="Browse", command=lambda: self._browse_dir(self.deploy_output)
        ).grid(row=4, column=2, padx=8, pady=6, sticky="w")

        ctk.CTkLabel(frame, text="Deploy config").grid(
            row=5, column=0, padx=18, pady=6, sticky="w"
        )
        self.deploy_config = ctk.CTkEntry(frame, placeholder_text="pcb_yolo/configs/deploy_config.yaml")
        self.deploy_config.grid(row=5, column=1, padx=18, pady=6, sticky="ew")
        ctk.CTkButton(
            frame, text="Browse", command=lambda: self._browse_file(self.deploy_config)
        ).grid(row=5, column=2, padx=8, pady=6, sticky="w")

        self.deploy_status = ctk.CTkLabel(frame, text="Status: Idle", text_color=TEXT_MUTED)
        self.deploy_status.grid(row=6, column=0, padx=18, pady=(6, 6), sticky="w")

        ctk.CTkButton(frame, text="Run Deploy", command=self.run_deploy).grid(
            row=6, column=1, padx=18, pady=12, sticky="w"
        )

    def _build_settings_tab(self):
        frame = self.tab_settings
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            frame, text="Settings", font=ctk.CTkFont(size=20, weight="bold")
        ).grid(row=0, column=0, padx=18, pady=(18, 4), sticky="w")

        note = (
            "This UI runs your existing CLI pipeline from pcb_yolo.\n"
            "Make sure dependencies are installed in the selected Python env.\n"
            "ONNX export requires: onnx, onnxslim, onnxruntime or onnxruntime-gpu."
        )
        ctk.CTkLabel(frame, text=note, justify="left").grid(
            row=1, column=0, padx=18, pady=8, sticky="w"
        )

    def _browse_file(self, entry):
        path = filedialog.askopenfilename()
        if path:
            entry.delete(0, "end")
            entry.insert(0, path)

    def _browse_dir(self, entry):
        path = filedialog.askdirectory()
        if path:
            entry.delete(0, "end")
            entry.insert(0, path)

    def _browse_input(self, entry):
        path = filedialog.askopenfilename()
        if path:
            entry.delete(0, "end")
            entry.insert(0, path)

    def _append_log(self, text):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _poll_logs(self):
        try:
            while True:
                line = self.log_q.get_nowait()
                self._append_log(line)
        except queue.Empty:
            pass
        self.after(100, self._poll_logs)

    def _run_command(self, cmd, label, on_done=None):
        if self.proc and self.proc.poll() is None:
            self._append_log("Another process is running. Stop it first.\n")
            return

        python_exe = self.python_entry.get().strip() or find_python_exe(self.repo_root)
        full_cmd = [python_exe] + cmd
        self._append_log(f"\n$ {' '.join(full_cmd)}\n")
        self.status_label.configure(text=f"Running: {label}")
        self.active_step = label.lower()
        self._set_step_status(self.active_step, "Running")
        self.run_progress.configure(mode="indeterminate")
        self.run_progress.start()
        self.run_progress_label.configure(text=f"Step status: running {label}")
        self.last_exit_code = None

        def worker():
            env = os.environ.copy()
            pcb_path = os.path.join(self.repo_root, "pcb_yolo")
            env["PYTHONPATH"] = os.pathsep.join(
                [pcb_path, env.get("PYTHONPATH", "")]
            ).strip(os.pathsep)

            self.proc = subprocess.Popen(
                full_cmd,
                cwd=self.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            for line in self.proc.stdout:
                self.log_q.put(line)
            self.proc.wait()
            code = self.proc.returncode
            self.last_exit_code = code
            self.log_q.put(f"\n[exit code {code}]\n")
            self.status_label.configure(text="Idle")
            self.run_progress.stop()
            self.run_progress.set(0)
            if code == 0:
                self._set_step_status(self.active_step, "Success")
                self.completed_steps += 1
            else:
                self._set_step_status(self.active_step, "Failed")
            self._update_pipeline_progress()
            self.run_progress_label.configure(text="Step status: idle")
            if on_done:
                on_done(code)

        threading.Thread(target=worker, daemon=True).start()

    def _require_inputs(self, phase_name, required_pairs):
        missing = []
        for label, value in required_pairs:
            if not value:
                missing.append(label)

        if missing:
            self.log_q.put(
                f"\n[{phase_name} skipped: missing required input(s): {', '.join(missing)}]\n"
            )
            return False
        return True

    def stop_proc(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self._append_log("\n[process terminated]\n")
            if self.active_step:
                self._set_step_status(self.active_step, "Stopped")
            self.run_progress.stop()
            self.run_progress.set(0)
            self.run_progress_label.configure(text="Step status: idle")
        else:
            self._append_log("\n[no active process]\n")

    def run_prepare(self):
        cfg = abs_path(self.prepare_config.get().strip(), self.repo_root)
        if not self._require_inputs("prepare", [("config", cfg)]):
            return
        cmd = ["pcb_yolo/scripts/prepare_dataset.py", "--config", cfg]
        if self.prepare_download.get():
            cmd.append("--download")
        self._save_state()
        self._run_command(cmd, "prepare")

    def run_train(self):
        cfg = abs_path(self.train_config.get().strip(), self.repo_root)
        if not self._require_inputs("train", [("train config", cfg)]):
            return
        seed = self.train_seed.get().strip() or "42"
        cmd = ["-m", "src.train", "--config", cfg, "--seed", seed]
        if self.train_deterministic.get():
            cmd.append("--deterministic")
        self._save_state()

        def after_train(code):
            if code != 0:
                return
            if not self.auto_delete_after_train.get():
                return
            model_path = abs_path(self.model_path.get().strip(), self.repo_root)
            if not os.path.exists(model_path):
                self.log_q.put(
                    "\n[auto-delete skipped: model path not found]\n"
                )
                return
            self._delete_dataset_internal()

        self._run_command(cmd, "train", on_done=after_train)

    def run_export(self):
        model = abs_path(self.export_model.get().strip(), self.repo_root)
        if not self._require_inputs("export", [("model path", model)]):
            return
        fmt = self.export_format.get().strip() or "onnx"
        cmd = ["-m", "src.export", "--model", model, "--format", fmt]
        self._save_state()
        self._run_command(cmd, "export")

    def run_deploy(self):
        model = abs_path(self.deploy_model.get().strip(), self.repo_root)
        input_path = abs_path(self.deploy_input.get().strip(), self.repo_root)
        output_dir = abs_path(self.deploy_output.get().strip(), self.repo_root)
        cfg = abs_path(self.deploy_config.get().strip(), self.repo_root)
        if not self._require_inputs(
            "deploy",
            [
                ("model path", model),
                ("input", input_path),
                ("output dir", output_dir),
                ("deploy config", cfg),
            ],
        ):
            return
        cmd = [
            "-m",
            "src.inference.detector",
            "--model",
            model,
            "--input",
            input_path,
            "--output",
            output_dir,
            "--config",
            cfg,
        ]
        self._save_state()
        self._run_command(cmd, "deploy")

    def run_full_pipeline(self):
        steps = []
        if self.chk_prepare.get():
            steps.append(self.run_prepare)
        if self.chk_train.get():
            steps.append(self.run_train)
        if self.chk_export.get():
            steps.append(self.run_export)
        if self.chk_deploy.get():
            steps.append(self.run_deploy)
        if not steps:
            self.log_q.put("\n[no steps selected]\n")
            return

        self.active_steps = [fn.__name__.replace("run_", "") for fn in steps]
        self.completed_steps = 0
        self._update_pipeline_progress()

        def sequence():
            for step in steps:
                self.after(0, step)
                while self.proc and self.proc.poll() is None:
                    time.sleep(0.2)
                if self.last_exit_code not in (0, None):
                    self.log_q.put("\n[pipeline stopped: step failed]\n")
                    return
            self.log_q.put("\n[Pipeline finished]\n")

        threading.Thread(target=sequence, daemon=True).start()

    def delete_dataset(self):
        self._save_state()
        self._delete_dataset_internal()

    def _delete_dataset_internal(self):
        dataset_dir = abs_path(self.dataset_dir.get().strip(), self.repo_root)
        if not dataset_dir or not os.path.exists(dataset_dir):
            self.log_q.put("\n[dataset delete skipped: path not found]\n")
            return
        try:
            shutil.rmtree(dataset_dir)
            self.log_q.put(f"\n[dataset deleted: {dataset_dir}]\n")
        except Exception as exc:
            self.log_q.put(f"\n[dataset delete failed: {exc}]\n")

    def _load_state(self):
        defaults = {
            "python": find_python_exe(self.repo_root),
            "prepare_config": "pcb_yolo/configs/data_config.yaml",
            "prepare_download": True,
            "dataset_dir": "Detecting-the-PCB-object-3",
            "train_config": "pcb_yolo/configs/train_config.yaml",
            "train_seed": "42",
            "train_deterministic": True,
            "model_path": "pcb_yolo/experiments/pcb_train/weights/best.pt",
            "export_model": "pcb_yolo/experiments/pcb_train/weights/best.pt",
            "export_format": "onnx",
            "deploy_model": "pcb_yolo/experiments/pcb_train/weights/best.pt",
            "deploy_input": "",
            "deploy_output": "pcb_yolo/experiments/predictions",
            "deploy_config": "pcb_yolo/configs/deploy_config.yaml",
            "auto_delete_after_train": False,
        }
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    defaults.update(json.load(f))
            except Exception:
                pass

        self.python_entry.insert(0, defaults["python"])
        self.prepare_config.insert(0, defaults["prepare_config"])
        if defaults["prepare_download"]:
            self.prepare_download.select()
        self.dataset_dir.insert(0, defaults["dataset_dir"])
        self.train_config.insert(0, defaults["train_config"])
        self.train_seed.insert(0, defaults["train_seed"])
        if defaults["train_deterministic"]:
            self.train_deterministic.select()
        self.model_path.insert(0, defaults["model_path"])
        self.export_model.insert(0, defaults["export_model"])
        self.export_format.set(defaults["export_format"])
        self.deploy_model.insert(0, defaults["deploy_model"])
        self.deploy_input.insert(0, defaults["deploy_input"])
        self.deploy_output.insert(0, defaults["deploy_output"])
        self.deploy_config.insert(0, defaults["deploy_config"])
        if defaults["auto_delete_after_train"]:
            self.auto_delete_after_train.select()

        if ctk.get_appearance_mode().lower() == "light":
            self.theme_toggle.select()

        self._update_pipeline_progress()

    def _save_state(self):
        state = {
            "python": self.python_entry.get().strip(),
            "prepare_config": self.prepare_config.get().strip(),
            "prepare_download": bool(self.prepare_download.get()),
            "dataset_dir": self.dataset_dir.get().strip(),
            "train_config": self.train_config.get().strip(),
            "train_seed": self.train_seed.get().strip(),
            "train_deterministic": bool(self.train_deterministic.get()),
            "model_path": self.model_path.get().strip(),
            "export_model": self.export_model.get().strip(),
            "export_format": self.export_format.get().strip(),
            "deploy_model": self.deploy_model.get().strip(),
            "deploy_input": self.deploy_input.get().strip(),
            "deploy_output": self.deploy_output.get().strip(),
            "deploy_config": self.deploy_config.get().strip(),
            "auto_delete_after_train": bool(self.auto_delete_after_train.get()),
        }
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass

    def toggle_theme(self):
        is_light = bool(self.theme_toggle.get())
        ctk.set_appearance_mode("light" if is_light else "dark")

    def _set_step_status(self, step_key, status):
        if not step_key:
            return
        label = self.step_status_labels.get(step_key)
        if label:
            label.configure(text=status)
        if step_key == "prepare":
            self.prepare_status.configure(text=f"Status: {status}")
        elif step_key == "train":
            self.train_status.configure(text=f"Status: {status}")
        elif step_key == "export":
            self.export_status.configure(text=f"Status: {status}")
        elif step_key == "deploy":
            self.deploy_status.configure(text=f"Status: {status}")

    def _update_pipeline_progress(self):
        total = len(self.active_steps) if self.active_steps else 0
        self.pipeline_progress_label.configure(text=f"Pipeline progress: {self.completed_steps}/{total}")
        if total == 0:
            self.pipeline_progress.set(0)
        else:
            self.pipeline_progress.set(min(1, self.completed_steps / total))

    def open_setup_wizard(self):
        wizard = ctk.CTkToplevel(self)
        wizard.title("Guided Setup")
        wizard.geometry("700x520")
        wizard.grab_set()
        wizard.grid_columnconfigure(0, weight=1)
        wizard.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(wizard, fg_color=CARD_BG, corner_radius=12)
        header.grid(row=0, column=0, padx=16, pady=16, sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header, text="Guided Setup", font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, padx=14, pady=(10, 2), sticky="w")
        ctk.CTkLabel(
            header,
            text="Follow the steps and click Finish to apply settings.",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        ).grid(row=1, column=0, padx=14, pady=(0, 10), sticky="w")

        container = ctk.CTkFrame(wizard, corner_radius=12)
        container.grid(row=1, column=0, padx=16, pady=(0, 16), sticky="nsew")
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(0, weight=1)

        steps = []

        def add_step(title):
            step = ctk.CTkFrame(container, corner_radius=12)
            step.grid(row=0, column=0, sticky="nsew")
            step.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(step, text=title, font=ctk.CTkFont(size=16, weight="bold")).grid(
                row=0, column=0, columnspan=2, padx=16, pady=(16, 10), sticky="w"
            )
            steps.append(step)
            return step

        step1 = add_step("Step 1: Python Environment")
        ctk.CTkLabel(step1, text="Python path").grid(row=1, column=0, padx=16, pady=6, sticky="w")
        wizard_python = ctk.CTkEntry(step1)
        wizard_python.grid(row=1, column=1, padx=16, pady=6, sticky="ew")
        wizard_python.insert(0, self.python_entry.get().strip())
        ctk.CTkButton(step1, text="Auto-detect", command=lambda: wizard_python.delete(0, "end") or wizard_python.insert(0, find_python_exe(self.repo_root))).grid(
            row=2, column=1, padx=16, pady=6, sticky="w"
        )

        step2 = add_step("Step 2: Prepare Dataset")
        ctk.CTkLabel(step2, text="Config").grid(row=1, column=0, padx=16, pady=6, sticky="w")
        wizard_prepare_config = ctk.CTkEntry(step2)
        wizard_prepare_config.grid(row=1, column=1, padx=16, pady=6, sticky="ew")
        wizard_prepare_config.insert(0, self.prepare_config.get().strip())
        ctk.CTkButton(step2, text="Browse", command=lambda: self._browse_file(wizard_prepare_config)).grid(
            row=1, column=2, padx=8, pady=6, sticky="w"
        )
        wizard_prepare_download = ctk.CTkCheckBox(step2, text="Download from Roboflow")
        wizard_prepare_download.grid(row=2, column=1, padx=16, pady=6, sticky="w")
        if self.prepare_download.get():
            wizard_prepare_download.select()
        ctk.CTkLabel(step2, text="Dataset dir").grid(row=3, column=0, padx=16, pady=6, sticky="w")
        wizard_dataset_dir = ctk.CTkEntry(step2)
        wizard_dataset_dir.grid(row=3, column=1, padx=16, pady=6, sticky="ew")
        wizard_dataset_dir.insert(0, self.dataset_dir.get().strip())
        ctk.CTkButton(step2, text="Browse", command=lambda: self._browse_dir(wizard_dataset_dir)).grid(
            row=3, column=2, padx=8, pady=6, sticky="w"
        )

        step3 = add_step("Step 3: Train and Export")
        ctk.CTkLabel(step3, text="Train config").grid(row=1, column=0, padx=16, pady=6, sticky="w")
        wizard_train_config = ctk.CTkEntry(step3)
        wizard_train_config.grid(row=1, column=1, padx=16, pady=6, sticky="ew")
        wizard_train_config.insert(0, self.train_config.get().strip())
        ctk.CTkButton(step3, text="Browse", command=lambda: self._browse_file(wizard_train_config)).grid(
            row=1, column=2, padx=8, pady=6, sticky="w"
        )
        ctk.CTkLabel(step3, text="Seed").grid(row=2, column=0, padx=16, pady=6, sticky="w")
        wizard_train_seed = ctk.CTkEntry(step3)
        wizard_train_seed.grid(row=2, column=1, padx=16, pady=6, sticky="w")
        wizard_train_seed.insert(0, self.train_seed.get().strip())
        wizard_train_det = ctk.CTkCheckBox(step3, text="Deterministic")
        wizard_train_det.grid(row=3, column=1, padx=16, pady=6, sticky="w")
        if self.train_deterministic.get():
            wizard_train_det.select()
        ctk.CTkLabel(step3, text="Model path").grid(row=4, column=0, padx=16, pady=6, sticky="w")
        wizard_model_path = ctk.CTkEntry(step3)
        wizard_model_path.grid(row=4, column=1, padx=16, pady=6, sticky="ew")
        wizard_model_path.insert(0, self.model_path.get().strip())
        ctk.CTkButton(step3, text="Browse", command=lambda: self._browse_file(wizard_model_path)).grid(
            row=4, column=2, padx=8, pady=6, sticky="w"
        )

        step4 = add_step("Step 4: Deploy / Inference")
        ctk.CTkLabel(step4, text="Model path").grid(row=1, column=0, padx=16, pady=6, sticky="w")
        wizard_deploy_model = ctk.CTkEntry(step4)
        wizard_deploy_model.grid(row=1, column=1, padx=16, pady=6, sticky="ew")
        wizard_deploy_model.insert(0, self.deploy_model.get().strip())
        ctk.CTkButton(step4, text="Browse", command=lambda: self._browse_file(wizard_deploy_model)).grid(
            row=1, column=2, padx=8, pady=6, sticky="w"
        )
        ctk.CTkLabel(step4, text="Input").grid(row=2, column=0, padx=16, pady=6, sticky="w")
        wizard_deploy_input = ctk.CTkEntry(step4)
        wizard_deploy_input.grid(row=2, column=1, padx=16, pady=6, sticky="ew")
        wizard_deploy_input.insert(0, self.deploy_input.get().strip())
        ctk.CTkButton(step4, text="Browse", command=lambda: self._browse_input(wizard_deploy_input)).grid(
            row=2, column=2, padx=8, pady=6, sticky="w"
        )
        ctk.CTkLabel(step4, text="Output dir").grid(row=3, column=0, padx=16, pady=6, sticky="w")
        wizard_deploy_output = ctk.CTkEntry(step4)
        wizard_deploy_output.grid(row=3, column=1, padx=16, pady=6, sticky="ew")
        wizard_deploy_output.insert(0, self.deploy_output.get().strip())
        ctk.CTkButton(step4, text="Browse", command=lambda: self._browse_dir(wizard_deploy_output)).grid(
            row=3, column=2, padx=8, pady=6, sticky="w"
        )
        ctk.CTkLabel(step4, text="Deploy config").grid(row=4, column=0, padx=16, pady=6, sticky="w")
        wizard_deploy_config = ctk.CTkEntry(step4)
        wizard_deploy_config.grid(row=4, column=1, padx=16, pady=6, sticky="ew")
        wizard_deploy_config.insert(0, self.deploy_config.get().strip())
        ctk.CTkButton(step4, text="Browse", command=lambda: self._browse_file(wizard_deploy_config)).grid(
            row=4, column=2, padx=8, pady=6, sticky="w"
        )

        nav = ctk.CTkFrame(wizard)
        nav.grid(row=2, column=0, padx=16, pady=(0, 16), sticky="ew")
        nav.grid_columnconfigure(1, weight=1)
        step_index = {"value": 0}

        def show_step(index):
            step_index["value"] = index
            for i, s in enumerate(steps):
                s.grid_remove()
                if i == index:
                    s.grid()
            back_btn.configure(state="normal" if index > 0 else "disabled")
            next_btn.configure(state="normal" if index < len(steps) - 1 else "disabled")

        def apply_settings():
            self.python_entry.delete(0, "end")
            self.python_entry.insert(0, wizard_python.get().strip())
            self.prepare_config.delete(0, "end")
            self.prepare_config.insert(0, wizard_prepare_config.get().strip())
            self.prepare_download.deselect()
            if wizard_prepare_download.get():
                self.prepare_download.select()
            self.dataset_dir.delete(0, "end")
            self.dataset_dir.insert(0, wizard_dataset_dir.get().strip())
            self.train_config.delete(0, "end")
            self.train_config.insert(0, wizard_train_config.get().strip())
            self.train_seed.delete(0, "end")
            self.train_seed.insert(0, wizard_train_seed.get().strip())
            self.train_deterministic.deselect()
            if wizard_train_det.get():
                self.train_deterministic.select()
            self.model_path.delete(0, "end")
            self.model_path.insert(0, wizard_model_path.get().strip())
            self.deploy_model.delete(0, "end")
            self.deploy_model.insert(0, wizard_deploy_model.get().strip())
            self.deploy_input.delete(0, "end")
            self.deploy_input.insert(0, wizard_deploy_input.get().strip())
            self.deploy_output.delete(0, "end")
            self.deploy_output.insert(0, wizard_deploy_output.get().strip())
            self.deploy_config.delete(0, "end")
            self.deploy_config.insert(0, wizard_deploy_config.get().strip())
            self._save_state()
            wizard.destroy()

        back_btn = ctk.CTkButton(nav, text="Back", command=lambda: show_step(step_index["value"] - 1))
        back_btn.grid(row=0, column=0, padx=6, pady=6, sticky="w")
        next_btn = ctk.CTkButton(nav, text="Next", command=lambda: show_step(step_index["value"] + 1))
        next_btn.grid(row=0, column=1, padx=6, pady=6, sticky="e")
        finish_btn = ctk.CTkButton(nav, text="Finish", fg_color=ACCENT, hover_color=ACCENT_DARK, command=apply_settings)
        finish_btn.grid(row=0, column=2, padx=6, pady=6, sticky="e")

        show_step(0)

    def _on_close(self):
        self._save_state()
        self.destroy()


if __name__ == "__main__":
    app = PipelineUI()
    app.mainloop()
