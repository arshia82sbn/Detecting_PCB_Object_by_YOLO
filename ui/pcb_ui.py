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
        self.geometry("1200x720")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.repo_root = os.path.dirname(self.root_dir)
        self.state_file = os.path.join(self.repo_root, ".pcb_ui_state.json")

        self.proc = None
        self.last_exit_code = None
        self.log_q = queue.Queue()

        self._build_layout()
        self._load_state()
        self._poll_logs()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        sidebar = ctk.CTkFrame(self, width=260)
        sidebar.grid(row=0, column=0, sticky="nsw")
        sidebar.grid_rowconfigure(10, weight=1)

        ctk.CTkLabel(
            sidebar, text="PCB YOLO", font=ctk.CTkFont(size=20, weight="bold")
        ).grid(row=0, column=0, padx=16, pady=(16, 6), sticky="w")
        ctk.CTkLabel(
            sidebar, text="Pipeline Launcher", font=ctk.CTkFont(size=12)
        ).grid(row=1, column=0, padx=16, pady=(0, 12), sticky="w")

        self.btn_run_all = ctk.CTkButton(
            sidebar, text="Run Full Pipeline", command=self.run_full_pipeline
        )
        self.btn_run_all.grid(row=2, column=0, padx=16, pady=(0, 8), sticky="ew")

        self.btn_stop = ctk.CTkButton(
            sidebar, text="Stop Current", fg_color="#7a2e2e", command=self.stop_proc
        )
        self.btn_stop.grid(row=3, column=0, padx=16, pady=(0, 8), sticky="ew")

        ctk.CTkLabel(
            sidebar, text="Python", font=ctk.CTkFont(size=12, weight="bold")
        ).grid(row=4, column=0, padx=16, pady=(16, 4), sticky="w")
        self.python_entry = ctk.CTkEntry(sidebar)
        self.python_entry.grid(row=5, column=0, padx=16, pady=(0, 8), sticky="ew")

        ctk.CTkLabel(
            sidebar, text="Logs", font=ctk.CTkFont(size=12, weight="bold")
        ).grid(row=6, column=0, padx=16, pady=(16, 4), sticky="w")
        self.log_box = ctk.CTkTextbox(sidebar, height=160)
        self.log_box.grid(row=7, column=0, padx=16, pady=(0, 12), sticky="ew")
        self.log_box.configure(state="disabled")

        self.status_label = ctk.CTkLabel(
            sidebar, text="Idle", font=ctk.CTkFont(size=12)
        )
        self.status_label.grid(row=8, column=0, padx=16, pady=(8, 16), sticky="w")

        main = ctk.CTkTabview(self)
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
            frame, text="Full Pipeline", font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, padx=16, pady=(16, 8), sticky="w")

        self.chk_prepare = ctk.CTkCheckBox(frame, text="1) Prepare dataset")
        self.chk_prepare.select()
        self.chk_prepare.grid(row=1, column=0, padx=16, pady=6, sticky="w")

        self.chk_train = ctk.CTkCheckBox(frame, text="2) Train model")
        self.chk_train.select()
        self.chk_train.grid(row=2, column=0, padx=16, pady=6, sticky="w")

        self.chk_export = ctk.CTkCheckBox(frame, text="3) Export model")
        self.chk_export.select()
        self.chk_export.grid(row=3, column=0, padx=16, pady=6, sticky="w")

        self.chk_deploy = ctk.CTkCheckBox(frame, text="4) Deploy / Inference")
        self.chk_deploy.select()
        self.chk_deploy.grid(row=4, column=0, padx=16, pady=6, sticky="w")

        self.auto_delete_after_train = ctk.CTkCheckBox(
            frame,
            text="Delete dataset after successful training (model must exist)",
        )
        self.auto_delete_after_train.grid(
            row=5, column=0, padx=16, pady=6, sticky="w"
        )

        ctk.CTkButton(
            frame, text="Run Selected Steps", command=self.run_full_pipeline
        ).grid(row=6, column=0, padx=16, pady=(12, 8), sticky="w")

    def _build_prepare_tab(self):
        frame = self.tab_prepare
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="Prepare Dataset", font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, columnspan=2, padx=16, pady=(16, 8), sticky="w")

        ctk.CTkLabel(frame, text="Config").grid(
            row=1, column=0, padx=16, pady=6, sticky="w"
        )
        self.prepare_config = ctk.CTkEntry(frame)
        self.prepare_config.grid(row=1, column=1, padx=16, pady=6, sticky="ew")
        ctk.CTkButton(
            frame, text="Browse", command=lambda: self._browse_file(self.prepare_config)
        ).grid(row=1, column=2, padx=8, pady=6, sticky="w")

        self.prepare_download = ctk.CTkCheckBox(frame, text="Download from Roboflow")
        self.prepare_download.grid(row=2, column=1, padx=16, pady=6, sticky="w")

        ctk.CTkLabel(frame, text="Dataset dir").grid(
            row=3, column=0, padx=16, pady=6, sticky="w"
        )
        self.dataset_dir = ctk.CTkEntry(frame)
        self.dataset_dir.grid(row=3, column=1, padx=16, pady=6, sticky="ew")
        ctk.CTkButton(
            frame, text="Browse", command=lambda: self._browse_dir(self.dataset_dir)
        ).grid(row=3, column=2, padx=8, pady=6, sticky="w")

        ctk.CTkButton(frame, text="Run Prepare", command=self.run_prepare).grid(
            row=4, column=1, padx=16, pady=12, sticky="w"
        )

        ctk.CTkButton(
            frame, text="Delete Dataset", fg_color="#7a2e2e", command=self.delete_dataset
        ).grid(row=5, column=1, padx=16, pady=(0, 12), sticky="w")

    def _build_train_tab(self):
        frame = self.tab_train
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="Train Model", font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, columnspan=2, padx=16, pady=(16, 8), sticky="w")

        ctk.CTkLabel(frame, text="Train config").grid(
            row=1, column=0, padx=16, pady=6, sticky="w"
        )
        self.train_config = ctk.CTkEntry(frame)
        self.train_config.grid(row=1, column=1, padx=16, pady=6, sticky="ew")
        ctk.CTkButton(
            frame, text="Browse", command=lambda: self._browse_file(self.train_config)
        ).grid(row=1, column=2, padx=8, pady=6, sticky="w")

        ctk.CTkLabel(frame, text="Seed").grid(
            row=2, column=0, padx=16, pady=6, sticky="w"
        )
        self.train_seed = ctk.CTkEntry(frame, width=120)
        self.train_seed.grid(row=2, column=1, padx=16, pady=6, sticky="w")

        self.train_deterministic = ctk.CTkCheckBox(frame, text="Deterministic")
        self.train_deterministic.select()
        self.train_deterministic.grid(row=3, column=1, padx=16, pady=6, sticky="w")

        ctk.CTkLabel(frame, text="Expected model path").grid(
            row=4, column=0, padx=16, pady=6, sticky="w"
        )
        self.model_path = ctk.CTkEntry(frame)
        self.model_path.grid(row=4, column=1, padx=16, pady=6, sticky="ew")
        ctk.CTkButton(
            frame, text="Browse", command=lambda: self._browse_file(self.model_path)
        ).grid(row=4, column=2, padx=8, pady=6, sticky="w")

        ctk.CTkButton(frame, text="Run Train", command=self.run_train).grid(
            row=5, column=1, padx=16, pady=12, sticky="w"
        )

    def _build_export_tab(self):
        frame = self.tab_export
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="Export Model", font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, columnspan=2, padx=16, pady=(16, 8), sticky="w")

        ctk.CTkLabel(frame, text="Model path").grid(
            row=1, column=0, padx=16, pady=6, sticky="w"
        )
        self.export_model = ctk.CTkEntry(frame)
        self.export_model.grid(row=1, column=1, padx=16, pady=6, sticky="ew")
        ctk.CTkButton(
            frame, text="Browse", command=lambda: self._browse_file(self.export_model)
        ).grid(row=1, column=2, padx=8, pady=6, sticky="w")

        ctk.CTkLabel(frame, text="Format").grid(
            row=2, column=0, padx=16, pady=6, sticky="w"
        )
        self.export_format = ctk.CTkComboBox(frame, values=["onnx", "torchscript"])
        self.export_format.set("onnx")
        self.export_format.grid(row=2, column=1, padx=16, pady=6, sticky="w")

        ctk.CTkButton(frame, text="Run Export", command=self.run_export).grid(
            row=3, column=1, padx=16, pady=12, sticky="w"
        )

    def _build_deploy_tab(self):
        frame = self.tab_deploy
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="Deploy / Inference", font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, columnspan=2, padx=16, pady=(16, 8), sticky="w")

        ctk.CTkLabel(frame, text="Model path").grid(
            row=1, column=0, padx=16, pady=6, sticky="w"
        )
        self.deploy_model = ctk.CTkEntry(frame)
        self.deploy_model.grid(row=1, column=1, padx=16, pady=6, sticky="ew")
        ctk.CTkButton(
            frame, text="Browse", command=lambda: self._browse_file(self.deploy_model)
        ).grid(row=1, column=2, padx=8, pady=6, sticky="w")

        ctk.CTkLabel(frame, text="Input").grid(
            row=2, column=0, padx=16, pady=6, sticky="w"
        )
        self.deploy_input = ctk.CTkEntry(frame)
        self.deploy_input.grid(row=2, column=1, padx=16, pady=6, sticky="ew")
        ctk.CTkButton(
            frame, text="Browse", command=lambda: self._browse_input(self.deploy_input)
        ).grid(row=2, column=2, padx=8, pady=6, sticky="w")

        ctk.CTkLabel(frame, text="Output dir").grid(
            row=3, column=0, padx=16, pady=6, sticky="w"
        )
        self.deploy_output = ctk.CTkEntry(frame)
        self.deploy_output.grid(row=3, column=1, padx=16, pady=6, sticky="ew")
        ctk.CTkButton(
            frame, text="Browse", command=lambda: self._browse_dir(self.deploy_output)
        ).grid(row=3, column=2, padx=8, pady=6, sticky="w")

        ctk.CTkLabel(frame, text="Deploy config").grid(
            row=4, column=0, padx=16, pady=6, sticky="w"
        )
        self.deploy_config = ctk.CTkEntry(frame)
        self.deploy_config.grid(row=4, column=1, padx=16, pady=6, sticky="ew")
        ctk.CTkButton(
            frame, text="Browse", command=lambda: self._browse_file(self.deploy_config)
        ).grid(row=4, column=2, padx=8, pady=6, sticky="w")

        ctk.CTkButton(frame, text="Run Deploy", command=self.run_deploy).grid(
            row=5, column=1, padx=16, pady=12, sticky="w"
        )

    def _build_settings_tab(self):
        frame = self.tab_settings
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            frame, text="Settings", font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, padx=16, pady=(16, 8), sticky="w")

        note = (
            "This UI runs your existing CLI pipeline from pcb_yolo.\n"
            "Make sure dependencies are installed in the selected Python env.\n"
            "ONNX export requires: onnx, onnxslim, onnxruntime or onnxruntime-gpu."
        )
        ctk.CTkLabel(frame, text=note, justify="left").grid(
            row=1, column=0, padx=16, pady=8, sticky="w"
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

    def _on_close(self):
        self._save_state()
        self.destroy()


if __name__ == "__main__":
    app = PipelineUI()
    app.mainloop()
