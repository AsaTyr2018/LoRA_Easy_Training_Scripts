"""Gradio based UI for LoRA Easy Training Scripts."""

import contextlib
import json
import os
import shutil
from pathlib import Path
from threading import Thread
from time import sleep

import requests
import toml
import gradio as gr


training_thread: Thread | None = None


def perform_name_replace(args: dict) -> dict:
    """Replace "${replace}" with easy_name in various args."""
    template_str = r"${replace}"
    saving_args = args.get("saving_args", {})
    replace_str = saving_args.get("easy_name", "")

    if "output_dir" in saving_args:
        saving_args["output_dir"] = saving_args["output_dir"].replace(
            template_str, replace_str
        )
    if "output_name" in saving_args:
        saving_args["output_name"] = saving_args["output_name"].replace(
            template_str, replace_str
        )
    if "tag_file_location" in saving_args:
        saving_args["tag_file_location"] = saving_args["tag_file_location"].replace(
            template_str, replace_str
        )
    if "save_toml_location" in saving_args:
        saving_args["save_toml_location"] = saving_args["save_toml_location"].replace(
            template_str, replace_str
        )
    if "resume" in saving_args:
        saving_args["resume"] = saving_args["resume"].replace(template_str, replace_str)

    logging_args = args.get("logging_args", {})
    if "log_prefix" in logging_args:
        logging_args["log_prefix"] = logging_args["log_prefix"].replace(
            template_str, replace_str
        )

    return args


def create_tag_file(tags: dict, output_location: Path | None = None, output_name: str = "output_tags") -> None:
    """Create a tag file from backend validation data."""
    if not tags:
        return
    if not output_location:
        output_location = Path("auto_save_store")
    if not output_location.exists():
        output_location.mkdir()
    if output_location.is_file():
        output_location = output_location.parent
    output_location = output_location.joinpath(f"{output_name}.txt")
    with output_location.open("w", encoding="utf-8") as f:
        f.write("Below is a list of keywords used during the training of this model:\n")
        for k, v in tags.items():
            f.write(f"[{v}] {k}\n")


def create_auto_save_toml(input_toml: Path, output_location: Path | None = None, output_name: str = "output_toml") -> None:
    """Save a copy of the training toml used by the backend."""
    if not output_location:
        output_location = Path("auto_save_store")
    if not output_location.exists():
        output_location.mkdir()
    if output_location.is_file():
        output_location = output_location.parent
    output_location = output_location.joinpath(f"{output_name}.toml")
    offset = 1
    orig_name = output_location.stem
    while output_location.exists():
        output_location = output_location.with_stem(f"{orig_name}_{offset}")
        offset += 1
    shutil.copy(input_toml, output_location)


def process_toml(file_name: Path) -> tuple[dict, dict]:
    """Load toml and return args and dataset."""
    if not file_name.exists():
        return {}, {}
    loaded_args = toml.loads(file_name.read_text())
    args = {}
    dataset_args = {}
    if "subsets" in loaded_args:
        dataset_args["subsets"] = loaded_args["subsets"]
        del loaded_args["subsets"]
    for arg, val in loaded_args.items():
        if "args" in val:
            args[arg] = val["args"]
        if "dataset_args" in val:
            dataset_args[arg] = val["dataset_args"]
    return args, dataset_args


def train_helper(url: str, train_toml: Path) -> bool:
    """Helper that validates and starts training via backend HTTP API."""
    args, dataset_args = process_toml(train_toml)
    final_args = {"args": perform_name_replace(args), "dataset": dataset_args}
    try:
        response = requests.post(f"{url}/validate", json=True, data=json.dumps(final_args))
    except requests.ConnectionError as e:
        print(e)
        return False
    if response.status_code != 200:
        print(f"Item Failed: {response.text}")
        return False
    validation_data = response.json()
    if args.get("saving_args", {}).get("tag_occurrence"):
        folder = args["saving_args"].get("tag_file_location")
        create_tag_file(validation_data.get("tags", {}), Path(folder) if folder else None, args["saving_args"].get("output_name", "output_tags"))
    if args.get("saving_args", {}).get("save_toml"):
        folder = args["saving_args"].get("save_toml_location")
        create_auto_save_toml(train_toml, Path(folder) if folder else None, args["saving_args"].get("output_name", "output_args"))
    os.remove(train_toml)
    is_sdxl = str(args.get("general_args", {}).get("sdxl", False))
    is_flux = str(bool(args.get("flux_args")))
    requests.get(f"{url}/train", params={"train_mode": "lora", "sdxl": is_sdxl, "flux": is_flux})
    training = True
    while training:
        sleep(5.0)
        try:
            response = requests.get(f"{url}/is_training")
        except Exception:
            print("Connection Failed, assuming training has stopped.")
            return False
        if response.status_code != 200:
            print("Connection Failed, assuming training has stopped.")
            return False
        response = response.json()
        if not response.get("training"):
            training = False
        if response.get("errored"):
            return False
    return True


def start_training_thread(url: str, config_path: Path) -> None:
    """Thread target to start backend training."""
    train_helper(url, config_path)


def start_training(args_json: str, subsets, queue_df, backend_url):
    """Entry point called by the gradio button."""
    global training_thread
    if training_thread and training_thread.is_alive():
        with contextlib.suppress(Exception):
            requests.get(f"{backend_url}/stop_training")
        return "Stopping training"

    # Build args dictionary from user JSON input
    try:
        args = json.loads(args_json) if args_json else {}
    except json.JSONDecodeError as e:
        return f"Invalid args JSON: {e}"

    dataset_args = {"subsets": []}
    if subsets:
        for row in subsets:
            if row[0]:
                dataset_args["subsets"].append({"image_dir": row[0], "num_repeats": int(row[1] or 1)})

    final_args = {"args": args, "dataset": dataset_args}

    config_dir = Path("queue_store")
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir.joinpath("temp.toml")
    config_path.write_text(toml.dumps(final_args))

    training_thread = Thread(target=start_training_thread, args=(backend_url, config_path), daemon=True)
    training_thread.start()
    return "Training started"

# Basic layout mimicking the PySide6 GUI
with gr.Blocks(title="LoRA Trainer") as demo:
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Tabs() as tabs:
                with gr.Tab("Main Args"):
                    with gr.Accordion("General Args", open=True):
                        general = gr.Textbox(label="General settings", placeholder="...")
                    with gr.Accordion("Network Args"):
                        network = gr.Textbox(label="Network settings", placeholder="...")
                    with gr.Accordion("Optimizer Args"):
                        optimizer = gr.Textbox(label="Optimizer settings", placeholder="...")
                    with gr.Accordion("Saving Args"):
                        saving = gr.Textbox(label="Saving settings", placeholder="...")
                    with gr.Accordion("Bucket Args"):
                        bucket = gr.Textbox(label="Bucket settings", placeholder="...")
                    with gr.Accordion("Noise Offset Args"):
                        noise_offset = gr.Textbox(label="Noise Offset settings", placeholder="...")
                    with gr.Accordion("Sample Args"):
                        sample = gr.Textbox(label="Sample settings", placeholder="...")
                    with gr.Accordion("Logging Args"):
                        logging = gr.Textbox(label="Logging settings", placeholder="...")
                    with gr.Accordion("Flux Args"):
                        flux = gr.Textbox(label="Flux settings", placeholder="...")
                    with gr.Accordion("Extra Args"):
                        extra = gr.Textbox(label="Extra settings", placeholder="...")
                with gr.Tab("Subset Args"):
                    subsets = gr.Dataframe(headers=["Subset Path", "Repeats"], datatype=["str", "number"], label="Subsets")
        with gr.Column(scale=1):
            queue = gr.Dataframe(headers=["Queue Item"], datatype=["str"], label="Queue")
            backend_url = gr.Textbox(value="http://127.0.0.1:8000", label="Backend Server URL")
            start_btn = gr.Button("Start Training")
            output = gr.Textbox(label="Output", interactive=False)
    start_btn.click(start_training, inputs=[general, subsets, queue, backend_url], outputs=output)

if __name__ == "__main__":
    demo.launch()
