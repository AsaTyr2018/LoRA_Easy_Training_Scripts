"""Gradio based UI for LoRA Easy Training Scripts."""

import contextlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from threading import Thread
from time import sleep

import requests
import toml
import gradio as gr


training_thread: Thread | None = None

# Default args mirroring the PySide6 UI
DEFAULT_ARGS = {
    "general": {
        "args": {
            "mixed_precision": "fp16",
            "seed": 23,
            "clip_skip": 2,
            "max_train_epochs": 1,
            "max_data_loader_n_workers": 1,
            "persistent_data_loader_workers": True,
            "max_token_length": 225,
            "prior_loss_weight": 1.0,
        },
        "dataset_args": {"resolution": 512, "batch_size": 1},
    },
    "network": {
        "args": {
            "network_dim": 32,
            "network_alpha": 16.0,
            "min_timestep": 0,
            "max_timestep": 1000,
        }
    },
    "optimizer": {
        "args": {
            "optimizer_type": "AdamW",
            "lr_scheduler": "cosine",
            "learning_rate": 1e-4,
            "max_grad_norm": 1.0,
            "loss_type": "l2",
        }
    },
    "saving": {
        "args": {"save_precision": "fp16", "save_model_as": "safetensors"}
    },
    "bucket": {
        "dataset_args": {
            "enable_bucket": True,
            "min_bucket_reso": 256,
            "max_bucket_reso": 1024,
            "bucket_reso_steps": 64,
        }
    },
    "noise_offset": {"args": {"noise_offset": 0.1}},
    "sample": {
        "args": {
            "sample_sampler": "ddim",
            "sample_every_n_steps": 1,
            "sample_prompts": "",
        }
    },
    "logging": {
        "args": {
            "log_with": "tensorboard",
            "logging_dir": "",
            "log_prefix": "",
            "log_tracker_name": "",
            "wandb_api_key": "",
        }
    },
    "flux": {"args": {}},
    "extra": {},
}


def build_args(
    general: str,
    network: str,
    optimizer: str,
    saving: str,
    bucket: str,
    noise_offset: str,
    sample: str,
    logging: str,
    flux: str,
    extra: str,
    subsets,
) -> tuple[dict, str | None]:
    """Combine JSON strings from the UI into a final args dictionary."""
    args = {}
    dataset = {"subsets": []}

    sections = {
        "general_args": general,
        "network_args": network,
        "optimizer_args": optimizer,
        "saving_args": saving,
        "bucket_args": bucket,
        "noise_offset_args": noise_offset,
        "sample_args": sample,
        "logging_args": logging,
        "flux_args": flux,
        "extra_args": extra,
    }

    for name, data in sections.items():
        if not data:
            continue
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as e:
            return {}, f"Invalid JSON in {name}: {e}"
        if not isinstance(parsed, dict):
            return {}, f"{name} must be a JSON object"
        # allow nested args/dataset_args to mimic original behaviour
        if "args" in parsed or "dataset_args" in parsed:
            if parsed.get("args"):
                args[name] = parsed["args"]
            if parsed.get("dataset_args"):
                dataset[name] = parsed["dataset_args"]
        else:
            args[name] = parsed

    if subsets:
        for row in subsets:
            if row[0]:
                dataset["subsets"].append(
                    {"image_dir": row[0], "num_repeats": int(row[1] or 1)}
                )

    return {"args": args, "dataset": dataset}, None


def save_config(
    general,
    network,
    optimizer,
    saving,
    bucket,
    noise_offset,
    sample,
    logging,
    flux,
    extra,
    subsets,
):
    """Return a TOML file of the current config for download."""
    config, err = build_args(
        general,
        network,
        optimizer,
        saving,
        bucket,
        noise_offset,
        sample,
        logging,
        flux,
        extra,
        subsets,
    )
    if err:
        return None, err
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".toml")
    tmp.write(toml.dumps(config).encode("utf-8"))
    tmp.close()
    return tmp.name, "Config saved"


def load_config(file):
    """Load toml config and return values for the UI."""
    if file is None:
        return ["" for _ in range(10)] + [[]] + ["No file provided"]
    path = Path(file.name if hasattr(file, "name") else file)
    args, dataset = process_toml(path)
    def _d(name):
        return json.dumps(args.get(name, {}), indent=2) if name in args else ""

    subsets = []
    for sub in dataset.get("subsets", []):
        subsets.append([sub.get("image_dir", ""), sub.get("num_repeats", 1)])

    return [
        _d("general_args"),
        _d("network_args"),
        _d("optimizer_args"),
        _d("saving_args"),
        _d("bucket_args"),
        _d("noise_offset_args"),
        _d("sample_args"),
        _d("logging_args"),
        _d("flux_args"),
        _d("extra_args"),
        subsets,
        "Config loaded",
    ]


def add_to_queue(
    general,
    network,
    optimizer,
    saving,
    bucket,
    noise_offset,
    sample,
    logging,
    flux,
    extra,
    subsets,
    queue_state,
):
    """Save current config to queue and update state."""
    config, err = build_args(
        general,
        network,
        optimizer,
        saving,
        bucket,
        noise_offset,
        sample,
        logging,
        flux,
        extra,
        subsets,
    )
    if err:
        return queue_state, None, err
    queue_dir = Path("queue_store")
    queue_dir.mkdir(exist_ok=True)
    ts = int(time.time() * 1000)
    file = queue_dir / f"{ts}.toml"
    file.write_text(toml.dumps(config))
    name = config["args"].get("saving_args", {}).get("output_name", str(ts))
    queue_state.append({"name": name, "file": file.as_posix()})
    return queue_state, [q["name"] for q in queue_state], "Added to queue"


def remove_from_queue(queue_state, selected):
    """Remove selected item from queue."""
    if not selected:
        return queue_state, [q["name"] for q in queue_state], "No selection"
    for i, item in enumerate(queue_state):
        if item["name"] == selected:
            path = Path(item["file"])
            if path.exists():
                path.unlink()
            queue_state.pop(i)
            break
    return queue_state, [q["name"] for q in queue_state], "Removed"


def load_queue_item(queue_state, selected):
    """Load selected queue item into the UI."""
    for item in queue_state:
        if item["name"] == selected:
            path = Path(item["file"])
            if not path.exists():
                return ["" for _ in range(10)] + [[]] + ["File missing"]
            return load_config(path)
    return ["" for _ in range(10)] + [[]] + ["Item not found"]




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


def start_training_thread(url: str, config_paths: list[Path]) -> None:
    """Thread target to start backend training."""
    for cfg in config_paths:
        success = train_helper(url, cfg)
        if not success:
            break


def start_training(
    general,
    network,
    optimizer,
    saving,
    bucket,
    noise_offset,
    sample,
    logging,
    flux,
    extra,
    subsets,
    queue_state,
    backend_url,
):
    """Entry point called by the gradio button."""
    global training_thread
    if training_thread and training_thread.is_alive():
        with contextlib.suppress(Exception):
            requests.get(f"{backend_url}/stop_training")
        return "Stopping training"

    config, err = build_args(
        general,
        network,
        optimizer,
        saving,
        bucket,
        noise_offset,
        sample,
        logging,
        flux,
        extra,
        subsets,
    )
    if err:
        return err

    config_dir = Path("queue_store")
    config_dir.mkdir(exist_ok=True)

    if queue_state:
        config_paths = [Path(item["file"]) for item in queue_state]
    else:
        tmp = config_dir / "temp.toml"
        tmp.write_text(toml.dumps(config))
        config_paths = [tmp]

    training_thread = Thread(
        target=start_training_thread, args=(backend_url, config_paths), daemon=True
    )
    training_thread.start()
    return "Training started"

# Basic layout mimicking the PySide6 GUI
with gr.Blocks(title="LoRA Trainer") as demo:
    queue_state = gr.State([])
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Tabs() as tabs:
                with gr.Tab("Main Args"):
                    with gr.Accordion("General Args", open=True):
                        general = gr.Textbox(
                            label="General settings",
                            value=json.dumps(DEFAULT_ARGS["general"], indent=2),
                        )
                    with gr.Accordion("Network Args"):
                        network = gr.Textbox(
                            label="Network settings",
                            value=json.dumps(DEFAULT_ARGS["network"], indent=2),
                        )
                    with gr.Accordion("Optimizer Args"):
                        optimizer = gr.Textbox(
                            label="Optimizer settings",
                            value=json.dumps(DEFAULT_ARGS["optimizer"], indent=2),
                        )
                    with gr.Accordion("Saving Args"):
                        saving = gr.Textbox(
                            label="Saving settings",
                            value=json.dumps(DEFAULT_ARGS["saving"], indent=2),
                        )
                    with gr.Accordion("Bucket Args"):
                        bucket = gr.Textbox(
                            label="Bucket settings",
                            value=json.dumps(DEFAULT_ARGS["bucket"], indent=2),
                        )
                    with gr.Accordion("Noise Offset Args"):
                        noise_offset = gr.Textbox(
                            label="Noise Offset settings",
                            value=json.dumps(DEFAULT_ARGS["noise_offset"], indent=2),
                        )
                    with gr.Accordion("Sample Args"):
                        sample = gr.Textbox(
                            label="Sample settings",
                            value=json.dumps(DEFAULT_ARGS["sample"], indent=2),
                        )
                    with gr.Accordion("Logging Args"):
                        logging = gr.Textbox(
                            label="Logging settings",
                            value=json.dumps(DEFAULT_ARGS["logging"], indent=2),
                        )
                    with gr.Accordion("Flux Args"):
                        flux = gr.Textbox(
                            label="Flux settings",
                            value=json.dumps(DEFAULT_ARGS["flux"], indent=2),
                        )
                    with gr.Accordion("Extra Args"):
                        extra = gr.Textbox(
                            label="Extra settings",
                            value=json.dumps(DEFAULT_ARGS["extra"], indent=2),
                        )
                with gr.Tab("Subset Args"):
                    subsets = gr.Dataframe(headers=["Subset Path", "Repeats"], datatype=["str", "number"], label="Subsets")
        with gr.Column(scale=1):
            load_file = gr.File(label="Load Config")
            load_btn = gr.Button("Load")
            save_btn = gr.Button("Save Config")
            download_file = gr.File()
            queue_radio = gr.Radio(choices=[], label="Queue")
            add_queue_btn = gr.Button("Add to Queue")
            remove_queue_btn = gr.Button("Remove From Queue")
            load_queue_btn = gr.Button("Load Selected")
            backend_url = gr.Textbox(value="http://127.0.0.1:8000", label="Backend Server URL")
            start_btn = gr.Button("Start Training")
            output = gr.Textbox(label="Output", interactive=False)

    load_btn.click(
        load_config,
        inputs=load_file,
        outputs=[
            general,
            network,
            optimizer,
            saving,
            bucket,
            noise_offset,
            sample,
            logging,
            flux,
            extra,
            subsets,
            output,
        ],
    )
    save_btn.click(
        save_config,
        inputs=[
            general,
            network,
            optimizer,
            saving,
            bucket,
            noise_offset,
            sample,
            logging,
            flux,
            extra,
            subsets,
        ],
        outputs=[download_file, output],
    )
    add_queue_btn.click(
        add_to_queue,
        inputs=[
            general,
            network,
            optimizer,
            saving,
            bucket,
            noise_offset,
            sample,
            logging,
            flux,
            extra,
            subsets,
            queue_state,
        ],
        outputs=[queue_state, queue_radio, output],
    )
    remove_queue_btn.click(
        remove_from_queue,
        inputs=[queue_state, queue_radio],
        outputs=[queue_state, queue_radio, output],
    )
    load_queue_btn.click(
        load_queue_item,
        inputs=[queue_state, queue_radio],
        outputs=[
            general,
            network,
            optimizer,
            saving,
            bucket,
            noise_offset,
            sample,
            logging,
            flux,
            extra,
            subsets,
            output,
        ],
    )
    start_btn.click(
        start_training,
        inputs=[
            general,
            network,
            optimizer,
            saving,
            bucket,
            noise_offset,
            sample,
            logging,
            flux,
            extra,
            subsets,
            queue_state,
            backend_url,
        ],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
