"""Restore one fitted normal-only model and score a private image."""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .models import Autoencoder, PatchMemory, image_scores
from .repro import write_json


def restore(checkpoint):
    if checkpoint["method"] == "autoencoder":
        model = Autoencoder().eval()
        model.load_state_dict(checkpoint["autoencoder_weights"])
        return model
    encoder = None
    if checkpoint["encoder_weights"] is not None:
        encoder = Autoencoder().eval()
        encoder.load_state_dict(checkpoint["encoder_weights"])
    model = PatchMemory(encoder=encoder)
    model.bank, model.mean, model.std = [checkpoint[key] for key in ("bank", "mean", "std")]
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--out", default="runs/inference")
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = restore(checkpoint)
    image = Image.open(args.image).convert("L").resize((checkpoint["size"], checkpoint["size"]), Image.Resampling.BILINEAR)
    x = torch.from_numpy(np.asarray(image, dtype=np.float32)[None, None] / 255)
    with torch.inference_mode():
        anomaly = model.score(x)
        score = float(image_scores(anomaly))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "anomaly_map.npy", anomaly[0, 0].numpy())
    Image.fromarray(((anomaly[0, 0].numpy() > checkpoint["pixel_threshold"]) * 255).astype("uint8")).save(out / "localization.png")
    write_json(out / "prediction.json", {"method": checkpoint["method"], "score": score,
               "image_threshold": checkpoint["image_threshold"], "alarm": score > checkpoint["image_threshold"]})


if __name__ == "__main__":
    main()
