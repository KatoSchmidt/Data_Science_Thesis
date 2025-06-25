import argparse
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange

from the_well.benchmark.metrics import VRMSE, MSE, NMSE, VMSE, LInfinity, binned_spectral_mse, PearsonR
from the_well.data import WellDataset
from the_well.data.normalization import ZScoreNormalization
from the_well.benchmark.models.unet_convnext import UNetConvNext
from the_well.benchmark.models.sinenet import SineNet
from the_well.benchmark.models.swinnet import SwinUnet


def compute_shared_metrics(fx, y, metadata):
    bsmse = binned_spectral_mse.eval(fx, y, meta=dataset.metadata)
    binned_metrics = {k: v.squeeze(1).mean(dim=0).tolist() for k, v in bsmse.items() if "per_bin" in k }
    return {
        "MSE": MSE.eval(fx, y, meta=metadata).squeeze(1).tolist()[0],
        "NMSE": NMSE.eval(fx, y, meta=metadata).squeeze(1).tolist()[0],
        "RMSE": VMSE.eval(fx, y, meta=metadata).squeeze(1).tolist()[0],
        "VMSE": VMSE.eval(fx, y, meta=metadata).squeeze(1).tolist()[0],
        "VRMSE": VRMSE.eval(fx, y, meta=metadata).squeeze(1).tolist()[0],
        "LInfinity": LInfinity.eval(fx, y, metadata).squeeze(1).tolist()[0],
        "PearsonR": PearsonR.eval(fx, y, metadata).squeeze(1).tolist()[0],
        **binned_metrics,
    }


def validate_onestep(model, dataset, loader, device, args):
    model.eval()
    log = []
    F = dataset.metadata.n_fields
    To = dataset.n_steps_output

    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader), total=len(loader), desc="Validating One-Step"):
            x = batch["input_fields"].to(device)
            y = batch["output_fields"].to(device)

            if "sinenet" in args.model:
                x = rearrange(x, "B Ti Lx Ly F -> B Ti F Lx Ly")
                y = rearrange(y, "B To Lx Ly F -> B To F Lx Ly")
                fx = model(x)
                fx = rearrange(fx, "B To F Lx Ly -> B To Lx Ly F", To=To, F=F)
                y = rearrange(y, "B To F Lx Ly -> B To Lx Ly F", To=To, F=F)
            else:
                x = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
                y = rearrange(y, "B To Lx Ly F -> B (To F) Lx Ly")
                fx = model(x)
                fx = rearrange(fx, "B (To F) Lx Ly -> B To Lx Ly F", To=To, F=F)
                y = rearrange(y, "B (To F) Lx Ly -> B To Lx Ly F", To=To, F=F)

            file_idx, sample_idx, time_idx, dt = dataset._load_one_sample(i)[-4:]
            current_timestep = time_idx + dataset.n_steps_input + 1
            current_t_cool = round(float(batch['constant_scalars'][0][0]), 3)

            metrics = {
                "t_cool": current_t_cool,
                "timestep": current_timestep, }
            metrics.update(compute_shared_metrics(fx, y, dataset.metadata))

            log.append(metrics)
    return log


def validate_rollout(model, dataset, loader, device, args):
    model.eval()
    log = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating Rollout"):
            x = batch["input_fields"].to(device)
            y = batch["output_fields"].to(device)

            context = x.clone()
            predictions = []

            for _ in range(dataset.max_rollout_steps):
                if "sinenet" in args.model:
                    x_in = rearrange(context, "B Ti H W F -> B Ti F H W")
                    pred = model(x_in)
                    pred = rearrange(pred, "B 1 F H W -> B 1 H W F")
                else:
                    x_in = rearrange(context, "B Ti H W F -> B (Ti F) H W")
                    pred = model(x_in)
                    pred = rearrange(pred, "B (1 F) H W -> B 1 H W F")

                predictions.append(pred)
                context = torch.cat([context[:, 1:], pred], dim=1)

            rollout_preds = torch.cat(predictions, dim=1)
            current_t_cool = round(float(batch['constant_scalars'][0][0]), 3)

            for t in range(min(rollout_preds.shape[1], y.shape[1])):
                pred_t = rollout_preds[:, t:t+1]
                true_t = y[:, t:t+1]
                current_timestep = t + dataset.n_steps_input + 1

                shared_metrics = compute_shared_metrics(pred_t, true_t, dataset.metadata)
                bsmse = binned_spectral_mse.eval(pred_t, true_t, meta=dataset.metadata)

                binned_metrics = {
                    k: v.squeeze(1).mean(dim=0).tolist()
                    for k, v in bsmse.items() if "per_bin" in k
                }

                log.append({
                    "t_cool": current_t_cool,
                    "timestep": current_timestep,
                    **shared_metrics,
                    **binned_metrics,
                })

    return log


def build_model(args, dataset):
    F = dataset.metadata.n_fields

    if args.model.startswith("sinenet"):
        num_waves = int(args.model.split("-")[1])
        return SineNet(
            dim_in=F * dataset.n_steps_input,
            dim_out=F * dataset.n_steps_output,
            n_spatial_dims=dataset.n_spatial_dims,
            spatial_resolution=dataset.metadata.spatial_resolution,
            n_input_scalar_components=2,
            n_input_vector_components=1,
            n_output_scalar_components=2,
            n_output_vector_components=1,
            time_history=4,
            time_future=1,
            hidden_channels=64,
            padding_mode="circular",
            activation="gelu",
            num_layers=4,
            num_waves=num_waves,
            norm=True,
            mult={4: 1.4, 8: 1.25, 12: 1.15}[num_waves],
        )

    elif args.model == "unet_convnext":
        return UNetConvNext(
            dim_in=F * dataset.n_steps_input,
            dim_out=F * dataset.n_steps_output,
            n_spatial_dims=dataset.n_spatial_dims,
            spatial_resolution=dataset.metadata.spatial_resolution,
            init_features=42,
            blocks_per_stage=2,
        )

    elif args.model.startswith("swinnet"):
        refinement = args.model == "swinnet_R"
        return SwinUnet(
            dim_in=F * dataset.n_steps_input,
            dim_out=F * dataset.n_steps_output,
            n_spatial_dims=dataset.n_spatial_dims,
            spatial_resolution=dataset.metadata.spatial_resolution,
            patch_size=4,
            embed_dim=64,
            num_heads=[4, 8, 16, 32],
            depths=[2, 2, 2],
            num_bottleneck_blocks=4,
            window_size=8,
            refinement=refinement,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--well_split_name", type=str, default="test")
    parser.add_argument("--data_workers", type=int, default=8)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="validation_json_files")
    parser.add_argument("--run", type=str, default="run1", required=True)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading datasets from {args.data_path}...")

    dataset = WellDataset(
        well_base_path=args.data_path,
        well_dataset_name="turbulent_radiative_layer_2D",
        well_split_name=args.well_split_name,
        n_steps_input=4,
        n_steps_output=1,
        use_normalization=True,
        normalization_type=ZScoreNormalization,
        min_dt_stride=1,
        max_dt_stride=1,
    )

    dataset_rollout = WellDataset(
        well_base_path=args.data_path,
        well_dataset_name="turbulent_radiative_layer_2D",
        well_split_name=args.well_split_name,
        n_steps_input=4,
        n_steps_output=1,
        use_normalization=True,
        normalization_type=ZScoreNormalization,
        min_dt_stride=1,
        max_dt_stride=1,
        full_trajectory_mode=True,
        max_rollout_steps=100,
    )

    onestep_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.data_workers)
    rollout_loader = torch.utils.data.DataLoader(dataset_rollout, batch_size=1, shuffle=False, num_workers=args.data_workers)

    model = build_model(args, dataset)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    print("Starting validation...")
    onestep_output = validate_onestep(model, dataset, onestep_loader, device, args)
    rollout_output = validate_rollout(model, dataset_rollout, rollout_loader, device, args)

    output = {"onestep": onestep_output, "rollout": rollout_output}
    output_path = os.path.join(args.output_dir, f"{args.model}_{args.run}.json")

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"✅ Validation complete. Results saved to: {output_path}")
