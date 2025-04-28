import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import random
import numpy as np
from tqdm import tqdm
from einops import rearrange
# from transformers import AutoModel
import wandb
import os
from the_well.benchmark.metrics import VRMSE, MSE,  NMSE,  VMSE, LInfinity, binned_spectral_mse
from the_well.data import WellDataset, WellDataModule
from the_well.data.normalization import ZScoreNormalization
from the_well.benchmark.models.unet_classic import UNetClassic
from the_well.benchmark.models.unet_convnext import UNetConvNext
from the_well.benchmark.models.sinenet import SineNet
import h5py


def validate_onestep(model, dataset, onestep_loader,  device, args):
    model.eval()
    F = dataset.metadata.n_fields
    To = dataset.n_steps_output
    log = []

    with torch.no_grad():
        for i, batch in tqdm(enumerate(onestep_loader), desc="Testing One-Step",  total=len(onestep_loader)):
            x = batch["input_fields"].to(device)
            x = rearrange(x, "B Ti Lx Ly F -> B Ti F Lx Ly")

            y = batch["output_fields"].to(device)
            y = rearrange(y, "B To Lx Ly F -> B To F Lx Ly")

            fx = model(x)
        
            fx = rearrange(fx, "B To F Lx Ly -> B To Lx Ly F", To=To, F=F)
            y = rearrange(y, "B To F Lx Ly -> B To Lx Ly F", To=To, F=F)

            file_idx, sample_idx, time_idx, dt = dataset._load_one_sample(i)[-4:]

            current_timestep = time_idx + dataset.n_steps_input + 1
            current_t_cool = round(float(batch['constant_scalars'][0][0]), 3)

            metrics = {
                "t_cool": round(float(batch['constant_scalars'][0][0]),3), 
                "timestep": time_idx + dataset.n_steps_input + 1, 
                "mse": MSE.eval(fx, y, meta=dataset.metadata).squeeze(1).tolist()[0],
                "nmse": NMSE.eval(fx, y, meta=dataset.metadata).squeeze(1).tolist()[0],
                "vmse": VMSE.eval(fx, y, meta=dataset.metadata).squeeze(1).tolist()[0],
                "vrmse": VRMSE.eval(fx, y, meta=dataset.metadata).squeeze(1).tolist()[0],
                "max_error": LInfinity.eval(fx, y, dataset.metadata).squeeze(1).tolist()[0],

            }
            log.append(metrics)
    return log

def validate_rollout(model, dataset, rollout_loader, device, args):
    model.eval()
    log = []
    rollout_steps = int(dataset.max_rollout_steps)
    count = 0

    with torch.no_grad():
        for batch in tqdm(rollout_loader, desc="Testing Rollout"):
            x = batch["input_fields"].to(device)  # [B, Ti, H, W, F]
            y = batch["output_fields"].to(device)  # [B, To, H, W, F]

            context = x.clone()
            predictions = []

            for step in range(rollout_steps):
                x_in = rearrange(context, "B Ti H W F -> B Ti F H W")  # [B, 4×F, H, W]
                pred = model(x_in)                                      # [B, F, H, W]
                pred = rearrange(pred, "B 1 F H W -> B 1 H W F")          # [B, 1, H, W, F]
                predictions.append(pred)
                context = torch.cat([context[:, 1:], pred], dim=1)

            rollout_preds = torch.cat(predictions, dim=1)  # [B, T, H, W, F]

            current_t_cool = round(float(batch['constant_scalars'][0][0]), 3)

            max_t = min(rollout_preds.shape[1], y.shape[1])
            for t in range(max_t):
                current_timestep = t + dataset.n_steps_input + 1
                pred_t = rollout_preds[:, t:t+1]
                true_t = y[:, t:t+1]

                mse = MSE.eval(pred_t, true_t, meta=dataset.metadata).squeeze(1)
                mse_avg = mse.mean().item()
                mse_fields = mse.mean(dim=0).tolist()

                vrmse = VRMSE.eval(pred_t, true_t, meta=dataset.metadata).squeeze(1)
                vrmse_avg = vrmse.mean().item()
                vrmse_fields = vrmse.mean(dim=0).tolist()

                bsmse_dict = binned_spectral_mse.eval(pred_t, true_t, meta=dataset.metadata)
                bsmse_fields_per_bin = {
                    k: v.squeeze(1).mean(dim=0).tolist()
                    for k, v in bsmse_dict.items() if "mse_per_bin" in k
                }
                bsnmse_fields_per_bin = {
                    k: v.squeeze(1).mean(dim=0).tolist()
                    for k, v in bsmse_dict.items() if "nmse_per_bin" in k
                }
                bsmse_avg_per_bin = {
                    k.replace("mse", "mse_avg"): float(np.mean(vals))
                    for k, vals in bsmse_fields_per_bin.items()
                }
                bsnmse_avg_per_bin = {
                    k.replace("nmse", "nmse_avg"): float(np.mean(vals))
                    for k, vals in bsnmse_fields_per_bin.items()
                }

                log.append({
                    "t_cool": current_t_cool,
                    "timestep": current_timestep,
                    "mse_avg": mse_avg,
                    "mse_fields": mse_fields,
                    "vrmse_avg": vrmse_avg,
                    "vrmse_fields": vrmse_fields,
                    **bsmse_fields_per_bin,
                    **bsnmse_fields_per_bin,
                    **bsmse_avg_per_bin,
                    **bsnmse_avg_per_bin,
                })
            count += 1
    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a computer vision model on The Well dataset")
    parser.add_argument("--model", type=str, default="unet_convnext", choices=["unet_classic", "unet_convnext", "sinenet"], help="Model type")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--data_path", type=str, default="/Users/katoschmidt/Desktop/Thesis_DS/data", help="Base path to dataset")
    parser.add_argument("--include_filters", type=str, nargs="+", default=None, help="Specific HDF5 files to include")
    parser.add_argument("--well_split_name", type=str, default="test", help="Name of split to load - options are 'train', 'valid', 'test'")
    parser.add_argument("--data_workers", type=str, default=14, help="Number of workers to use for data loading.")

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Dataset laden
    dataset = WellDataset(
        well_base_path=args.data_path,
        well_dataset_name="turbulent_radiative_layer_2D",
        well_split_name=args.well_split_name,
        n_steps_input=4,
        n_steps_output=1,
        use_normalization=True,
        normalization_type= ZScoreNormalization,
        min_dt_stride= 1,
        max_dt_stride= 1,
    )

    dataset_rollout = WellDataset(
        well_base_path=args.data_path,
        well_dataset_name="turbulent_radiative_layer_2D",
        well_split_name=args.well_split_name,
        n_steps_input=4,
        n_steps_output=1,
        use_normalization=True,
        normalization_type= ZScoreNormalization,
        min_dt_stride= 1,
        max_dt_stride= 1,
        full_trajectory_mode=True,
        max_rollout_steps=100
        
    )

    onestep_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers= args.data_workers)
    rollout_loader = torch.utils.data.DataLoader(dataset_rollout, batch_size=args.batch_size, shuffle=False, num_workers= args.data_workers)

    if args.model == "sinenet":
        F = dataset.metadata.n_fields
        model = SineNet(dim_in= F * dataset.n_steps_input, dim_out= F * dataset.n_steps_output,
                            n_spatial_dims= dataset.n_spatial_dims,
                            spatial_resolution= dataset.metadata.spatial_resolution, n_input_scalar_components= 2,
                             n_input_vector_components= 1, n_output_scalar_components = 2, n_output_vector_components= 1,
                             time_history= 4, time_future = 1, hidden_channels = 64, padding_mode = "circular", activation =  "gelu", num_layers= 4, num_waves= 4, norm = True, mult= 1.4)

        checkpoint = torch.load("/home/kschmidt/the_well_sinenet/experiments/SineNet_4_(64-140)-SineNet-0.0002/2/checkpoints/best.pt")
        print("Keys in checkpoint:", checkpoint.keys())
        print("Epoch:", checkpoint.get("epoch"))

        model.load_state_dict(checkpoint["model_state_dict"]) 
       
    else:
        F = dataset.metadata.n_fields
        model = UNetConvNext(dim_in= F * dataset.n_steps_input, dim_out= F * dataset.n_steps_output,
                            n_spatial_dims= dataset.n_spatial_dims,
                            spatial_resolution= dataset.metadata.spatial_resolution,
                            init_features=42, blocks_per_stage=2)

        checkpoint = torch.load("/home/kschmidt/the_well/experiments/turbulent_radiative_layer_2D-unet_convnext-UNetConvNext-0.005/5/checkpoints/checkpoint_475.pt")
        print("Keys in checkpoint:", checkpoint.keys())
        print("Epoch:", checkpoint.get("epoch"))

        model.load_state_dict(checkpoint["model_state_dict"]) 

    model.to(device)

    onestep_output = validate_onestep(model, dataset, onestep_loader,  device, args)
    rollout_output = validate_rollout(model, dataset_rollout, rollout_loader, device, args)
   
    output = {"onestep": onestep_output, "rollout": rollout_output}

    os.makedirs("json_files", exist_ok=True)
    with open(f"./json_files/{args.model}_{args.well_split_name}_64_140_0.0002.json", "w") as f:
        json.dump(output, f, indent=4)

    print("Training complete! Results saved in the training log folder")