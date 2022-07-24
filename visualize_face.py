import os
from matplotlib import pyplot as plt
import glob
import configparser

log_pardir = "/home/sravit/multimodal/multimodal_biometric_authentication/exp/face"

all_metrics = {
    "XNOR 2/1": {
        "log_name": "face_2022-07-22-06-47-53",
        "Epoch": [],
        "Val Epoch": [],
        "Loss": [],
        "Top1": [],
        "Top5": [],
        "LR": [],
        "EER": []
    },
    "FP 2/1": {
        "log_name": "face_2022-07-22-21-58-35",
        "Epoch": [],
        "Val Epoch": [],
        "Loss": [],
        "Top1": [],
        "Top5": [],
        "LR": [],
        "EER": []
    }
}


log_vis_path = os.path.join(log_pardir, list(all_metrics.values())[0]["log_name"], "vis")
if not os.path.exists(log_vis_path):
    os.mkdir(log_vis_path)

fig_loss, ax_loss = plt.subplots()
fig_top1, ax_top1 = plt.subplots()
fig_top5, ax_top5 = plt.subplots()
fig_lr, ax_lr = plt.subplots()
fig_eer, ax_eer = plt.subplots()

for modelName in all_metrics.keys():
    log_name = all_metrics[modelName]["log_name"]
    log_dir = os.path.join(log_pardir, log_name)
    log_path = glob.glob(os.path.join(log_dir, "*.log"))[0]
    config_path = glob.glob(os.path.join(log_dir, "*.conf"))[0]

    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.optionxform=str
    config.read(config_path)

    conf_params = {
        "exp_params": {k: eval(v) for k, v in config['exp'].items()},
        "data_params": {k: eval(v) for k, v in config['data'].items()},
        "optim_params": {k: eval(v) for k, v in config['optimization'].items()}
    }

    epoch_counter = 1
    val_epoch_counter = 1
    val_freq = conf_params["optim_params"]["val_frequency_epoch"]
    
    with open(log_path, 'r') as f:
        metrics = all_metrics[modelName]
        for line in f.readlines():
            if "3100/3125" in line:
                metrics["Epoch"].append(epoch_counter)
                metrics["Loss"].append(float(line.split("Loss")[1].split(")")[0].split("(")[1]))
                metrics["Top1"].append(float(line.split("Acc@1")[1].split(")")[0].split("(")[1]))
                metrics["Top5"].append(float(line.split("Acc@5")[1].split(")")[0].split("(")[1]))
                epoch_counter += 1
            elif "LR" in line:
                metrics["LR"].append(float(line.split(" ")[-1]))
            elif "Validation EER" in line:
                metrics["EER"].append(float(line.split(" ")[-1]))
                metrics["Val Epoch"].append(val_epoch_counter)
                val_epoch_counter += val_freq
        
        ax_loss.plot("Epoch", "Loss", data=metrics, label=modelName)
        ax_top1.plot("Epoch", "Top1", data=metrics, label=modelName)
        ax_top5.plot("Epoch", "Top5", data=metrics, label=modelName)
        #ax_lr.plot("Epoch", "LR", data=metrics, label=modelName)
        ax_eer.plot("Val Epoch", "EER", data=metrics, label=modelName)

ax_loss.set_title("Loss vs Epoch")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Loss")
ax_loss.legend()
fig_loss.savefig(os.path.join(log_vis_path, "loss.png"))

ax_top1.set_title("Top1 vs Epoch")
ax_top1.set_xlabel("Epoch")
ax_top1.set_ylabel("Top1")
ax_top1.legend()
fig_top1.savefig(os.path.join(log_vis_path, "top1.png"))

ax_top5.set_title("Top5 vs Epoch")
ax_top5.set_xlabel("Epoch")
ax_top5.set_ylabel("Top5")
ax_top5.legend()
fig_top5.savefig(os.path.join(log_vis_path, "top5.png"))

"""
ax_lr.set_title("LR vs Epoch")
ax_lr.set_xlabel("Epoch")
ax_lr.set_ylabel("LR")
ax_lr.legend()
fig_lr.savefig(os.path.join(log_vis_path, "lr.png"))
"""

ax_eer.set_title("EER vs Epoch")
ax_eer.set_xlabel("Epoch")
ax_eer.set_ylabel("EER")
ax_eer.legend()
fig_eer.savefig(os.path.join(log_vis_path, "eer.png"))