import os
from matplotlib import pyplot as plt
import glob

log_pardir = "/home/sravit/multimodal/multimodal_biometric_authentication/exp/face"

all_metrics = {
    "Full precision": {
        "log_name": "face_2022-07-17-21-17-27",
        "Epoch": [],
        "Loss": [],
        "Top1": [],
        "Top5": [],
        "LR": [],
        "EER": []
    },
    "XNOR 8/8": {
        "log_name": "face_2022-07-17-15-52-16",
        "Epoch": [],
        "Loss": [],
        "Top1": [],
        "Top5": [],
        "LR": [],
        "EER": []
    },
    "XNOR 2/2": {
        "log_name": "face_2022-07-19-06-46-34",
        "Epoch": [],
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
    
    epoch_counter = 1
    
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
        
        print(metrics)

        ax_loss.plot("Epoch", "Loss", data=metrics, label=modelName)
        ax_top1.plot("Epoch", "Top1", data=metrics, label=modelName)
        ax_top5.plot("Epoch", "Top5", data=metrics, label=modelName)
        #ax_lr.plot("Epoch", "LR", data=metrics, label=modelName)
        metrics["Epoch"] = metrics["Epoch"][:len(metrics["EER"])]
        ax_eer.plot("Epoch", "EER", data=metrics, label=modelName)

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