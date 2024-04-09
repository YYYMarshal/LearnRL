import os

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
root_dir = os.path.join(desktop_path, r"hymenoptera_data\train")
target_dir = "ants_image"
img_path = os.listdir(os.path.join(root_dir, target_dir))
label = target_dir.split("_")[0]
out_dir = "ants_label"
for item in img_path:
    file_name = item.split(".jpg")[0]
    with open(os.path.join(root_dir, out_dir, f"{file_name}.txt"), "w") as f:
        f.write(label)
