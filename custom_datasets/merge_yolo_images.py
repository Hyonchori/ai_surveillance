import os


def get_mot_paths(mot_root):
    train_imgs = os.listdir(os.path.join(mot_root, "train", "train", "images"))
    val_imgs = os.listdir(os.path.join(mot_root, "train", "val", "images"))
    train_paths = [os.path.join(mot_root, "train", "train", "images", x) for x in train_imgs]
    val_paths = [os.path.join(mot_root, "train", "val", "images", x) for x in val_imgs]
    print("\n--- MOT17")
    print(f"\ttrain: {len(train_paths)}/ val: {len(val_paths)}")
    return train_paths, val_paths


def get_citypersons_paths(citypersons_root):
    train_paths = []
    val_paths = []
    train_dirs = [os.path.join(citypersons_root, "images", "train", x) for x in
                  os.listdir(os.path.join(citypersons_root, "images", "train"))]
    val_dirs = [os.path.join(citypersons_root, "images", "val", x) for x in
                os.listdir(os.path.join(citypersons_root, "images", "val"))]
    for train_dir in train_dirs:
        tmp_imgs = os.listdir(train_dir)
        for tmp_img in tmp_imgs:
            train_paths.append(os.path.join(train_dir, tmp_img))
    for val_dir in val_dirs:
        tmp_imgs = os.listdir(val_dir)
        for tmp_img in tmp_imgs:
            val_paths.append(os.path.join(val_dir, tmp_img))
    print("\n--- Citypersons")
    print(f"\ttrain: {len(train_paths)} / val: {len(val_paths)}")
    return train_paths, val_paths


def get_crowdhuman_paths(crowdhuman_root):
    train_img_dir = os.path.join(crowdhuman_root, "Crowdhuman_train", "images")
    val_img_dir = os.path.join(crowdhuman_root, "Crowdhuman_val", "images")
    train_paths = [os.path.join(train_img_dir, x) for x in os.listdir(train_img_dir)]
    val_paths = [os.path.join(val_img_dir, x) for x in os.listdir(val_img_dir)]
    print("\n--- Crowdhuman")
    print(f"\ttrain: {len(train_paths)} / val: {len(val_paths)}")
    return train_paths, val_paths


def get_ethz_paths(ethz_root):
    train_paths = []
    dirs = [x for x in os.listdir(ethz_root) if "eth" in x]
    for dir in dirs:
        img_dir = os.path.join(ethz_root, dir, "images")
        img_names = os.listdir(img_dir)
        for img_name in img_names:
            train_paths.append(os.path.join(img_dir, img_name))
    print("\n--- ETHZ")
    print(f"\ttrain: {len(train_paths)}")
    return train_paths


def path2txt(paths, out_name):
    with open(out_name, "w") as f:
        f.write("\n".join(paths))

if __name__ == "__main__":
    root = "/media/daton/Data/datasets"
    mot_root = os.path.join(root, "MOT17")
    crowdhuman_root = os.path.join(root, "crowdhuman")
    citypersons_root = os.path.join(root, "citypersons")
    ethz_root = os.path.join(root, "ETHZ")

    mot_train_paths, mot_val_paths = get_mot_paths(mot_root)
    city_train_paths, city_val_paths = get_citypersons_paths(citypersons_root)
    crowd_train_paths, crowd_val_paths = get_crowdhuman_paths(crowdhuman_root)
    ethz_train_paths = get_ethz_paths(ethz_root)

    train_paths = mot_train_paths + crowd_train_paths + city_train_paths #+ ethz_train_paths
    val_paths = mot_val_paths + crowd_val_paths + city_val_paths

    path2txt(train_paths, "train.txt")
    path2txt(val_paths, "val.txt")