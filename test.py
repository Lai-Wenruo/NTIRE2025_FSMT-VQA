import torch
import vqa.models as models
import vqa.datasets as datasets
import argparse
import numpy as np
from tqdm import tqdm
import pickle
import yaml
from thop import profile



def profile_inference2(inf_set, model, device):
    video = {}
    data = inf_set[0]
    for key in sample_types:
        if key in data:
            video[key] = data[key].to(device)
            c, t, h, w = video[key].shape
            video[key] = video[key].reshape(1, c, data["num_clips"][key], t // data["num_clips"][key], h,
                                            w).permute(0, 2, 1, 3, 4, 5).reshape(data["num_clips"][key], c,
                                                                                 t // data["num_clips"][key], h, w)
    with torch.no_grad():
        flops, params = profile(model, (video,))
    print(f"The FLOps of the Variant is {flops / 1e9:.1f}G, with Params {params / 1e6:.2f}M.")



def rescale(pr, gt=None):
    if gt is None:
        pr = (pr - np.mean(pr)) / np.std(pr)
        score = 1 / (1+np.exp(-pr))
        score = score*100

    else:
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return score


sample_types = ["resize", "fragments", "crop", "arp_resize", "arp_fragments"]




def inference_set(inf_loader, model, device, best_, save_model=False, suffix='s', set_name="na"):
        print(f"Validating for {set_name}.")
        results = []

        model.eval()

        keys = []

        for i, data in enumerate(tqdm(inf_loader, desc="Validating")):
            result = dict()
            video = {}
            for key in sample_types:
                if key not in keys:
                    keys.append(key)
                if key in data:
                    video[key] = data[key].to(device)
                    b, c, t, h, w = video[key].shape
                    video[key] = video[key].reshape(b, c, data["num_clips"][key], t // data["num_clips"][key], h,
                                                    w).permute(0, 2, 1, 3, 4, 5).reshape(b * data["num_clips"][key], c,
                                                                                         t // data["num_clips"][key], h,
                                                                                         w)
            with torch.no_grad():
                labels = model(video, reduce_scores=False)
                labels = [np.mean(l.cpu().numpy()) for l in labels]
                result["pr_labels"] = labels
            # result["gt_label"] = data["gt_label"].item()
            result["name"] = data["name"]
            # result['frame_inds'] = data['frame_inds']
            # del data
            results.append(result)

        ## generate the demo video for video quality localization
        # gt_labels = [r["gt_label"] for r in results]
        pr_labels = 0
        pr_dict = {}
        for i, key in zip(range(len(results[0]["pr_labels"])), keys):
            key_pr_labels = np.array([np.mean(r["pr_labels"][i]) for r in results])
            pr_dict[key] = key_pr_labels
            pr_labels += rescale(key_pr_labels)

        with open(f"dover_predictions/{set_name}.pkl", "wb") as f:
            pickle.dump(pr_dict, f)

        pr_labels = rescale(pr_labels, None)
        print(pr_labels)
        # 指定要写入的 txt 文件路径
        file_path="label.txt"
        # 将 pre_label 中的每个数值写入 txt 文件，一行一个数
        with open(file_path, "w") as file:
            for num in pr_labels:
                file.write(f"{num}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--opt", type=str, default="./options/test/test-color.yml", help="the option file"
    )

    args = parser.parse_args()
    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
    print(opt)

    ## adaptively choose the device

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    ## defining vqa and loading checkpoint

    bests_ = []

    model = getattr(models, opt["model"]["type"])(**opt["model"]["args"]).to(device)

    state_dict = torch.load(opt["test_load_path"], map_location=device)["state_dict"]

    model.load_state_dict(state_dict, strict=True)

    for key in opt["data"].keys():

        if "val" not in key and "test" not in key:
            continue



        val_dataset = getattr(datasets, opt["data"][key]["type"])(opt["data"][key]["args"])

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, num_workers=opt["num_workers"], pin_memory=True,
        )

        profile_inference2(val_dataset, model, device)

        # test the vqa
        print(len(val_loader))

        inference_set(
            val_loader,
            model,
            device, bests_,
            set_name=key,
        )




if __name__ == "__main__":
    main()
