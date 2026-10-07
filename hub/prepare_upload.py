"""
prepare_upload.py -- stage the dataset and the two published checkpoints for Hugging Face.

Copies (never modifies) the originals into hub/staging/, adds the cards as README.md, sets the
dataset's robot_type, and adds the BLIND_STATE marker to the Diffusion checkpoint.

  python hub/prepare_upload.py \
      --dataset  path/to/MasterDataset \
      --act      path/to/act_n100_s50k_050000 \
      --diffusion path/to/diffusionnostateh48_n100_s50k/checkpoints/050000/pretrained_model

Any of --dataset/--act/--diffusion can be left out (e.g. stage the Diffusion model on the machine
that has it). Then upload with the commands printed at the end (see PUBLISHING.md).
"""

import argparse
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE = os.path.join(HERE, "staging")
NEEDED = ["config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"]


def fill(card, a):
    s = open(os.path.join(HERE, card), encoding="utf-8").read()
    return (s.replace("<HF_USER>", a.hf_user).replace("<GITHUB_USER>", a.github_user)
             .replace("<YOUR NAME>", a.author))


def copy_tree(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("training_state", "images"))


def stage_model(src, name, card, a, blind=False):
    missing = [f for f in NEEDED if not os.path.exists(os.path.join(src, f))]
    if missing:
        raise SystemExit(f"{src} is not a LeRobot checkpoint folder (missing {missing}). "
                         "Point at the pretrained_model folder.")
    dst = os.path.join(STAGE, name)
    copy_tree(src, dst)
    open(os.path.join(dst, "README.md"), "w", encoding="utf-8").write(fill(card, a))
    if blind:
        open(os.path.join(dst, "BLIND_STATE"), "w").write(
            "This Diffusion Policy was trained with observation.state zeroed after normalization.\n"
            "Zero it the same way at inference (robot/run_policy.py does this automatically).\n")
    print(f"staged {name}: {dst}")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-user", default="KirkHaskell")
    ap.add_argument("--github-user", default="KirkEHaskell")
    ap.add_argument("--author", default="Kirk Haskell")
    ap.add_argument("--dataset")
    ap.add_argument("--act")
    ap.add_argument("--diffusion")
    a = ap.parse_args()
    os.makedirs(STAGE, exist_ok=True)
    cmds = []

    if a.dataset:
        dst = os.path.join(STAGE, "delta_robot_bolt_pick_place")
        copy_tree(a.dataset, dst)
        info_p = os.path.join(dst, "meta", "info.json")
        info = json.load(open(info_p))
        info["robot_type"] = "delta_3dof"
        json.dump(info, open(info_p, "w"), indent=4)
        open(os.path.join(dst, "README.md"), "w", encoding="utf-8").write(fill("dataset_card.md", a))
        print(f"staged dataset: {dst}  ({info['total_episodes']} episodes, robot_type=delta_3dof)")
        cmds.append(f"hf upload {a.hf_user}/delta_robot_bolt_pick_place \"{dst}\" . --repo-type dataset")
    if a.act:
        dst = stage_model(a.act, "act_delta_robot_bolt_pick_place", "model_card_act.md", a)
        cmds.append(f"hf upload {a.hf_user}/act_delta_robot_bolt_pick_place \"{dst}\" .")
    if a.diffusion:
        dst = stage_model(a.diffusion, "diffusion_delta_robot_bolt_pick_place", "model_card_diffusion.md", a, blind=True)
        cmds.append(f"hf upload {a.hf_user}/diffusion_delta_robot_bolt_pick_place \"{dst}\" .")

    if cmds:
        print("\nUpload (after `hf auth login`):")
        for c in cmds:
            print("  " + c)
        print("\nThen tag the dataset as LeRobot v3.0 (needed by LeRobotDataset to download it):")
        print(f"  python -c \"from huggingface_hub import HfApi; HfApi().create_tag('{a.hf_user}/delta_robot_bolt_pick_place', "
              "tag='v3.0', repo_type='dataset')\"")


if __name__ == "__main__":
    main()
