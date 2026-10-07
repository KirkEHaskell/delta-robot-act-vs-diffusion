# Publishing checklist (first time on GitHub + Hugging Face)

This file is for the author. Delete it before publishing, or keep it as notes.

## 0. Fill in names

Every `KirkHaskell`, `KirkEHaskell` and `Kirk Haskell` in the repo needs to be replaced with your
Hugging Face username, GitHub username and name.

## 1. Hugging Face: dataset + two models

1. Create an account at <https://huggingface.co/join>.
2. Create an access token with **write** permission at <https://huggingface.co/settings/tokens>.
3. Log in from this laptop. Don't upgrade huggingface_hub past 1.x; LeRobot 0.5 needs it below 2.
   ```bash
   hf auth login
   ```
   Paste the token when it asks (nothing shows while you paste). Answer `n` to "Add token as git credential?".
4. From this folder (`delta-robot-imitation`), stage all three uploads:
   ```bash
   python hub/prepare_upload.py --dataset ../bolt_pickplace/datasets/MasterDataset --act ../bolt_pickplace/TRAINEDMODELSFROMMASTERDATASET/checkpoints/act_n100_s50k_050000 --diffusion ../050000/pretrained_model
   ```
5. Run the three `hf upload ...` commands the script prints, then its `create_tag` command.
   The tag is what lets `LeRobotDataset("KirkHaskell/delta_robot_bolt_pick_place")` download the dataset.
6. Open the three pages on huggingface.co and check that they render.

## 2. GitHub: the code repo

1. Create an account at <https://github.com/signup>.
2. Create a **new, empty** repository named `delta-robot-act-vs-diffusion`. Make it public, and
   don't add a README, license or .gitignore (this folder already has them).
3. From this folder (`delta-robot-imitation`), run:
   ```bash
   git init -b main
   ```
   ```bash
   git add .
   ```
   ```bash
   git commit -m "Initial release: ACT vs Diffusion Policy on a delta robot"
   ```
   ```bash
   git remote add origin https://github.com/KirkEHaskell/delta-robot-act-vs-diffusion.git
   ```
   ```bash
   git push -u origin main
   ```
   The first push opens a browser window to sign in to GitHub.
4. On the repo page, click the gear next to **About** and set:
   - **Description:** "ACT vs Diffusion Policy on a home-built delta robot: 240 real-world trials,
     open dataset, CAD, firmware and models (LeRobot)"
   - **Topics:** `robotics` `imitation-learning` `lerobot` `diffusion-policy` `act` `robot-learning`
     `delta-robot` `teleoperation` `pick-and-place` `open-hardware` `pytorch` `real-robot`
5. Settings → General → Social preview: upload `media/photos/leader_and_follower.jpg` or `media/robot_carrying_bolt.jpg` (the image shown when the link is shared).
6. Optional: drag `media/diffusion_best_n100_s50k.mp4` into a README edit on github.com. GitHub turns it into an
   inline video player, which looks better than the GIF.

## 3. Getting it seen

- In the Hugging Face dataset page, the `LeRobot` tag puts it in the LeRobot community datasets list
  (<https://huggingface.co/datasets?other=LeRobot>). That list is where people assembling pretraining
  mixtures look.
- Share it in the LeRobot Discord (#show-and-tell) and on X/LinkedIn with
  `media/diffusion_best_n100_s50k.mp4` and the one-line result: "Default Diffusion Policy learned to
  ignore its cameras on leader–follower data. Blinding proprioception fixed it, and the best model
  went 10/10 on the real robot."
- Link the GitHub repo from the Hugging Face cards (already done) and vice versa (already done).
