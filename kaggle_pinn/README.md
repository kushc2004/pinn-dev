# Kaggle execution track

Runs the full experiment pipeline on a Tesla P100 kernel, mirroring the
workflow used by the RankLab project.

## One-time setup

1. Upload the (cleared, private) dataset and create the artifact-cache slot:

   ```sh
   mkdir -p artifacts/source-dataset
   cp data/cleaned_chiller-3.csv artifacts/source-dataset/
   cat > artifacts/source-dataset/dataset-metadata.json <<'EOF'
   {"title": "Chiller-3 PINN Data", "id": "kushchaudhari/chiller3-pinn-data", "licenses": [{"name": "other"}]}
   EOF
   kaggle datasets create -p artifacts/source-dataset
   ```

2. Push this repository to GitHub; the driver notebook clones it at launch.
3. `pip install kaggle` locally with `~/.kaggle/kaggle.json` present.

## Launch and follow

```sh
kaggle kernels push -p kaggle_pinn
python3 scripts/watch_kaggle_kernel.py \
  --slug kushchaudhari/pinn-chiller-pipeline \
  --output outputs/logs/kaggle_pinn.log
```

## Artifact cache / resume

After a successful run, download the kernel output `pinn_artifacts.tar.gz`
and version the artifact dataset so future reruns resume instead of
recomputing:

```sh
python3 scripts/restore_kaggle_artifacts.py --archive pinn_artifacts.tar.gz
python3 scripts/publish_kaggle_artifacts.py --create   # first time
# later: python3 scripts/publish_kaggle_artifacts.py
```

Then add `"kushchaudhari/pinn-chiller-artifacts"` to `dataset_sources` in
`kernel-metadata.json` and re-push: the next run attaches the cache,
validates its manifest against the dataset fingerprint, restores
`results/`, and `src/run_all.py` executes only the stages that are not yet
complete.
