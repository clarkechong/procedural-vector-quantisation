"""Build machine-readable cross-experiment metrics from raw CSV/JSON artifacts."""

import csv
import hashlib
import json
from pathlib import Path


ROOT=Path("results")
OUT=ROOT/"e04-07-reference-optimization"


def readcsv(path):return list(csv.DictReader((ROOT/path).open()))
def readjson(path):return json.loads((ROOT/path).read_text())


def main():
    metrics=[]
    def add(experiment,method,name,value,unit,status,source):
        metrics.append(dict(experiment=experiment,method=method,metric=name,value=value,
                            unit=unit,status=status,source=str(source)))
    for row in readcsv("e04-07-reference-optimization/reference/qtip-gaussian-pilot/metrics.csv"):
        add("R2.1",row["codebook"],"open_gaussian_mse",row["mse"],"mse","component-validated",
            "e04-07-reference-optimization/reference/qtip-gaussian-pilot/metrics.csv")
    for row in readcsv("e04-07-reference-optimization/activation-ptq/pilot/metrics.csv"):
        method=row["family"]+"_alpha_"+row["alpha"]
        for metric,unit in (("excess_nll","nats/target"),("activation_nmse","ratio"),
                            ("stored_bpw","bits/weight")):
            add("04B",method,metric,row[metric],unit,"adaptation-only","e04-07-reference-optimization/activation-ptq/pilot/metrics.csv")
    for row in readcsv("e04-07-reference-optimization/activation-ptq/scalar-model-float32/perplexity.csv"):
        for metric,unit in (("perplexity","ppl"),("projection_bpw","bits/weight"),
                            ("weight_nmse","ratio")):
            add("04A",row["method"],metric,row[metric],unit,"adaptation-only",
                "e04-07-reference-optimization/activation-ptq/scalar-model-float32/perplexity.csv")
    for dataset in ("mnist_qat","fashion_qat","mnist_direct","mnist_direct_lr10","mnist_direct_weight10"):
        run = dataset.replace("_", "-")
        for row in readcsv(f"e08-12-low-rate-training/procedural-training/{run}/curves.csv"):
            if int(row["epoch"])==20:
                add("05",dataset+"/"+row["method"]+"/seed"+row["seed"],"test_accuracy",
                    row["test_accuracy"],"fraction","adaptation-only",
                    f"e08-12-low-rate-training/procedural-training/{run}/curves.csv")
    for row in readcsv("e04-07-reference-optimization/learned-optimizer/inverse-pilot/metrics.csv"):
        for metric,unit in (("mean_regret","squared_error"),("oracle_hit","fraction")):
            add("07",row["method"],metric,row[metric],unit,"exploratory",
                "e04-07-reference-optimization/learned-optimizer/inverse-pilot/metrics.csv")
    for row in readcsv("e04-07-reference-optimization/learned-optimizer/real-trace/metrics.csv"):
        for metric,unit in (("mean_regret","squared_error"),("nontrivial_oracle_hit","fraction")):
            add("07_real_trace",row["method"],metric,row[metric],unit,"exploratory",
                "e04-07-reference-optimization/learned-optimizer/real-trace/metrics.csv")
    task=readjson("e04-07-reference-optimization/learned-optimizer/task-update/metrics.json")
    for method in ("exact","learned_top4","one_bit"):
        add("07_task_update",method,"holdout_batch_ce",task[method]["holdout_loss"],
            "nats/example","exploratory","e04-07-reference-optimization/learned-optimizer/task-update/metrics.json")
    identity=readjson("e04-07-reference-optimization/reference/identity/metrics.json")
    add("R0","identity","max_logit_diff",identity["max_logit_diff"],"logit",
        "component-validated","e04-07-reference-optimization/reference/identity/metrics.json")
    smooth=readjson("e04-07-reference-optimization/reference/smoothquant-model-parity/metrics.json")
    add("R1","smoothquant_smollm2","max_logit_diff",smooth["max_logit_diff"],"logit",
        "component-validated","e04-07-reference-optimization/reference/smoothquant-model-parity/metrics.json")
    with (OUT/"metrics.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=metrics[0].keys());writer.writeheader();writer.writerows(metrics)
    ledger=[
      dict(reference="SmoothQuant README Llama-2-7B FP16",upstream_commit="c61476d728e42ae0d8a35e7e78494edcac3237b5",expected="5.474 PPL",actual="",tolerance="3%",deviation="no local HF token; token=False request returned 401; no upstream dataset run",status="blocked",artifact="e04-07-reference-optimization/reference/smoothquant-model-parity/metrics.json"),
      dict(reference="SmoothQuant README Llama-2-7B W8A8 alpha .85",upstream_commit="c61476d728e42ae0d8a35e7e78494edcac3237b5",expected="5.515 PPL",actual="",tolerance="3%",deviation="no local HF token; token=False request returned 401; only transform parity",status="blocked",artifact="e04-07-reference-optimization/reference/smoothquant-model-parity/metrics.json"),
      dict(reference="QTIP v2 Table 1 1MAD 2 bpw",upstream_commit="e90c6688c8dfae326a3a81b5eb032db7c6680ec0",expected="0.069 MSE",actual="0.065661 open-boundary 256-vector pilot",tolerance="0.005",deviation="open boundary rather than paper convention",status="component-validated",artifact="e04-07-reference-optimization/reference/qtip-gaussian-pilot/metrics.csv"),
      dict(reference="QTIP v2 Table 1 3INST 2 bpw",upstream_commit="e90c6688c8dfae326a3a81b5eb032db7c6680ec0",expected="0.069 MSE",actual="0.070866 open-boundary 256-vector pilot",tolerance="0.005",deviation="open boundary rather than paper convention",status="component-validated",artifact="e04-07-reference-optimization/reference/qtip-gaussian-pilot/metrics.csv"),
      dict(reference="QTIP v2 Table 5 Llama-2-7B HYB WikiText-2",upstream_commit="e90c6688c8dfae326a3a81b5eb032db7c6680ec0",expected="5.12 dense / 5.86 quantized PPL",actual="",tolerance="3% dense, 5% quantized",deviation="upstream CUDA kernel dependency and gated model/tokenizer",status="blocked",artifact=""),
      dict(reference="QTIP v2 Table 3 Llama-2-7B 1MAD 2 bpw",upstream_commit="e90c6688c8dfae326a3a81b5eb032db7c6680ec0",expected="7.05 PPL",actual="",tolerance="5%",deviation="full Hessian/RHT/BlockLDLQ quantizer not run",status="blocked",artifact=""),
    ]
    with (OUT/"reference_ledger.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=ledger[0].keys());writer.writeheader();writer.writerows(ledger)
    (OUT/"config.json").write_text(json.dumps(dict(plan_sha256=(json.loads((OUT/"config.json").read_text()).get("plan_sha256") if (OUT/"config.json").exists() else None),
        date_utc="2026-11-27",phase="bounded pilots and independent components",selection="validation only; old test already seen",
        reference_source_roots=["/tmp/qtip-official","/tmp/smoothquant-official"]),indent=2)+"\n")
    (OUT/"manifest.json").write_text(json.dumps(dict(
        token_pilot="e04-07-reference-optimization/activation-ptq/pilot/manifest.json",mnist="e08-12-low-rate-training/procedural-training/mnist-qat/manifest.json",
        fashion="e08-12-low-rate-training/procedural-training/fashion-qat/manifest.json",
        blind_final_test=False,reason="Experiment 01-03 already used local test rows/tokens"),indent=2)+"\n")
    hashes={}
    for directory in (ROOT/"e04-07-reference-optimization"/"reference",ROOT/"e04-07-reference-optimization"/"activation-ptq",ROOT/"e08-12-low-rate-training"/"procedural-training",ROOT/"e04-07-reference-optimization"/"learned-optimizer"):
        if directory.exists():
            for path in directory.rglob("*"):
                if path.is_file():
                    hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT/"artifact_hashes.json").write_text(json.dumps(hashes,indent=2)+"\n")
    print(len(metrics),"metric rows",len(hashes),"hashed artifacts")


if __name__=="__main__":main()
