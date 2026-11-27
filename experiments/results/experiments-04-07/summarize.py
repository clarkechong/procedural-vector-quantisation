"""Build machine-readable cross-experiment metrics from raw CSV/JSON artifacts."""

import csv
import hashlib
import json
from pathlib import Path


ROOT=Path("experiments/results")
OUT=ROOT/"experiments-04-07"


def readcsv(path):return list(csv.DictReader((ROOT/path).open()))
def readjson(path):return json.loads((ROOT/path).read_text())


def main():
    metrics=[]
    def add(experiment,method,name,value,unit,status,source):
        metrics.append(dict(experiment=experiment,method=method,metric=name,value=value,
                            unit=unit,status=status,source=str(source)))
    for row in readcsv("reference/qtip-gaussian-pilot/metrics.csv"):
        add("R2.1",row["codebook"],"open_gaussian_mse",row["mse"],"mse","component-validated",
            "reference/qtip-gaussian-pilot/metrics.csv")
    for row in readcsv("activation_ptq/pilot/metrics.csv"):
        method=row["family"]+"_alpha_"+row["alpha"]
        for metric,unit in (("excess_nll","nats/target"),("activation_nmse","ratio"),
                            ("stored_bpw","bits/weight")):
            add("04B",method,metric,row[metric],unit,"adaptation-only","activation_ptq/pilot/metrics.csv")
    for row in readcsv("activation_ptq/scalar_model_float32/perplexity.csv"):
        for metric,unit in (("perplexity","ppl"),("projection_bpw","bits/weight"),
                            ("weight_nmse","ratio")):
            add("04A",row["method"],metric,row[metric],unit,"adaptation-only",
                "activation_ptq/scalar_model_float32/perplexity.csv")
    for dataset in ("mnist_qat","fashion_qat","mnist_direct","mnist_direct_lr10","mnist_direct_weight10"):
        for row in readcsv(f"procedural_training/{dataset}/curves.csv"):
            if int(row["epoch"])==20:
                add("05",dataset+"/"+row["method"]+"/seed"+row["seed"],"test_accuracy",
                    row["test_accuracy"],"fraction","adaptation-only",
                    f"procedural_training/{dataset}/curves.csv")
    for row in readcsv("learned_optimizer/inverse_pilot/metrics.csv"):
        for metric,unit in (("mean_regret","squared_error"),("oracle_hit","fraction")):
            add("07",row["method"],metric,row[metric],unit,"exploratory",
                "learned_optimizer/inverse_pilot/metrics.csv")
    for row in readcsv("learned_optimizer/real_trace/metrics.csv"):
        for metric,unit in (("mean_regret","squared_error"),("nontrivial_oracle_hit","fraction")):
            add("07_real_trace",row["method"],metric,row[metric],unit,"exploratory",
                "learned_optimizer/real_trace/metrics.csv")
    task=readjson("learned_optimizer/task_update/metrics.json")
    for method in ("exact","learned_top4","one_bit"):
        add("07_task_update",method,"holdout_batch_ce",task[method]["holdout_loss"],
            "nats/example","exploratory","learned_optimizer/task_update/metrics.json")
    identity=readjson("reference/identity/metrics.json")
    add("R0","identity","max_logit_diff",identity["max_logit_diff"],"logit",
        "component-validated","reference/identity/metrics.json")
    smooth=readjson("reference/smoothquant-model-parity/metrics.json")
    add("R1","smoothquant_smollm2","max_logit_diff",smooth["max_logit_diff"],"logit",
        "component-validated","reference/smoothquant-model-parity/metrics.json")
    with (OUT/"metrics.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=metrics[0].keys());writer.writeheader();writer.writerows(metrics)
    ledger=[
      dict(reference="SmoothQuant README Llama-2-7B FP16",upstream_commit="c61476d728e42ae0d8a35e7e78494edcac3237b5",expected="5.474 PPL",actual="",tolerance="3%",deviation="no local HF token; token=False request returned 401; no upstream dataset run",status="blocked",artifact="reference/smoothquant-model-parity/metrics.json"),
      dict(reference="SmoothQuant README Llama-2-7B W8A8 alpha .85",upstream_commit="c61476d728e42ae0d8a35e7e78494edcac3237b5",expected="5.515 PPL",actual="",tolerance="3%",deviation="no local HF token; token=False request returned 401; only transform parity",status="blocked",artifact="reference/smoothquant-model-parity/metrics.json"),
      dict(reference="QTIP v2 Table 1 1MAD 2 bpw",upstream_commit="e90c6688c8dfae326a3a81b5eb032db7c6680ec0",expected="0.069 MSE",actual="0.065661 open-boundary 256-vector pilot",tolerance="0.005",deviation="open boundary rather than paper convention",status="component-validated",artifact="reference/qtip-gaussian-pilot/metrics.csv"),
      dict(reference="QTIP v2 Table 1 3INST 2 bpw",upstream_commit="e90c6688c8dfae326a3a81b5eb032db7c6680ec0",expected="0.069 MSE",actual="0.070866 open-boundary 256-vector pilot",tolerance="0.005",deviation="open boundary rather than paper convention",status="component-validated",artifact="reference/qtip-gaussian-pilot/metrics.csv"),
      dict(reference="QTIP v2 Table 5 Llama-2-7B HYB WikiText-2",upstream_commit="e90c6688c8dfae326a3a81b5eb032db7c6680ec0",expected="5.12 dense / 5.86 quantized PPL",actual="",tolerance="3% dense, 5% quantized",deviation="upstream CUDA kernel dependency and gated model/tokenizer",status="blocked",artifact=""),
      dict(reference="QTIP v2 Table 3 Llama-2-7B 1MAD 2 bpw",upstream_commit="e90c6688c8dfae326a3a81b5eb032db7c6680ec0",expected="7.05 PPL",actual="",tolerance="5%",deviation="full Hessian/RHT/BlockLDLQ quantizer not run",status="blocked",artifact=""),
    ]
    with (OUT/"reference_ledger.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=ledger[0].keys());writer.writeheader();writer.writerows(ledger)
    (OUT/"config.json").write_text(json.dumps(dict(plan_sha256=hashlib.sha256(Path("experiments_04_07_agent_plan.md").read_bytes()).hexdigest(),
        date_utc="2026-11-27",phase="bounded pilots and independent components",selection="validation only; old test already seen",
        reference_source_roots=["/tmp/qtip-official","/tmp/smoothquant-official"]),indent=2)+"\n")
    (OUT/"manifest.json").write_text(json.dumps(dict(
        token_pilot="activation_ptq/pilot/manifest.json",mnist="procedural_training/mnist_qat/manifest.json",
        fashion="procedural_training/fashion_qat/manifest.json",
        blind_final_test=False,reason="Experiment 01-03 already used local test rows/tokens"),indent=2)+"\n")
    hashes={}
    for directory in (ROOT/"reference",ROOT/"activation_ptq",ROOT/"procedural_training",ROOT/"learned_optimizer"):
        if directory.exists():
            for path in directory.rglob("*"):
                if path.is_file():
                    hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT/"artifact_hashes.json").write_text(json.dumps(hashes,indent=2)+"\n")
    print(len(metrics),"metric rows",len(hashes),"hashed artifacts")


if __name__=="__main__":main()
