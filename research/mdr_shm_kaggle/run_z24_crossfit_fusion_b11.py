from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from run_z24_b2 import Model,features,learn_anchors,load_vertical,normalized_sensor_spectra
from run_z24_model_a_v3 import prepare_frame
from run_z24_model_a_v31 import corrupt_v31,safe_metric_dict
from run_z24_shape_fusion_b1 import CONDITIONS,ShapeModel,damage_groups,record_metadata,shape_feature

METHODS=("modal_cf","shape_cf","union_cf","consensus_cf")

def fit_modal_raw(rows):
    spectra=[]; frequencies=None
    for row in rows.itertuples():
        frequencies,psd=normalized_sensor_spectra(load_vertical(Path(row.resolved_path)))
        spectra.append(np.median(psd,axis=0))
    if frequencies is None or len(spectra)<4: raise RuntimeError("Need >=4 healthy records")
    anchors=learn_anchors(spectra,frequencies)
    matrix=np.stack([features(x,frequencies,anchors) for x in spectra])
    return Model.fit(anchors,matrix)

def fit_shape_raw(rows):
    matrix=[]
    for row in rows.itertuples():
        f,p=normalized_sensor_spectra(load_vertical(Path(row.resolved_path)))
        matrix.append(shape_feature(f,p,np.ones(p.shape[0],dtype=bool)))
    if len(matrix)<4: raise RuntimeError("Need >=4 healthy records")
    return ShapeModel.fit(np.stack(matrix))

def raw_scores(modal,shape,signals,availability=None):
    f,p=normalized_sensor_spectra(signals)
    if availability is None: availability=np.ones(p.shape[0],dtype=bool)
    spectrum=np.median(p[availability],axis=0)
    return (
        modal.raw_score(features(spectrum,f,modal.anchors)),
        shape.raw_score(shape_feature(f,p,availability)),
    )

def qhigher(values,p):
    return float(np.quantile(np.asarray(values),p,method="higher"))

def crossfit_excitation(healthy,excitation,alpha):
    subset=healthy[healthy.excitation_resolved==excitation].copy()
    subset["setup_cf"]=[record_metadata(str(x))[1] for x in subset.record_id]
    groups=sorted(subset.setup_cf.unique())
    if len(groups)<3:
        subset["setup_cf"]=np.arange(len(subset)); groups=list(subset.setup_cf)
    rows=[]
    for group in groups:
        fit_rows=subset[subset.setup_cf!=group]
        holdout=subset[subset.setup_cf==group]
        modal=fit_modal_raw(fit_rows); shape=fit_shape_raw(fit_rows)
        for row in holdout.itertuples():
            mr,sr=raw_scores(modal,shape,load_vertical(Path(row.resolved_path)))
            rows.append({"record_id":row.record_id,"excitation":excitation,
                         "setup":int(row.setup_cf),"modal_raw":mr,"shape_raw":sr})
    oof=pd.DataFrame(rows); probability=1-alpha
    mt=qhigher(oof.modal_raw,probability); st=qhigher(oof.shape_raw,probability)
    oof["modal_cf"]=oof.modal_raw/max(mt,1e-12)
    oof["shape_cf"]=oof.shape_raw/max(st,1e-12)
    ur=np.maximum(oof.modal_cf,oof.shape_cf)
    cr=np.sqrt(np.maximum(oof.modal_cf,0)*np.maximum(oof.shape_cf,0))
    ut=qhigher(ur,probability); ct=qhigher(cr,probability)
    oof["union_cf"]=ur/max(ut,1e-12)
    oof["consensus_cf"]=cr/max(ct,1e-12)
    final_modal=fit_modal_raw(subset); final_shape=fit_shape_raw(subset)
    final_modal.threshold=mt; final_shape.threshold=st
    calibration={
        "alpha":alpha,"healthy_records":int(len(subset)),"folds":int(len(groups)),
        "modal_threshold":mt,"shape_threshold":st,
        "union_threshold":ut,"consensus_threshold":ct,
        "oof_false_alarm_rate":{
            method:float(np.mean(oof[method].to_numpy()>1.0)) for method in METHODS
        },
    }
    return final_modal,final_shape,calibration,oof

def normalized_scores(mr,sr,c):
    modal=mr/max(c["modal_threshold"],1e-12)
    shape=sr/max(c["shape_threshold"],1e-12)
    union=max(modal,shape)/max(c["union_threshold"],1e-12)
    consensus=np.sqrt(max(modal,0)*max(shape,0))/max(c["consensus_threshold"],1e-12)
    return {"modal_cf":float(modal),"shape_cf":float(shape),
            "union_cf":float(union),"consensus_cf":float(consensus)}

def metrics_table(frame):
    rows=[]; cols=["method","condition","replicate"]
    for key,group in frame.groupby(cols,sort=True):
        rows.append({**dict(zip(cols,key)),"scope":"overall","excitation":"all",
                     "records":len(group),
                     **safe_metric_dict(group.label.to_numpy(),group.score.to_numpy())})
        for excitation,subset in group.groupby("excitation"):
            rows.append({**dict(zip(cols,key)),"scope":"excitation",
                         "excitation":excitation,"records":len(subset),
                         **safe_metric_dict(subset.label.to_numpy(),subset.score.to_numpy())})
    return pd.DataFrame(rows)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input-root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,
                        default=Path("/kaggle/working/z24_crossfit_fusion_b11"))
    parser.add_argument("--mode",choices=("quick","full"),default="quick")
    parser.add_argument("--alpha",type=float,default=.10)
    args=parser.parse_args()
    if not 0<args.alpha<.5: raise ValueError("alpha must be between 0 and 0.5")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    frame,manifest=prepare_frame(args.input_root)
    healthy=frame[(frame.label==0)&frame.role.isin(["train","calibration"])]
    models={}; calibrations={}; oof_frames=[]
    for excitation in ("avt","fvt"):
        modal,shape,calibration,oof=crossfit_excitation(healthy,excitation,args.alpha)
        models[excitation]=(modal,shape); calibrations[excitation]=calibration
        oof_frames.append(oof)
        print(f"{excitation}: healthy={calibration['healthy_records']} "
              f"folds={calibration['folds']} "
              f"modal_threshold={calibration['modal_threshold']:.6f} "
              f"shape_threshold={calibration['shape_threshold']:.6f} "
              f"oof_fpr={calibration['oof_false_alarm_rate']}",flush=True)
    output=[]; test=frame[frame.role=="test"]
    for index,row in enumerate(test.itertuples(),start=1):
        signals=load_vertical(Path(row.resolved_path)); excitation=row.excitation_resolved
        modal,shape=models[excitation]; calibration=calibrations[excitation]
        state,setup=record_metadata(str(row.record_id))
        for condition in CONDITIONS:
            repeats=1 if condition=="clean" or args.mode=="quick" else 3
            for replicate in range(repeats):
                changed,availability,_,signature=corrupt_v31(
                    signals,condition,str(row.record_id),replicate,modal)
                mr,sr=raw_scores(modal,shape,changed,availability)
                common={"record_id":row.record_id,"label":int(row.label),
                        "excitation":excitation,"state":state,"setup":setup,
                        "condition":condition,"replicate":replicate,
                        "corrupted_sensors":signature}
                for method,score in normalized_scores(mr,sr,calibration).items():
                    output.append({**common,"method":method,"score":score})
        if index%36==0: print(f"records={index}/{len(test)}",flush=True)
    scores=pd.DataFrame(output); metrics=metrics_table(scores); groups=damage_groups(scores)
    pd.concat(oof_frames,ignore_index=True).to_csv(
        args.output_dir/"healthy_oof_scores.csv",index=False)
    scores.to_csv(args.output_dir/"record_scores.csv",index=False)
    metrics.to_csv(args.output_dir/"metrics.csv",index=False)
    groups.to_csv(args.output_dir/"damage_group_metrics.csv",index=False)
    artifact={
        "protocol":"z24-healthy-only-crossfit-fusion-b1.1","manifest":str(manifest),
        "alpha":args.alpha,"mode":args.mode,"selection_uses_damage":False,
        "threshold_uses_damage":False,
        "methods_are_reported_without_posthoc_selection":True,
        "calibrations":calibrations,
        "modal_models":{k:models[k][0].as_dict() for k in models},
        "shape_models":{k:models[k][1].as_dict() for k in models},
    }
    (args.output_dir/"model.json").write_text(json.dumps(artifact,indent=2))
    clean=metrics[metrics.condition=="clean"].sort_values(
        ["scope","excitation","balanced_accuracy"],ascending=[True,True,False])
    targeted=metrics[
        metrics.condition.isin(["targeted_0.30","targeted_0.50"])&
        (metrics.scope=="overall")
    ].sort_values(["condition","balanced_accuracy"],ascending=[True,False])
    print("\n=== HEALTHY OOF CALIBRATION ===")
    print(json.dumps(calibrations,indent=2))
    print("\n=== CLEAN ==="); print(clean.to_string(index=False))
    print("\n=== TARGETED ==="); print(targeted.to_string(index=False))
    print("\n=== WEAKEST DAMAGE GROUPS ===")
    print(groups.sort_values(["recall","score_median"]).head(30).to_string(index=False))
    print(f"CROSSFIT_FUSION_B11_PASS output={args.output_dir}")

if __name__=="__main__": main()
