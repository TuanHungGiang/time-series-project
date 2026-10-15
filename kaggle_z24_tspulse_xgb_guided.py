# %% [markdown]
# # Z24 — XGBoost-guided TSPulse trên Kaggle
#
# Notebook tự tải Z24 từ Hugging Face và TSPulse từ IBM. Quy trình đánh giá giữ
# nguyên measurement record: 153 records = 17 damage states × 9 setups, mỗi
# record có 10 segments. Không chia ngẫu nhiên 1.530 segments.
#
# **Kaggle:** bật Internet và chọn GPU accelerator. Code tự nhận số GPU; nếu có
# nhiều T4, `DataParallel` chỉ được dùng khi trích embedding. XGBoost dùng GPU 0.

# %%
# Cài dependency nhưng không thay torch/CUDA có sẵn của Kaggle.
import os, subprocess, sys
from pathlib import Path

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
GRANITE = WORK / "granite-tsfm"
if not GRANITE.exists():
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/ibm-granite/granite-tsfm.git", str(GRANITE)],
        check=True,
    )
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "datasets>=4,<5", "transformers>=4.57.6,<6",
     "einops>=0.7", "xgboost>=3,<4", "huggingface_hub>=0.25", "scikit-learn>=1.4"],
    check=True,
)
sys.path.insert(0, str(GRANITE))

# %% [markdown]
# ## 1. Cấu hình
#
# `PILOT_SETUPS=[0,3,6]` là chạy pilot nhanh. Đổi thành `list(range(9))` sau khi
# pilot hợp lý. `FORCE_REBUILD=False` sẽ tái sử dụng feature/embedding/teacher.

# %%
import gc, hashlib, json, math, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader, Dataset
from xgboost import DMatrix, XGBClassifier

SEED = 42
PILOT_SETUPS = [0, 3, 6]       # full CV: list(range(9))
VARIANTS = ["tspulse", "physical", "xgb_residual"]
TOP_FEATURES = 60
TEACHER_TREES = 250
EPOCHS = 100                    # head-only nên chạy nhanh
PATIENCE = 15
HEAD_BATCH = 128
EMBED_BATCH_PER_GPU = 32
NUM_WORKERS = min(4, os.cpu_count() or 1)
FORCE_REBUILD = False

DATA_DIR = WORK / "z24_hf"
CACHE_DIR = WORK / "z24_tspulse_cache"
OUT_DIR = WORK / "z24_tspulse_xgb_results"
for path in (DATA_DIR, CACHE_DIR, OUT_DIR): path.mkdir(parents=True, exist_ok=True)

def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

seed_all()
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
print("torch:", torch.__version__, "GPUs:", torch.cuda.device_count(),
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

# %% [markdown]
# ## 2. Tải và kiểm tra dataset
#
# Chỉ tải hai file cần thiết. `inputs.npy` phải có SHA-256 đúng với file trên
# Hugging Face và shape `(1530,27,6000)`.

# %%
REPO_ID = "thanglexuan/Z24-dataset-processed"
inputs_path = Path(hf_hub_download(REPO_ID, "inputs.npy", repo_type="dataset", local_dir=DATA_DIR))
labels_path = Path(hf_hub_download(REPO_ID, "labels.npy", repo_type="dataset", local_dir=DATA_DIR))

def sha256(path, block=8 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(block): digest.update(chunk)
    return digest.hexdigest()

expected = "56fb5923433e4b47cbd9c19ce079fc098f59d501b3385f9cb26636cc3928ef6e"
actual = sha256(inputs_path)
assert actual == expected, (actual, expected)
signals = np.load(inputs_path, mmap_mode="r")
original_labels = np.load(labels_path).astype(np.int64)
label_values = np.unique(original_labels)
labels = np.searchsorted(label_values, original_labels).astype(np.int64)
assert signals.shape == (1530, 27, 6000) and labels.shape == (1530,)

sample_id = np.arange(1530)
record_id = sample_id // 10
setup = record_id % 9
assert len(np.unique(record_id)) == 153
assert np.all([np.unique(labels[record_id == r]).size == 1 for r in np.unique(record_id)])
print("signals", signals.shape, signals.dtype, "records", len(np.unique(record_id)),
      "classes", np.unique(labels).size, "setups", np.unique(setup).size)

# %% [markdown]
# ## 3. Trích 405 physical features
#
# Mỗi sensor có 15 feature: 8 time-domain và 7 frequency-domain. Feature dùng
# crop giữa 1.024 điểm; TSPulse dùng crop giữa 512 điểm từ cùng segment.

# %%
from scipy.stats import kurtosis, skew

FS = 100.0
FEATURE_CACHE = CACHE_DIR / "physical_405.npz"

def extract_physical(window):
    x = np.asarray(window, dtype=np.float64)
    eps = 1e-12
    std = x.std(1); rms = np.sqrt(np.mean(x*x, 1)); maxabs = np.max(np.abs(x), 1)
    time_f = np.column_stack([x.mean(1), std, rms, np.mean(np.abs(x),1), np.ptp(x,axis=1),
                              np.nan_to_num(skew(x,axis=1,bias=False)),
                              np.nan_to_num(kurtosis(x,axis=1,fisher=True,bias=False)),
                              maxabs/np.maximum(rms,eps)])
    freq = np.fft.rfftfreq(x.shape[1], 1/FS); power = np.abs(np.fft.rfft(x,axis=1))**2
    valid = (freq>=.5)&(freq<=40); vp=power[:,valid]; vf=freq[valid]
    total=vp.sum(1,keepdims=True)+eps; norm=vp/total
    dom=vf[np.argmax(vp,1)]; centroid=(norm*vf).sum(1)
    entropy=-(norm*np.log(norm+eps)).sum(1)/np.log(vp.shape[1])
    bands=[(.5,5),(5,10),(10,20),(20,40.01)]
    rel=np.column_stack([power[:,(freq>=lo)&(freq<hi)].sum(1)/total[:,0] for lo,hi in bands])
    return np.concatenate([time_f,np.column_stack([dom,centroid,entropy,rel])],1).reshape(-1).astype("float32")

if FEATURE_CACHE.exists() and not FORCE_REBUILD:
    physical = np.load(FEATURE_CACHE)["X"]
else:
    physical = np.empty((1530, 405), dtype="float32")
    hann = np.hanning(1024).astype("float32")[None, :]
    left = (6000-1024)//2
    for i in range(1530):
        physical[i] = extract_physical(np.asarray(signals[i,:,left:left+1024],dtype="float32")*hann)
        if (i+1)%255==0: print("physical",i+1,"/ 1530")
    np.savez_compressed(FEATURE_CACHE, X=physical)
assert np.isfinite(physical).all()
print("physical", physical.shape)

# %% [markdown]
# ## 4. Precompute TSPulse embedding bằng tất cả GPU
#
# Backbone pretrained bị đóng băng và chỉ chạy một lần. Embedding FP16 khoảng
# 274 MB, sau đó các ablation/fold huấn luyện rất nhanh và ổn định hơn với 153
# records. Đây cũng tránh lặp backbone 9 × 3 lần.

# %%
from tsfm_public.models.tspulse import TSPulseForReconstruction

EMBED_CACHE = CACHE_DIR / "tspulse_embeddings_fp16.npy"

class SignalDataset(Dataset):
    def __init__(self, array): self.array=array; self.left=(6000-512)//2; self.hann=np.hanning(512).astype("float32")[:,None]
    def __len__(self): return len(self.array)
    def __getitem__(self,i):
        x=np.array(self.array[i,:,self.left:self.left+512].T,dtype="float32",copy=True)
        return torch.from_numpy(x*self.hann)

class BackboneTensor(nn.Module):
    def __init__(self, backbone): super().__init__(); self.backbone=backbone
    def forward(self,x):
        return self.backbone(x,return_dict=True,output_hidden_states=False,enable_masking=False).last_hidden_flatten_state

if EMBED_CACHE.exists() and not FORCE_REBUILD:
    embeddings=np.load(EMBED_CACHE,mmap_mode="r")
else:
    assert torch.cuda.is_available(), "Hãy bật GPU Accelerator trong Kaggle"
    pretrained=TSPulseForReconstruction.from_pretrained(
        "ibm-granite/granite-timeseries-tspulse-r1", revision="tspulse-block-dualhead-512-p16-r1")
    net=BackboneTensor(pretrained.backbone).eval().cuda()
    for p in net.parameters(): p.requires_grad=False
    ngpu=max(1,torch.cuda.device_count())
    if ngpu>1: net=nn.DataParallel(net,device_ids=list(range(ngpu)))
    loader=DataLoader(SignalDataset(signals),batch_size=EMBED_BATCH_PER_GPU*ngpu,shuffle=False,
                      num_workers=NUM_WORKERS,pin_memory=True,persistent_workers=NUM_WORKERS>0)
    chunks=[]
    with torch.inference_mode():
        for step,x in enumerate(loader):
            with torch.autocast("cuda",dtype=torch.float16): z=net(x.cuda(non_blocking=True))
            chunks.append(z.cpu().half().numpy())
            if (step+1)%5==0: print("embedding batch",step+1,"/",len(loader))
    embeddings=np.concatenate(chunks)
    np.save(EMBED_CACHE,embeddings)
    del net,pretrained,chunks; gc.collect(); torch.cuda.empty_cache()
    embeddings=np.load(EMBED_CACHE,mmap_mode="r")
assert embeddings.shape[:2] == (1530,27)
print("embeddings",embeddings.shape,embeddings.dtype,round(EMBED_CACHE.stat().st_size/2**20,1),"MB")

# %% [markdown]
# ## 5. XGBoost SHAP ranking và OOF teacher
#
# Mỗi outer fold chỉ sử dụng 7 setup train để tính ranking. Teacher logits cho
# train được tạo bằng leave-one-setup-out; validation/test do model fit trên 7
# setup train dự đoán. Vì vậy không có in-sample teacher leakage.

# %%
def xgb_model(trees,seed,n_classes=17):
    return XGBClassifier(n_estimators=trees,max_depth=4,learning_rate=.05,subsample=.8,
        colsample_bytree=.8,min_child_weight=3,reg_lambda=2,reg_alpha=.05,
        objective="multi:softprob",num_class=n_classes,eval_metric="mlogloss",
        tree_method="hist",device="cuda" if torch.cuda.is_available() else "cpu",
        random_state=seed,n_jobs=max(1,(os.cpu_count() or 2)//2))

def native_shap(model,X,max_rows=512):
    idx=np.linspace(0,len(X)-1,min(len(X),max_rows),dtype=int)
    c=model.get_booster().predict(DMatrix(X[idx]),pred_contribs=True)
    c=np.asarray(c)[...,:-1]
    return np.mean(np.abs(c),axis=tuple(range(c.ndim-1)))

def fold_artifacts(test_setup):
    val_setup=(test_setup+1)%9
    train=~np.isin(setup,[test_setup,val_setup]); val=setup==val_setup; test=setup==test_setup
    cache=CACHE_DIR/f"fold_{test_setup}_xgb_t{TEACHER_TREES}.npz"
    if cache.exists() and not FORCE_REBUILD:
        d=np.load(cache); return train,val,test,d["ranking"],d["teacher"]
    damage=xgb_model(TEACHER_TREES,SEED+test_setup); damage.fit(physical[train],labels[train])
    setups_train=np.unique(setup[train]); mapping={v:i for i,v in enumerate(setups_train)}
    setup_y=np.array([mapping[v] for v in setup[train]])
    shortcut=xgb_model(TEACHER_TREES,SEED+100+test_setup,len(setups_train)); shortcut.fit(physical[train],setup_y)
    d_imp=native_shap(damage,physical[train]); s_imp=native_shap(shortcut,physical[train])
    d_imp/=max(np.median(d_imp[d_imp>0]),1e-12); s_imp/=max(np.median(s_imp[s_imp>0]),1e-12)
    ranking=np.argsort(d_imp/(1+s_imp))[::-1]
    teacher=np.full((1530,17),np.nan,dtype="float32")
    for held in setups_train:
        itr=train&(setup!=held); iva=train&(setup==held)
        m=xgb_model(TEACHER_TREES,SEED+1000+test_setup*10+int(held)); m.fit(physical[itr],labels[itr])
        teacher[iva]=m.predict_proba(physical[iva]); del m
    teacher[val|test]=damage.predict_proba(physical[val|test])
    assert np.isfinite(teacher).all()
    np.savez_compressed(cache,ranking=ranking,teacher=teacher)
    return train,val,test,ranking,teacher

# %% [markdown]
# ## 6. Head fusion và đánh giá cấp recording

# %%
class HeadDataset(Dataset):
    def __init__(self,idx,phys,teacher): self.idx=np.flatnonzero(idx); self.phys=phys; self.teacher=teacher
    def __len__(self): return len(self.idx)
    def __getitem__(self,k):
        i=self.idx[k]; p=np.clip(self.teacher[i],1e-7,1); logp=np.log(p); logp-=logp.mean()
        return (torch.from_numpy(np.asarray(embeddings[i],dtype="float32")),
                torch.from_numpy(self.phys[i]),torch.from_numpy(logp.astype("float32")),
                torch.tensor(labels[i]).long(),torch.tensor(record_id[i]).long())

class FusionHead(nn.Module):
    def __init__(self,variant,prior):
        super().__init__(); self.variant=variant
        prior=torch.tensor(prior).clamp(1e-4,1-1e-4); self.register_buffer("prior",prior)
        self.gate_logits=nn.Parameter(torch.logit(prior)); h=embeddings.shape[-1]
        self.wave=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,64),nn.GELU(),nn.Dropout(.25))
        self.base=nn.Linear(64,17)
        self.phys=nn.Sequential(nn.LayerNorm(TOP_FEATURES),nn.Linear(TOP_FEATURES,64),nn.GELU(),nn.Dropout(.25))
        self.residual=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Dropout(.25),nn.Linear(64,17))
        self.conf=nn.Sequential(nn.Linear(65,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,z,p,t):
        gates=torch.sigmoid(self.gate_logits)
        pooled=z.mean(1) if self.variant=="tspulse" else (z*gates[None,:,None]).sum(1)/gates.sum()
        w=self.wave(pooled); dl=self.base(w)
        if self.variant!="tspulse": dl=dl+self.residual(torch.cat([w,self.phys(p)],1))
        if self.variant=="xgb_residual":
            tc=t.softmax(1).amax(1,keepdim=True); alpha=torch.sigmoid(self.conf(torch.cat([w,tc],1)))
            final=t+alpha*dl
        else: final=dl; alpha=torch.ones((len(z),1),device=z.device)
        return final,dl,gates,alpha

def record_metrics(prob,y,rid):
    ids=np.unique(rid); rp=[]; ry=[]
    for r in ids:
        m=rid==r; rp.append(prob[m].mean(0)); ry.append(np.unique(y[m])[0])
    rp=np.asarray(rp); ry=np.asarray(ry); pred=rp.argmax(1)
    return {"record_accuracy":float(accuracy_score(ry,pred)),
            "record_macro_f1":float(f1_score(ry,pred,average="macro",zero_division=0)),
            "record_mcc":float(matthews_corrcoef(ry,pred))}, (ids,ry,pred,rp)

@torch.no_grad()
def evaluate(model,loader):
    model.eval(); ps=[]; ys=[]; rs=[]
    for z,p,t,y,r in loader:
        logits,*_=model(z.cuda(non_blocking=True),p.cuda(non_blocking=True),t.cuda(non_blocking=True))
        ps.append(logits.softmax(1).cpu().numpy()); ys.append(y.numpy()); rs.append(r.numpy())
    return record_metrics(np.concatenate(ps),np.concatenate(ys),np.concatenate(rs))

def train_one(variant,train,val,test,ranking,teacher,test_setup):
    chosen=ranking[:TOP_FEATURES]; mu=physical[train][:,chosen].mean(0); sd=physical[train][:,chosen].std(0); sd[sd<1e-8]=1
    phys=np.clip((physical[:,chosen]-mu)/sd,-10,10).astype("float32")
    rankscore=np.empty(405); rankscore[ranking]=np.linspace(1,0,405,endpoint=False)
    sensor=rankscore.reshape(27,15).mean(1); sensor=(sensor-sensor.min())/max(np.ptp(sensor),1e-12); prior=.2+.6*sensor
    loaders=[DataLoader(HeadDataset(mask,phys,teacher),batch_size=HEAD_BATCH,shuffle=(j==0),
             num_workers=0,pin_memory=True) for j,mask in enumerate([train,val,test])]
    seed_all(SEED+test_setup); model=FusionHead(variant,prior).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
    best=None; best_score=(-1,-1); stale=0; history=[]
    for epoch in range(EPOCHS):
        model.train(); losses=[]
        for z,p,t,y,_ in loaders[0]:
            z,p,t,y=[q.cuda(non_blocking=True) for q in (z,p,t,y)]; opt.zero_grad(set_to_none=True)
            final,dl,gates,_=model(z,p,t); loss=F.cross_entropy(final,y)
            if variant=="xgb_residual": loss=loss+.2*F.cross_entropy(dl,y)
            if variant!="tspulse" and epoch<EPOCHS//2: loss=loss+.01*F.mse_loss(gates,model.prior)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); losses.append(loss.item())
        vm,_=evaluate(model,loaders[1]); score=(vm["record_macro_f1"],vm["record_accuracy"])
        history.append({"epoch":epoch+1,"loss":float(np.mean(losses)),**vm})
        if score>best_score: best_score=score; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else: stale+=1
        if stale>=PATIENCE: break
    model.load_state_dict(best); tm,pred=evaluate(model,loaders[2])
    return {"variant":variant,"validation":{k:best_score[i] for i,k in enumerate(["record_macro_f1","record_accuracy"])},
            "test":tm,"epochs":len(history),"history":history,"selected_features":chosen.tolist(),
            "parameters":sum(p.numel() for p in model.parameters())},pred

# %% [markdown]
# ## 7. Chạy pilot
#
# Kết quả smoke/pilot không được dùng để chọn setup thuận lợi. Nếu quyết định
# chuyển sang 9-fold, giữ nguyên hyperparameter rồi đổi `PILOT_SETUPS` ở đầu.

# %%
all_folds=[]; started=time.time()
for test_setup in PILOT_SETUPS:
    train,val,test,ranking,teacher=fold_artifacts(test_setup)
    fold={"test_setup":test_setup,"validation_setup":(test_setup+1)%9,"results":[]}
    for variant in VARIANTS:
        result,pred=train_one(variant,train,val,test,ranking,teacher,test_setup)
        fold["results"].append(result)
        np.savez_compressed(OUT_DIR/f"setup_{test_setup}_{variant}.npz",record_id=pred[0],y_true=pred[1],y_pred=pred[2],probabilities=pred[3])
        print("setup",test_setup,variant,result["test"])
    all_folds.append(fold)
    (OUT_DIR/"progress.json").write_text(json.dumps(all_folds,indent=2))

aggregate={}
for variant in VARIANTS:
    rows=[r for f in all_folds for r in f["results"] if r["variant"]==variant]
    aggregate[variant]={m:{"mean":float(np.mean([r["test"][m] for r in rows])),
                            "std":float(np.std([r["test"][m] for r in rows],ddof=1)) if len(rows)>1 else 0.0}
                        for m in ["record_accuracy","record_macro_f1","record_mcc"]}
report={"protocol":"setup holdout; recording-level evaluation","test_setups":PILOT_SETUPS,
        "transductive":False,"backbone":"TSPulse 2025 frozen pretrained","teacher_trees":TEACHER_TREES,
        "folds":all_folds,"aggregate":aggregate,"elapsed_minutes":(time.time()-started)/60}
(OUT_DIR/"report.json").write_text(json.dumps(report,indent=2))
print(json.dumps(aggregate,indent=2)); print("saved:",OUT_DIR/"report.json")

# %% [markdown]
# ## 8. Tải kết quả về máy

# %%
import shutil
archive=shutil.make_archive(str(WORK/"z24_tspulse_xgb_results"),"zip",OUT_DIR)
print("Kaggle Output:",archive)
