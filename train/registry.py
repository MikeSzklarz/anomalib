"""Model and dataset name -> class mappings for the --model/--dataset CLI flags."""

from typing import Dict, Type

from anomalib.data import (
    MVTecAD, MVTecLOCO, MVTecAD2, MVTec3D,
    BTech, Visa, Folder, Kolektor,
    Avenue, ShanghaiTech, UCSDped, AutoVI, Kaputt
)

from anomalib.models import (
    AnomalyDINO, AnomalyVFM, Cfa, Cflow, Csflow, Dfkde, Dfm,
    Dinomaly, Draem, Dsr, EfficientAd, Fastflow, FoundAD,
    Fre, Ganomaly, GeneralAD, Glass, InpFormer, L2BT, MHPatchcore, Padim, Patchcore, Patchflow,
    Rad, ReverseDistillation, Stfpm, SuperADD, Supersimplenet, Uflow, UniNet
)

MODEL_MAP: Dict[str, Type] = {
    "anomalydino": AnomalyDINO,
    "anomalyvfm": AnomalyVFM,
    "cfa": Cfa,
    "cflow": Cflow,
    "csflow": Csflow,
    "dfkde": Dfkde,
    "dfm": Dfm,
    "dinomaly": Dinomaly,
    "draem": Draem,
    "dsr": Dsr,
    "efficientad": EfficientAd,
    "fastflow": Fastflow,
    "foundad": FoundAD,
    "fre": Fre,
    "ganomaly": Ganomaly,
    "generalad": GeneralAD,
    "glass": Glass,
    "inpformer": InpFormer,
    "l2bt": L2BT,
    "mhpatchcore": MHPatchcore,
    "padim": Padim,
    "patchcore": Patchcore,
    "patchflow": Patchflow,
    "rad": Rad,
    "reversedistillation": ReverseDistillation,
    "stfpm": Stfpm,
    "superadd": SuperADD,
    "supersimplenet": Supersimplenet,
    "uflow": Uflow,
    "uninet": UniNet
}

DATASET_MAP: Dict[str, Type] = {
    "mvtecad": MVTecAD,
    "mvtecloco": MVTecLOCO,
    "mvtecad2": MVTecAD2,
    "mvtec3d": MVTec3D,
    "btech": BTech,
    "visa": Visa,
    "kolektor": Kolektor,
    "folder": Folder,
    "avenue": Avenue,
    "shanghaitech": ShanghaiTech,
    "ucsdped": UCSDped,
    "autovi": AutoVI,
    "kaputt": Kaputt
}
