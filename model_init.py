# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:34:37 2024

@author: aszor
"""

from models import CombinedSpatiotemporalModel, JumpCNFSpatiotemporalModel, SelfAttentiveCNFSpatiotemporalModel, JumpGMMSpatiotemporalModel
from models.spatial import GaussianMixtureSpatialModel, IndependentCNF, JumpCNF, SelfAttentiveCNF
from models.spatial.cnf import TimeVariableCNF
from models.temporal import HomogeneousPoissonPointProcess, HawkesPointProcess, SelfCorrectingPointProcess, NeuralPointProcess
from models.temporal.neural import ACTFNS as TPP_ACTFNS
from models.temporal.neural import TimeVariableODE
import torch
import utils
from viz_dataset import MAPS
import toy_datasets
import argparse
import json

def get_t0_t1(data):
    if data == "waldo":
        return torch.tensor([0.0]), torch.tensor([180.0])
    elif data == "waldo_short":
        return torch.tensor([0.0]), torch.tensor([30.1])
    elif data == "waldo_shorter":
        return torch.tensor([0.0]), torch.tensor([15.05])
    elif data == "citibike":
        return torch.tensor([0.0]), torch.tensor([24.0])
    elif data == "covid_nj_cases":
        return torch.tensor([0.0]), torch.tensor([7.0])
    elif data == "earthquakes_jp":
        return torch.tensor([0.0]), torch.tensor([30.0])
    elif data == "pinwheel":
        return torch.tensor([0.0]), torch.tensor([toy_datasets.END_TIME])
    elif data == "gmm":
        return torch.tensor([0.0]), torch.tensor([toy_datasets.END_TIME])
    elif data == "fmri":
        return torch.tensor([0.0]), torch.tensor([10.0])
    else:
        raise ValueError(f"Unknown dataset {data}")


def model_init(args,x_dim,device):
    if args.model == "jumpcnf" and args.tpp == "neural":
        model = JumpCNFSpatiotemporalModel(dim=x_dim,
                                           hidden_dims=list(map(int, args.hdims.split("-"))),
                                           tpp_hidden_dims=list(map(int, args.tpp_hdims.split("-"))),
                                           actfn=args.actfn,
                                           tpp_cond=args.tpp_cond,
                                           tpp_style=args.tpp_style,
                                           tpp_actfn=args.tpp_actfn,
                                           share_hidden=args.share_hidden,
                                           solve_reverse=args.solve_reverse,
                                           tol=args.tol,
                                           otreg_strength=args.otreg_strength,
                                           tpp_otreg_strength=args.tpp_otreg_strength,
                                           layer_type=args.layer_type,
                                           beta=args.beta,
                                           ).to(device)
    elif args.model == "attncnf" and args.tpp == "neural":
        model = SelfAttentiveCNFSpatiotemporalModel(dim=x_dim,
                                                    hidden_dims=list(map(int, args.hdims.split("-"))),
                                                    tpp_hidden_dims=list(map(int, args.tpp_hdims.split("-"))),
                                                    actfn=args.actfn,
                                                    tpp_cond=args.tpp_cond,
                                                    tpp_style=args.tpp_style,
                                                    tpp_actfn=args.tpp_actfn,
                                                    share_hidden=args.share_hidden,
                                                    solve_reverse=args.solve_reverse,
                                                    l2_attn=args.l2_attn,
                                                    tol=args.tol,
                                                    otreg_strength=args.otreg_strength,
                                                    tpp_otreg_strength=args.tpp_otreg_strength,
                                                    layer_type=args.layer_type,
                                                    lowvar_trace=not args.naive_hutch,
                                                    beta=args.beta,
                                                    ).to(device)
    elif args.model == "cond_gmm" and args.tpp == "neural":
        model = JumpGMMSpatiotemporalModel(dim=x_dim,
                                           hidden_dims=list(map(int, args.hdims.split("-"))),
                                           tpp_hidden_dims=list(map(int, args.tpp_hdims.split("-"))),
                                           actfn=args.actfn,
                                           tpp_cond=args.tpp_cond,
                                           tpp_style=args.tpp_style,
                                           tpp_actfn=args.tpp_actfn,
                                           share_hidden=args.share_hidden,
                                           tol=args.tol,
                                           tpp_otreg_strength=args.tpp_otreg_strength,
                                           ).to(device)
    else:
        # Mix and match between spatial and temporal models.
        if args.tpp == "poisson":
            tpp_model = HomogeneousPoissonPointProcess()
        elif args.tpp == "hawkes":
            tpp_model = HawkesPointProcess()
        elif args.tpp == "correcting":
            tpp_model = SelfCorrectingPointProcess()
        elif args.tpp == "neural":
            tpp_hidden_dims = list(map(int, args.tpp_hdims.split("-")))
            tpp_model = NeuralPointProcess(
                cond_dim=x_dim, hidden_dims=tpp_hidden_dims, cond=args.tpp_cond, style=args.tpp_style, actfn=args.tpp_actfn,
                otreg_strength=args.tpp_otreg_strength, tol=args.tol)
        else:
            raise ValueError(f"Invalid tpp model {args.tpp}")

        if args.model == "gmm":
            model = CombinedSpatiotemporalModel(GaussianMixtureSpatialModel(), tpp_model).to(device)
        elif args.model == "cnf":
            model = CombinedSpatiotemporalModel(
                IndependentCNF(dim=x_dim, hidden_dims=list(map(int, args.hdims.split("-"))),
                               layer_type=args.layer_type, actfn=args.actfn, tol=args.tol, otreg_strength=args.otreg_strength,
                               squash_time=True, beta=args.beta),
                tpp_model).to(device)
        elif args.model == "tvcnf":
            model = CombinedSpatiotemporalModel(
                IndependentCNF(dim=x_dim, hidden_dims=list(map(int, args.hdims.split("-"))),
                               layer_type=args.layer_type, actfn=args.actfn, tol=args.tol, otreg_strength=args.otreg_strength, beta=args.beta),
                tpp_model).to(device)
        elif args.model == "jumpcnf":
            model = CombinedSpatiotemporalModel(
                JumpCNF(dim=x_dim, hidden_dims=list(map(int, args.hdims.split("-"))),
                        layer_type=args.layer_type, actfn=args.actfn, tol=args.tol, otreg_strength=args.otreg_strength, beta=args.beta),
                tpp_model).to(device)
        elif args.model == "attncnf":
            model = CombinedSpatiotemporalModel(
                SelfAttentiveCNF(dim=x_dim, hidden_dims=list(map(int, args.hdims.split("-"))),
                                 layer_type=args.layer_type, actfn=args.actfn, l2_attn=args.l2_attn, tol=args.tol, otreg_strength=args.otreg_strength, beta=args.beta),
                tpp_model).to(device)
        else:
            raise ValueError(f"Invalid model {args.model}")

    params = []
    attn_params = []
    for name, p in model.named_parameters():
        if "self_attns" in name:
            attn_params.append(p)
        else:
            params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": params},
        {"params": attn_params}
    ], lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))

    ema = utils.ExponentialMovingAverage(model)

    return model,optimizer,ema

def get_model_args(argfile):
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    with open(argfile, 'r') as f:
       args.__dict__ = json.load(f)
    return args  

def get_defaults(parser):
    parser.add_argument("--data", type=str, choices=MAPS.keys(), default="earthquakes_jp")
    
    parser.add_argument("--model", type=str, choices=["cond_gmm", "gmm", "cnf", "tvcnf", "jumpcnf", "attncnf"], default="attncnf")
    parser.add_argument("--tpp", type=str, choices=["poisson", "hawkes", "correcting", "neural"], default="neural")
    parser.add_argument("--actfn", type=str, default="swish")
    parser.add_argument("--tpp_actfn", type=str, choices=TPP_ACTFNS.keys(), default="softplus")
    parser.add_argument("--hdims", type=str, default="64-64-64")
    parser.add_argument("--layer_type", type=str, choices=["concat", "concatsquash"], default="concat")
    parser.add_argument("--tpp_hdims", type=str, default="32-32")
    parser.add_argument("--tpp_nocond", action="store_false", dest='tpp_cond')
    parser.add_argument("--tpp_style", type=str, choices=["split", "simple", "gru"], default="gru")
    parser.add_argument("--no_share_hidden", action="store_false", dest='share_hidden')
    parser.add_argument("--solve_reverse", action="store_true")
    parser.add_argument("--naive_hutch", action="store_true")
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--otreg_strength", type=float, default=1e-4)
    parser.add_argument("--tpp_otreg_strength", type=float, default=1e-4)

    parser.add_argument("--warmup_itrs", type=int, default=0)
    parser.add_argument("--num_iterations", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--gradclip", type=float, default=0)
    parser.add_argument("--max_events", type=int, default=4000)
    parser.add_argument("--test_bsz", type=int, default=32)

    parser.add_argument("--experiment_dir", type=str, default="experiments")
    parser.add_argument("--experiment_id", type=str, default=None)
    parser.add_argument("--ngpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--logfreq", type=int, default=10)
    parser.add_argument("--testfreq", type=int, default=100)
    parser.add_argument("--beta", type=int, default=2)
    parser.add_argument("--port", type=int, default=None)

    return parser