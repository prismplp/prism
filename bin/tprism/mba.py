from importlib import import_module
import json
from types import ModuleType
from typing import Any, Dict, Iterable, List, MutableMapping, Sequence, Tuple

import numpy as np
from tprism.expl_print import node2str, sw2str, remap_index
from tprism.expl_print import print_expl

try:
    legendre_decomp: ModuleType | None = import_module("legendre_decomp")
    print("[legendre_decomp] enabled")
except ModuleNotFoundError:
    legendre_decomp = None
    print(
        "[legendre_decomp] disabled (Installation: pip install git+https://github.com/kojima-r/pyLegendreDecomposition.git)"
    )


Goals = List[Dict[str, Any]]
MappingIndex = MutableMapping[Any, List[str]]
MappingValues = MutableMapping[str, int]



# e.g. [2, 0, 1, 5, 3, 4]<= [(0, 1, 2, 3), (4, 5)]
# local remap:[0, 1, 2, 3, 4, 5]<=[(1, 2, 0, 4), (5, 3)]
def local_index_remap(out_index,in_index_list):
  local_mapping={e:i for i,e in enumerate(out_index)}
  new_in_index_list=[]
  for in_index in in_index_list:
    new_in_index=[local_mapping[e] for e in in_index]
    new_in_index_list.append(tuple(new_in_index))
  return new_in_index_list

def run_subgoal(g, X_subgoal,mapping_index,n_iter=1000, lr=0.5, gpu=False,verbose=False,verbose_ld=False):
  verb_index=False
  if legendre_decomp is None:
    raise RuntimeError("legendre_decomp module is not available.")
  gg=g["node"]
  gI= gg["new_index"]
  #print(node2str2(gg, mapping_index,mapping_values),"<=>")
  
  out={}
  recons={}
  if len(g["paths"])>0:
    in_components=[]
    node_lists=[]
    for path in g["paths"]:
      arr1=[tuple(node['new_index']) for node in path["nodes"]]
      arr2=[tuple(sw['new_index']) for sw in path["tensor_switches"]]
      arr1s=[node2str(node, verb_index, mapping_index) for node in path["nodes"]]
      arr2s=[sw2str(sw, verb_index) for sw in path["tensor_switches"]]
      I=arr1+arr2
      if verbose:
        print(tuple(gI),"=>",I)
      I2=local_index_remap(gI,I)
      #print([i for i in range(len(gI))],"=>",I2)
      ldc=legendre_decomp.LDComponent(I2)
      ldc.node=arr1s
      ldc.sw=arr2s
      in_components.append(ldc)
      node_lists.append(arr1s+arr2s)
    # LD
    all_history_kl, scaleX, P, Q, components =legendre_decomp.MixLD_MBA(
        X_subgoal,in_components,n_iter=n_iter, lr=lr, gpu=gpu, error_tol=0.00001, verbose=verbose_ld)
    X_outs=[]
    Q2=np.zeros_like(components[0].Q)
    for comp in components:
      X_out=legendre_decomp.module_mba.compute_nbody(comp.theta,X_subgoal.shape,I_x=comp.I,gpu=False)
      Q2_=legendre_decomp.module_mba.recons_nbody(X_out, len(X_subgoal.shape),gpu=False)
      X_outs.append(X_out)
      Q2+=Q2_*comp.pi
    goal_s=node2str(gg, verb_index, mapping_index)
    recons[goal_s]=(Q, Q2, scaleX, components)
    if verbose_ld:
      #MAE between P and Q
      print("P-Q",np.mean(np.abs(Q-P)))
      #MAE between X and Q
      print("X-Q",np.mean(np.abs(scaleX*Q-X_subgoal)))
      #MAE between X and Q2
      print("X-Q2",np.mean(np.abs(scaleX*Q2-X_subgoal)))
      #MAE between Q2-Q
      print("Q-Q2",np.mean(np.abs(scaleX*(Q2-Q))))
    ###
    for c, comp in enumerate(components):
      for i,(k,x) in enumerate(X_outs[c]):
        node_s=node_lists[c][i]
        if verbose:
          print(node_s,":",k,"=> shape:",x.shape)
        out[node_s]=x
    return out,recons
  return out,recons

def run_mba(goals, X_goal, n_iter=1000, lr=0.5, gpu=False, verbose=False, verbose_ld=False):
  mapping_index =remap_index(goals)
  goal_dict={}
  recons_dict={}
  verb_index=False
  for g in goals[::-1]:
    gg=g["node"]
    g_node_s=node2str(gg,False, None)
    if verbose:      
      print("==== "+g_node_s+" ====")
    if g_node_s in goal_dict:
      X_subgoal=goal_dict[g_node_s]
    else:
      X_subgoal=X_goal
    o,recons=run_subgoal(g, X_subgoal,mapping_index,n_iter, lr, gpu,verbose,verbose_ld)
    goal_dict.update(o)
    recons_dict.update(recons)
  return goal_dict,recons_dict

def transpose(X, src_index, to_index):
  mapping={e:i for i, e in enumerate(src_index)}
  trans=[mapping[e] for e in to_index]
  print(trans)
  return np.transpose(X, trans)
  
def run(X, X_index, expl_filename, n_iter=1000, lr=0.5, gpu=False, verbose=False,verbose_expl=False, verbose_ld=False, verbose_recons=False):
  obj=json.load(open(expl_filename))
  goals=obj['goals']
  if verbose_expl:
    print_expl(goals,verb_index=True)
    print("===")
  mapping_index =remap_index(goals)
  # input transpose
  expl_index=goals[-1]['node']['new_index']
  X_ = transpose(X, X_index, expl_index)
  # run MBA
  if verbose_expl:
    print_expl(goals,True,mapping_index)
  goal_dict, recons_dict=run_mba(goals, X_, n_iter=n_iter, lr=lr, gpu=gpu, verbose=verbose, verbose_ld=verbose_ld)
  recons_out, last_node=recons(goals, recons_dict, verbose=verbose_recons)
  # output transpose
  recons_index, recons_X=recons_out[last_node]  
  X_ = transpose(recons_X, recons_index, X_index)
  recons_out[last_node]=(X_index, X_)
  return goal_dict,recons_dict,recons_out


def recons(goals, recons_dict, verbose=False):
  if legendre_decomp is None:
    raise RuntimeError("legendre_decomp module is not available.")
  recons_out={}
  last_g_node_s=None
  for g in goals:
      gg=g["node"]
      gI=gg["new_index"]
      g_node_s=node2str(gg,False, None)
      last_g_node_s=g_node_s
      Q,Q2,scaleX,components=recons_dict[g_node_s]
      S=Q.shape
      if verbose:
        print(g_node_s,"  shape =",S,"  index=",gI)
      Q_out=np.zeros_like(components[0].Q)
      for comp in components:
          X_out=legendre_decomp.module_mba.compute_nbody(comp.theta,S,I_x=comp.I,gpu=False)
          for i,n in enumerate(comp.node):
              s,x=X_out[i]
              _, new_x=recons_out[n]
              if len(s)!=len(new_x.shape):
                  print(s,n,new_x.shape)
              X_out[i]=(s, new_x)
          Q_out_=legendre_decomp.module_mba.recons_nbody(X_out, len(S),gpu=False)
          Q_out+=Q_out_*comp.pi
          if verbose:
            print(">>",comp.theta.shape )
            print("  ",comp.I )
            print("  ",comp.node+comp.sw)
            print("  ",[(s,"shape={}".format(x.shape)) for s, x in X_out])
      #print(Q_out)
      recons_out[g_node_s]=(tuple(gI), Q_out*scaleX)
  return recons_out, last_g_node_s

