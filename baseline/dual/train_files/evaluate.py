import os
import sys
import glob
import argparse
import pathlib
import numpy as np
import ecole
import json

import time as tmm

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_deviceS'] = ''
    device = "cpu"

    running_dir = 'train_files/trained_models/set_partitioning'
    #running_dir = 'train_files/trained_models/item_placement'

    # import pytorch **after** cuda setup
    import torch
    import torch.nn.functional as F
    import torch_geometric
    from utilities import log, pad_tensor, GraphDataset, Scheduler
    sys.path.insert(0,'.')
    from model import GNNPolicy

    policy = GNNPolicy().to(device)
    policy.load_state_dict(torch.load(pathlib.Path(running_dir)/'best_params.pkl', map_location=device))

    time_limit = 600
    scip_parameters = {
        "separating/maxrounds": 0,
        "presolving/maxrestarts": 0,
        "limits/time": time_limit,
    }
    env = ecole.environment.Branching(
        observation_function=ecole.observation.NodeBipartite(),
        information_function={
            "nb_nodes": ecole.reward.NNodes(),
            "time": ecole.reward.SolvingTime(),
        },
        scip_params=scip_parameters,
    )
    default_env = ecole.environment.Configuring(
        observation_function=None,
        information_function={
            "nb_nodes": ecole.reward.NNodes(),
            "time": ecole.reward.SolvingTime(),
        },
        scip_params=scip_parameters,
    )

    records = {}

    instances = sorted(glob.glob('../../instances/set_partitioning/train/*.mps.gz'))
    #instances = sorted(glob.glob('../../instances/1_item_placement/valid/*.mps.gz'))
    for inst_cnt, instance in enumerate(instances):
        inst_name = instance.rpartition('/')[-1]

        print(inst_cnt, instance)
        sys.stdout.flush()
        # Run the GNN brancher
        start = tmm.time()
        nb_nodes, time = 0, 0
        obs, action_set, _, done, info = env.reset(instance)
        nb_nodes += info["nb_nodes"]
        #time += info["time"]
        while not done and tmm.time() - start < time_limit:
            
            # WTF??
            # mask variable features (no incumbent info)
            variable_features = np.delete(obs.column_features, 14, axis=1)
            variable_features = np.delete(variable_features, 13, axis=1)

            constraint_features = torch.FloatTensor(obs.row_features)
            edge_indices = torch.LongTensor(
                    obs.edge_features.indices.astype(np.int32))
            edge_features = torch.FloatTensor(np.expand_dims(
                obs.edge_features.values, axis=-1))
            variable_features = torch.FloatTensor(variable_features)

            with torch.no_grad():
                logits = policy(constraint_features, edge_indices, 
                        edge_features, variable_features)
                action = action_set[logits[action_set.astype(np.int64)].argmax()]
                obs, action_set, _, done, info = env.step(action)
            nb_nodes += info["nb_nodes"]
            #time += info["time"]
            time = tmm.time() - start

            dual, primal = env.model.dual_bound, env.model.primal_bound
            gap = (abs(dual - primal)) / max(1e-8, min(abs(dual), abs(primal)))
            #print(' time {:6.2f}  nodes {: >6}  '
            #      'dual {:9.2e}  primal {:9.2e}  gap {:9.2e}'.format(
            #          time, int(nb_nodes), dual, primal, gap), end='\r')

        dual, primal = env.model.dual_bound, env.model.primal_bound
        gap = (abs(dual - primal)) / max(1e-8, min(abs(dual), abs(primal)))
        print(' time {:6.2f}  nodes {: >6}  '
              'dual {:9.2e}  primal {:9.2e}  gap {:9.2e}'.format(
                  time, int(nb_nodes), dual, primal, gap))

        # Run SCIP's default brancher
        start = tmm.time()
        default_env.reset(instance)
        _, _, _, _, default_info = default_env.step({})
        def_dual, def_primal = default_env.model.dual_bound, default_env.model.primal_bound
        def_gap = (abs(def_dual - def_primal)) / max(1e-8, min(abs(def_dual), abs(def_primal)))
        #def_time, def_nodes = default_info['time'], default_info['nb_nodes']
        def_time, def_nodes = tmm.time() - start, default_info['nb_nodes']
        print(' time {:6.2f}  nodes {: >6}  '
              'dual {:9.2e}  primal {:9.2e}  gap {:9.2e}'.format(
                  def_time, int(def_nodes), def_dual, def_primal, def_gap))
        print('dtime {:6.2f} dnodes {: >6} '
              'ddual {:9.2e} dprimal {:9.2e} dgap {:9.2e}'.format(
                  time-def_time, int(nb_nodes-def_nodes), 
                  dual-def_dual, primal-def_primal, gap-def_gap))
        print()

        records[inst_name] = {
            'gnn': { 'time': time, 'nodes': int(nb_nodes), 'dual': dual, 
                'primal': primal, 'gap': gap },
            'scip': { 'time': def_time, 'nodes': int(def_nodes), 'dual': def_dual, 
                'primal': def_primal, 'gap': def_gap },
        }
        with open('records.json', 'w') as f:
            json.dump(records, f)


