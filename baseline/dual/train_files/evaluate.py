import os
import sys
import glob
import argparse
import pathlib
import numpy as np
import ecole


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_deviceS'] = ''
    device = "cpu"

    running_dir = 'train_files/trained_models/miplib'

    # import pytorch **after** cuda setup
    import torch
    import torch.nn.functional as F
    import torch_geometric
    from utilities import log, pad_tensor, GraphDataset, Scheduler
    sys.path.insert(0,'.')
    from model import GNNPolicy

    policy = GNNPolicy().to(device)
    policy.load_state_dict(torch.load(pathlib.Path(running_dir)/'1_56.pkl', map_location=device))

    time_limit = 1200
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

    instances_valid = sorted(glob.glob('../../instances/miplib/eval/*.mps.gz'))
    for inst_cnt, instance in enumerate(instances_valid):
        print(inst_cnt, instance)
        sys.stdout.flush()
        # Run the GNN brancher
        nb_nodes, time = 0, 0
        obs, action_set, _, done, info = env.reset(instance)
        nb_nodes += info["nb_nodes"]
        time += info["time"]
        while not done and time < time_limit:
            
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
                '''
                observation = (
                    torch.from_numpy(observation.row_features.astype(np.float32)).to(device),
                    torch.from_numpy(observation.edge_features.indices.astype(np.int64)).to(device),
                    torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(-1, 1).to(device),
                    torch.from_numpy(observation.column_features.astype(np.float32)).to(device),
                )
                '''
                logits = policy(constraint_features, edge_indices, 
                        edge_features, variable_features)
                action = action_set[logits[action_set.astype(np.int64)].argmax()]
                obs, action_set, _, done, info = env.step(action)
            nb_nodes += info["nb_nodes"]
            time += info["time"]

        # Run SCIP's default brancher
        default_env.reset(instance)
        _, _, _, _, default_info = default_env.step({})


        print(f"Instance {inst_cnt: >3} | SCIP nb nodes    {int(default_info['nb_nodes']): >4d}  | SCIP time   {default_info['time']: >6.2f} ")
        print(f"             | GNN  nb nodes    {int(nb_nodes): >4d}  | GNN  time   {time: >6.2f} ")
        print(f"             | Gain         {100*(1-nb_nodes/default_info['nb_nodes']): >8.2f}% | Gain      {100*(1-time/default_info['time']): >8.2f}%")
        sys.stdout.flush()


