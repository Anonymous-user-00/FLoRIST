from peft import (
    set_peft_model_state_dict,
)
import torch
import os
from torch.nn.functional import normalize
from torch.nn import ZeroPad2d
from torch.linalg import svd
import numpy as np

def FedAvg(model, selected_clients_set, output_dir, local_dataset_len_dict, epoch, stacking, florist, flex, ffa, threshold, lora_r, heter, local_ranks, zero_padding, full):
    weights_array = normalize(
        torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
                     dtype=torch.float32),
        p=1, dim=0)
    print("Weights:", weights_array)

    if florist:
        aggregated_BA = {}
        for k, client_id in enumerate(selected_clients_set):
            single_output_dir = os.path.join(output_dir, str(epoch), "local_output_{}".format(client_id), "pytorch_model.bin")
            single_weights = torch.load(single_output_dir, map_location = 'cpu')
            x = 0
            if k == 0:
                weighted_single_weights = single_weights
                for key in weighted_single_weights.keys():
                    #weighted_single_weights[key] = weighted_single_weights[key] * (weights_array[k])
                    #print(weighted_single_weights[key].shape)
                    if heter:
                        x += 1
                        if weighted_single_weights[key].shape[0] == local_ranks[client_id]:
                            weighted_single_weights[key] = weighted_single_weights[key] * (weights_array[k] * 1)
                    else:
                        if weighted_single_weights[key].shape[0] == lora_r:
                            weighted_single_weights[key] = weighted_single_weights[key] * (weights_array[k] * 1)

            else:
                for key in single_weights.keys():
                    if heter:
                        x += 1
                        if single_weights[key].shape[0] == local_ranks[client_id]:
                            new = [weighted_single_weights[key], single_weights[key] * (weights_array[k]) * 1]
                            weighted_single_weights[key] = torch.cat(new, dim=0)
                    else:
                        if single_weights[key].shape[0] == lora_r:
                            new = [weighted_single_weights[key], single_weights[key] * (weights_array[k]) * 1]
                            weighted_single_weights[key] = torch.cat(new, dim=0)

                    if heter:
                        if single_weights[key].shape[1] == local_ranks[client_id]:
                            new = [weighted_single_weights[key], single_weights[key]]#  * (weights_array[k])]
                            weighted_single_weights[key] = torch.cat(new, dim=1)
                    else:
                        if single_weights[key].shape[1] == lora_r:
                            new = [weighted_single_weights[key], single_weights[key]]#  * (weights_array[k])]
                            weighted_single_weights[key] = torch.cat(new, dim=1)
        print("Shape")
        print(len(weighted_single_weights))
        for key in weighted_single_weights:

            if "lora_A" in key:
                B_key = key.replace(".lora_A.weight", ".lora_B.weight")
                U_B, Sigma_B, VT_B = svd(weighted_single_weights[B_key], full_matrices=False)
                U_A, Sigma_A, VT_A = svd(weighted_single_weights[key], full_matrices=False)
                Sigma_A = torch.diag(Sigma_A)
                Sigma_B = torch.diag(Sigma_B)
                Q = VT_B @ U_A
                P = Sigma_B @ Q @ Sigma_A
                U_P, Sigma_P, VT_P = svd(P, full_matrices=False)
                U_g = U_B @ U_P
                Sigma_g =  torch.diag(Sigma_P)
                VT_g = VT_P @ VT_A
                A_g = U_g @ Sigma_g
                B_g = VT_g

                cumulative_energy = torch.cumsum(Sigma_P ** 2, dim=0) / torch.sum(Sigma_P ** 2)
                indices = (cumulative_energy >= threshold).nonzero(as_tuple=False)
                if indices.numel() == 0:
                    B_new = U_g @ Sigma_g
                    A_new = VT_g
                else:
                    k_optimal = indices.min().item() + 1

                    U_k = U_g[:, :k_optimal]
                    S_k = torch.diag(Sigma_P[:k_optimal])
                    V_k = VT_g[:k_optimal, :]

                    B_new = U_k @ S_k
                    A_new = V_k

                weighted_single_weights[key] = A_new
                weighted_single_weights[B_key] = B_new


    elif flex:
        aggregated_BA = {}
        for k, client_id in enumerate(selected_clients_set):
            single_output_dir = os.path.join(output_dir, str(epoch), "local_output_{}".format(client_id), "pytorch_model.bin")
            single_weights = torch.load(single_output_dir, map_location = 'cpu')

            for key in single_weights.keys():
                if ".lora_A.weight" in key:
                    B_key = key.replace(".lora_A.weight", ".lora_B.weight")

                    B = single_weights[B_key]
                    A = single_weights[key]
                    BA = torch.matmul(B, A)

                    weighted_BA = BA * weights_array[k]

                    if key not in aggregated_BA:
                        aggregated_BA[key] = weighted_BA
                    else:
                        aggregated_BA[key] += weighted_BA

        weighted_single_weights = {}
        client_weights = []
        for client_id in selected_clients_set:
            single_output_dir = os.path.join(output_dir, str(epoch), "local_output_{}".format(client_id), "pytorch_model.bin")
            single_weights = torch.load(single_output_dir, map_location = 'cpu')
            # print(single_weights.keys())
            for key, BA in aggregated_BA.items():
                U, S, Vt = svd(BA, full_matrices=False)
                client_rank = single_weights[key].shape[0]
                U_k = U[:, :client_rank]
                S_k = torch.diag(S[:client_rank])
                V_k = Vt[:client_rank, :]

                B_new = U_k @ S_k  # New B matrix
                A_new = V_k  # New A matrix

                B_key = key.replace(".lora_A.weight", ".lora_B.weight")
                weighted_single_weights[key] = A_new
                weighted_single_weights[B_key] = B_new
            client_weights.append(weighted_single_weights)

    else:
        for k, client_id in enumerate(selected_clients_set):
            single_output_dir = os.path.join(output_dir, str(epoch), "local_output_{}".format(client_id),
                                             "pytorch_model.bin")
            single_weights = torch.load(single_output_dir, map_location = 'cpu')
            #print(single_weights)
            #print("y")

            x = 0
            if full:
                if k == 0:
                    weighted_single_weights = single_weights
                    for key in weighted_single_weights.keys():
                        weighted_single_weights[key] = weighted_single_weights[key] * (weights_array[k])
                else:
                    for key in single_weights.keys():
                        weighted_single_weights[key] += single_weights[key] * (weights_array[k])

            else:
                if stacking:
                    if zero_padding:
                        max_lora = max(local_ranks)
                        if k == 0:
                            weighted_single_weights = single_weights
                            for key in weighted_single_weights.keys():
                                if single_weights[key].shape[0] == local_ranks[client_id]:
                                    pad = ZeroPad2d(padding=(0, 0, 0, max_lora-local_ranks[client_id]))
                                    weighted_single_weights[key] = pad(weighted_single_weights[key]) * (weights_array[k])
                                elif single_weights[key].shape[1] == local_ranks[client_id]:
                                    pad = ZeroPad2d(padding=(0, max_lora-local_ranks[client_id], 0, 0))
                                    weighted_single_weights[key] = pad(weighted_single_weights[key]) * (weights_array[k])
                        else:
                            for key in single_weights.keys():
                                #print(single_weights[key].shape)
                                if single_weights[key].shape[0] == local_ranks[client_id]:
                                    pad = ZeroPad2d(padding=(0, 0, 0, max_lora-local_ranks[client_id]))
                                    single_weights[key] = pad(single_weights[key]) * (weights_array[k])
                                    weighted_single_weights[key] += single_weights[key]
                                elif single_weights[key].shape[1] == local_ranks[client_id]:
                                    pad = ZeroPad2d(padding=(0, max_lora-local_ranks[client_id], 0, 0))
                                    single_weights[key] = pad(single_weights[key]) * (weights_array[k])
                                    #print(single_weights[key][255,32])
                                    weighted_single_weights[key] += single_weights[key]

                    else:
                        if k == 0:
                            weighted_single_weights = single_weights
                            for key in weighted_single_weights.keys():
                                #weighted_single_weights[key] = weighted_single_weights[key] * (weights_array[k])
                                #print(weighted_single_weights[key].shape)
                                if heter:
                                    x += 1
                                    if weighted_single_weights[key].shape[0] == local_ranks[client_id]:
                                        weighted_single_weights[key] = weighted_single_weights[key] * (weights_array[k] * 1)
                                else:
                                    if weighted_single_weights[key].shape[0] == lora_r:
                                        weighted_single_weights[key] = weighted_single_weights[key] * (weights_array[k] * 1)

                        else:
                            for key in single_weights.keys():
                                if heter:
                                    x += 1
                                    if single_weights[key].shape[0] == local_ranks[client_id]:
                                        new = [weighted_single_weights[key], single_weights[key] * (weights_array[k]) * 1]
                                        weighted_single_weights[key] = torch.cat(new, dim=0)
                                else:
                                    if single_weights[key].shape[0] == lora_r:
                                        new = [weighted_single_weights[key], single_weights[key] * (weights_array[k]) * 1]
                                        weighted_single_weights[key] = torch.cat(new, dim=0)

                                if heter:
                                    if single_weights[key].shape[1] == local_ranks[client_id]:
                                        new = [weighted_single_weights[key], single_weights[key]]#  * (weights_array[k])]
                                        weighted_single_weights[key] = torch.cat(new, dim=1)
                                else:
                                    if single_weights[key].shape[1] == lora_r:
                                        new = [weighted_single_weights[key], single_weights[key]]#  * (weights_array[k])]
                                        weighted_single_weights[key] = torch.cat(new, dim=1)

                else:
                    if zero_padding:
                        max_lora = max(local_ranks)
                        if k == 0:
                            weighted_single_weights = single_weights
                            for key in weighted_single_weights.keys():
                                if single_weights[key].shape[0] == local_ranks[client_id]:
                                    pad = ZeroPad2d(padding=(0, 0, 0, max_lora-local_ranks[client_id]))
                                    weighted_single_weights[key] = pad(weighted_single_weights[key]) * (weights_array[k])
                                elif single_weights[key].shape[1] == local_ranks[client_id]:
                                    pad = ZeroPad2d(padding=(0, max_lora-local_ranks[client_id], 0, 0))
                                    weighted_single_weights[key] = pad(weighted_single_weights[key]) * (weights_array[k])
                        else:
                            for key in single_weights.keys():
                                #print(single_weights[key].shape)
                                if single_weights[key].shape[0] == local_ranks[client_id]:
                                    pad = ZeroPad2d(padding=(0, 0, 0, max_lora-local_ranks[client_id]))
                                    single_weights[key] = pad(single_weights[key]) * (weights_array[k])
                                    weighted_single_weights[key] += single_weights[key]
                                elif single_weights[key].shape[1] == local_ranks[client_id]:
                                    pad = ZeroPad2d(padding=(0, max_lora-local_ranks[client_id], 0, 0))
                                    single_weights[key] = pad(single_weights[key]) * (weights_array[k])
                                    #print(single_weights[key][255,32])
                                    weighted_single_weights[key] += single_weights[key]
                                # print(weighted_single_weights[key])
                    else:
                        if k == 0:
                            weighted_single_weights = {key: single_weights[key] * (weights_array[k]) for key in
                                                single_weights.keys()}
                        else:
                            weighted_single_weights = {key: weighted_single_weights[key] + single_weights[key] * (weights_array[k])
                                                for key in
                                                single_weights.keys()}

    filename = output_dir + 'rank_log.txt'
    file = open(filename,'a')
    print("florist", florist)
    print("full", full)
    print("stacking", stacking)
    file.write("florist:" + str(florist) + '\n')
    file.write("full:" + str(full) + '\n')
    file.write("stacking:" + str(stacking) + '\n')
    file.write("threshold:" + str(threshold) + '\n')
    total_rank = 0
    total_param = 0
    if ffa:
        for n, key in enumerate(weighted_single_weights.keys()):
            wt = np.array(weighted_single_weights[key])
            if 'lora_A' in key:
                print(wt.shape)
                # print(wt)
                file.write(str(wt.shape) + '\n')
                total_param += wt.shape[0] * wt.shape[1]
                total_rank += wt.shape[0]
    else:
        for n, key in enumerate(weighted_single_weights.keys()):
            wt = np.array(weighted_single_weights[key])
            print(wt.shape)
            # print(wt)
            file.write(str(wt.shape) + '\n')
            total_param += wt.shape[0] * wt.shape[1]
            if n%2 == 0:
                total_rank += wt.shape[0]
            else:
                total_rank += wt.shape[1]

    print("Total rank: ",total_rank)
    file.write("Total rank:"+  str(total_rank) + '\n*************\n\n')
    print("Total parameters: ", total_param)

    os.makedirs(output_dir,exist_ok=True)
    if flex:
        for i, weights in enumerate(client_weights):
            out_path = os.path.join(output_dir, str(epoch), 'flex', str(i))
            os.makedirs(out_path,exist_ok=True)
            print("Saving flex adapters")
            torch.save(weights, os.path.join(out_path, "adapter_model.bin"))
        return model, total_rank, total_param
    if florist:
        out_path = os.path.join(output_dir, str(epoch), 'florist')
        os.makedirs(out_path,exist_ok=True)
        print("Saving florist adapters")
        torch.save(weighted_single_weights, os.path.join(out_path, "adapter_model.bin"))
        return model, total_rank, total_param
    elif stacking:
        out_path = os.path.join(output_dir, str(epoch), 'flora')
        os.makedirs(out_path,exist_ok=True)
        print("Saving flora adapters")
        torch.save(weighted_single_weights, os.path.join(out_path, "adapter_model.bin"))
        return model, total_rank, total_param
    elif full:
        torch.save(weighted_single_weights, os.path.join(output_dir, str(epoch), "pytorch_model.bin"))
        model.load_state_dict(weighted_single_weights)
        return model, total_rank, total_param
    else:
        out_path = os.path.join(output_dir, str(epoch), 'fedit')
        os.makedirs(out_path,exist_ok=True)
        torch.save(weighted_single_weights, os.path.join(out_path, "adapter_model.bin"))
        # set_peft_model_state_dict(model, weighted_single_weights, "default")
        print("Saving FedIT adapters")
        return model, total_rank, total_param
