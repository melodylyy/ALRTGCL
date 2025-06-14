import numpy as np
import torch as th
from sklearn.model_selection import KFold
import torch.nn as nn
from loading_data.loading_data import SimilarityFusion
from torch.utils.data import Dataset, DataLoader



class SampleDataset(Dataset):
    def __init__(self, final_ids, final_labels):
        self.final_ids = final_ids
        self.final_labels = final_labels

    def __len__(self):
        return len(self.final_ids)

    def __getitem__(self, idx):
        return self.final_ids[idx], self.final_labels[idx]


class Data_Process(nn.Module):
    def __init__(self, args):
        super(Data_Process, self).__init__()
        self.args = args
        self.m_linear = nn.Linear(args.m_d, args.d_h, bias=True)
        self.d_linear = nn.Linear(args.d_d, args.d_h, bias=True)
        self.fusion = SimilarityFusion(args)

    def Embedding(self, m_sim, d_sim):
        # m_embed: [431, 256]
        # d_embed: [140, 256]
        m_embed = self.m_linear(m_sim)
        d_embed = self.d_linear(d_sim)

        return m_embed, d_embed

    def get_all_the_samplest(self, association_m, neg_ratio):
        with th.no_grad():
            positive_samples = th.where(association_m != 0)
            positive_samples = th.stack(positive_samples, dim=1)
        # positive_samples = (association_m != 0).nonzero(as_tuple=False)
        with th.no_grad():
            negative_samples = th.where(association_m == 0)
            negative_samples = th.stack(negative_samples, dim=1)
        positive_samples[:, 1] += 431
        negative_samples[:, 1] += 431

        positive_labels = th.ones(positive_samples.size(0), 1, device=self.args.device, dtype=th.long)
        negative_labels = th.zeros(negative_samples.size(0), 1, device=self.args.device, dtype=th.long)


        num_positive = positive_samples.size(0)
        num_negative_to_select = num_positive * neg_ratio
        selected_indices = th.randint(0, negative_samples.size(0), (num_negative_to_select,), device=self.args.device)
        selected_negative_samples = negative_samples[selected_indices]

        return th.cat([positive_samples, selected_negative_samples], dim=0), th.cat(
            [positive_labels, negative_labels[selected_indices]], dim=0)

    def forward(self):
        m_sim, d_sim = self.fusion.calculate_fusion(self.args.G_weight)
        association_m = th.from_numpy(np.loadtxt(self.args.association_m_dir, dtype=int)).to(self.args.device)

        m_sim = m_sim.float()
        d_sim = d_sim.float()


        final_ids, final_labels = self.get_all_the_samplest(association_m, self.args.neg_ratio)


        m_embed, d_embed = self.Embedding(m_sim, d_sim)

        return final_ids, final_labels, m_embed, d_embed


class Data_divide(nn.Module):
    def __init__(self, args, final_ids, final_labels):
        super(Data_divide, self).__init__()
        self.args = args


        self.final_ids = final_ids
        self.final_labels = final_labels

    def forward(self):

        dataset = SampleDataset(self.final_ids, self.final_labels)


        batch_size = self.args.batch_size


        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


        for batch_idx, (data, labels) in enumerate(data_loader):
            data = data.to(self.args.device)
            labels = labels.to(self.args.device)



        train_ids_5 = []
        test_ids_5 = []
        train_labels_5 = []
        test_labels_5 = []

        num_samples, _ = self.final_ids.shape

        kf = KFold(n_splits=self.args.fold, shuffle=True, random_state=1)


        indices = np.arange(num_samples)
        kf_splits = list(kf.split(indices))

        indices_tensor = th.tensor(indices, device=self.args.device)

        for train_index, test_index in kf_splits:

            train_ids_5.append(self.final_ids[indices_tensor[train_index]])
            test_ids_5.append(self.final_ids[indices_tensor[test_index]])

            train_labels_5.append(self.final_labels[indices_tensor[train_index]])
            test_labels_5.append(self.final_labels[indices_tensor[test_index]])

        return train_ids_5, test_ids_5, train_labels_5, test_labels_5









