import torch
import torch.nn as nn


class SelfAttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SelfAttentionLayer, self).__init__()

        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x is (batch_size, num_points, input_dim)
        Q = self.query(x)  # (batch_size, num_points, hidden_dim)
        K = self.key(x)  # (batch_size, num_points, hidden_dim)
        V = self.value(x)  # (batch_size, num_points, hidden_dim)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(K.size(-1), dtype=torch.float32))
        attention_weights = self.softmax(attention_scores)  # (batch_size, num_points, num_points)

        # Aggregate the values based on attention weights
        attention_output = torch.matmul(attention_weights, V)  # (batch_size, num_points, hidden_dim)

        # Final linear layer
        output = self.fc(attention_output)  # (batch_size, num_points, output_dim)
        return output


class ClusterFeatureNetWithAttention(nn.Module):
    def __init__(self, cfg):
        super(ClusterFeatureNetWithAttention, self).__init__()

        input_dim = cfg.input_feat_dim
        hidden_dim = cfg.hidden_feat_dim
        # hidden_dim1 = cfg.hidden_feat_dim1
        # hidden_dim2 = cfg.hidden_feat_dim2
        output_dim = cfg.output_feat_dim

        self.mode = cfg.mode

        self.self_attention = SelfAttentionLayer(input_dim, hidden_dim, output_dim)

        self.mlp = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Define the pooling layer (e.g., max pooling)
        # self.pool = nn.AdaptiveAvgPool1d(1)  # Adaptive pooling to get fixed-size output

        # Define the MLP layers
        # self.mlp = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim1),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim1, hidden_dim1),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim1, hidden_dim2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim2, hidden_dim2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim2, output_dim)
        # )

    # just reduce competitive coding
    def aggregation(self, idx_spt2pts_src, src_pts_feats, src_pts_coord, idx_pts2voxel_src=None):
        spt_feat_src, spt_coord_src = [], []

        for pts_idx_src in idx_spt2pts_src:
            if self.mode == "train":
                pts_feats_in_spt_src = src_pts_feats[:, pts_idx_src, :]
                pts_coord_in_spt_src = src_pts_coord[:, pts_idx_src, :]
            elif self.mode == "test":
                idx_voxel_in_spt_src = idx_pts2voxel_src[pts_idx_src]
                idx_voxel_in_spt_src = idx_voxel_in_spt_src[idx_voxel_in_spt_src >= 0]
                pts_feats_in_spt_src = src_pts_feats[:, idx_voxel_in_spt_src, :]
                pts_coord_in_spt_src = src_pts_coord[:, idx_voxel_in_spt_src, :]

            # Permute to (batch_size, feature_dim, num_points) for pooling
            # pts_feats_in_spt_src = pts_feats_in_spt_src.permute(0, 2, 1)
            # Apply pooling layer, x shape: (batch_size, feature_dim, 1)
            attention_output = self.self_attention(pts_feats_in_spt_src)
            src_cluster_feats = torch.mean(attention_output, dim=1)  # Aggregate across points
            # src_cluster_feats = self.pool(pts_feats_in_spt_src)
            # Remove singleton dimension, x shape: (batch_size, feature_dim), (1, 192)
            # src_cluster_feats = src_cluster_feats.squeeze(dim=2)
            # Apply MLP
            src_cluster_feats = self.mlp(src_cluster_feats)
            # get the cluster centroid coord
            src_cluster_coord = torch.mean(pts_coord_in_spt_src, dim=1)

            spt_feat_src.append(src_cluster_feats)
            spt_coord_src.append(src_cluster_coord)

        spt_feat_src = torch.cat(spt_feat_src)
        spt_coord_src = torch.cat(spt_coord_src)
        return spt_feat_src, spt_coord_src

    def forward(self, dict_input):
        if self.mode == 'train':
            # src_clustering : Nx7 is not necessary
            idx_pts2spt_src = dict_input['src_clustering'][:, :, 6].to(torch.int64)
            idx_pts2spt_tgt = dict_input['tgt_clustering'][:, :, 6].to(torch.int64)

            spt_src_id = torch.unique(idx_pts2spt_src[0, :], sorted=True, return_inverse=False)
            spt_tgt_id = torch.unique(idx_pts2spt_tgt[0, :], sorted=True, return_inverse=False)

            ##############
            # Assuming spt_src_id and idx_pts2spt_src are tensors
            # idx_spt_src = spt_src_id.clone()  # You don't need a loop for this
            # idx_spt_tgt = spt_tgt_id.clone()

            # Use vectorized operations to find where matches occur
            mask_src = (idx_pts2spt_src[0, :].unsqueeze(0) == spt_src_id.unsqueeze(1))
            idx_spt2pts_src = mask_src.nonzero(as_tuple=True)[1].split(mask_src.sum(dim=1).tolist())

            mask_tgt = (idx_pts2spt_tgt[0, :].unsqueeze(0) == spt_tgt_id.unsqueeze(1))
            idx_spt2pts_tgt = mask_tgt.nonzero(as_tuple=True)[1].split(mask_tgt.sum(dim=1).tolist())
            ##############
        elif self.mode == 'test':
            # idx_pts2spt_src = dict_input['idx_pts2spt_src']
            # idx_pts2spt_tgt = dict_input['idx_pts2spt_tgt']
            idx_spt2pts_src = dict_input['idx_spt2pts_src']
            idx_spt2pts_tgt = dict_input['idx_spt2pts_tgt']
            idx_pts2voxel_src = dict_input['idx_pts2voxel_src']
            idx_pts2voxel_tgt = dict_input['idx_pts2voxel_tgt']

        # x shape: (batch_size, num_points, feature_dim)
        src_pts_feats = dict_input['src_feats'].clone().detach()
        tgt_pts_feats = dict_input['tgt_feats'].clone().detach()

        src_pts_coord = dict_input['src_pts']
        tgt_pts_coord = dict_input['tgt_pts']

        # spt_feat_src, spt_feat_tgt, spt_coord_src, spt_coord_tgt = [], [], [], []
        if self.mode == 'train':
            spt_feat_src, spt_coord_src = self.aggregation(idx_spt2pts_src, src_pts_feats, src_pts_coord)
            spt_feat_tgt, spt_coord_tgt = self.aggregation(idx_spt2pts_tgt, tgt_pts_feats, tgt_pts_coord)
        elif self.mode == 'test':
            # considering small patch removal, the pts2voxel indices are not the same as the segment indices
            spt_feat_src, spt_coord_src = self.aggregation(idx_spt2pts_src, src_pts_feats, src_pts_coord, idx_pts2voxel_src)
            spt_feat_tgt, spt_coord_tgt = self.aggregation(idx_spt2pts_tgt, tgt_pts_feats, tgt_pts_coord, idx_pts2voxel_tgt)

        dict_output = dict()
        dict_output['spt_feat_src'] = spt_feat_src
        dict_output['spt_feat_tgt'] = spt_feat_tgt
        dict_output['spt_coord_src'] = spt_coord_src
        dict_output['spt_coord_tgt'] = spt_coord_tgt
        return dict_output

# Example usage:
# model = ClusterFeatureNet(input_dim=feature_size, output_dim=cluster_feature_size)
# previous version, input_dim --> 128 --> 64 --> output_dim

# # Create a ClusterFeatureNet instance
# input_dim = 64  # Number of features per point
# hidden_dim = 128  # Hidden layer dimension
# output_dim = 256  # Output feature dimension
# model = ClusterFeatureNet(input_dim, hidden_dim, output_dim)
# # Example input: batch of point clouds with 10 clusters, each with 32 points and 64 features
# x = torch.rand(10, 32, input_dim)  # (batch_size, num_points, feature_dim)
# # Forward pass
# features = model(x)
# # Output shape: (batch_size, output_dim)
# print(features.shape)  # Should be (10, output_dim)
