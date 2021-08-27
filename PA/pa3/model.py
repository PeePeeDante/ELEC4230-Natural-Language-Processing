



import torch
import torch.nn as nn
import torch.nn.functional as F


class WordCNN(nn.Module):

    def __init__(self, args, vocab_size, embedding_matrix=None):
        super(WordCNN, self).__init__()
        # TODO
        # Some useful function: nn.Embedding(), nn.Dropout(), nn.Linear(), nn.Conv1d() or nn.Conv2d(),
        
        dropout_r = args.dropout
        kernel_num = args.kernel_num
        kernel_size_list = [int(i) for i in args.kernel_sizes.split(',')]
        total_num_filter = kernel_num * len(kernel_size_list)
        embed_dim = args.embed_dim
        class_num = args.class_num
        
        convs = [nn.Conv1d(in_channels=embed_dim, out_channels=kernel_num, kernel_size=kernel_size) for kernel_size in kernel_size_list]
        
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.conv_modules = nn.ModuleList(convs)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_r)
        self.linear = nn.Linear(in_features=total_num_filter, out_features=class_num)
        

    def forward(self, x):
        # TODO
        # Input x dim: (batch_size, max_seq_len, D)
        # Output logit dim: (batch_size, num_classes)
        embed = self.embeddings(x) # (batch_size, sentence_length, wordvec_size)
        
        embed = embed.transpose(1,2) # (batch_size, wordvec_size, sentence_length)
        
        feature_list = []
        for conv in self.conv_modules:
            feature_map = self.tanh(conv(embed))
            max_pooled, _ = feature_map.max(dim=2)
            #avg_pooled = feature_map.mean(dim=2)
            #feature_list.append(avg_pooled)
            feature_list.append(max_pooled)
            
        features = torch.cat(feature_list, dim=1)
        features_regularized = self.dropout(features)
        out_wo_softmax = self.linear(features_regularized)
        
        return out_wo_softmax
