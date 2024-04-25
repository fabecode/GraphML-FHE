import torch.nn as nn
from torch_geometric.nn import GINEConv, BatchNorm, Linear, GATConv, PNAConv, RGCNConv
import torch.nn.functional as F
import torch
import logging
from brevitas import nn as qnn
from torch.nn.utils import prune

from torch import Tensor
from typing import Optional

class GINe(torch.nn.Module):
    def __init__(self, num_features, num_gnn_layers, n_classes=2, 
                n_hidden=100, edge_updates=False, residual=True, 
                edge_dim=None, dropout=0.0, final_dropout=0.5):
        super().__init__()
        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            conv = GINEConv(nn.Sequential(
                nn.Linear(self.n_hidden, self.n_hidden), 
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_hidden)
                ), edge_dim=self.n_hidden)
            if self.edge_updates: self.emlps.append(nn.Sequential(
                nn.Linear(3 * self.n_hidden, self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_hidden),
            ))
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))

        self.mlp = nn.Sequential(Linear(n_hidden*3, 50), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
                              Linear(25, n_classes))

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
            if self.edge_updates: 
                edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2

        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        out = x
        
        return self.mlp(out)

class DebugQuantLinear(qnn.QuantLinear):
    def inner_forward_impl(self, x: Tensor, quant_weight: Tensor, quant_bias: Optional[Tensor]) -> Tensor:
        print(f"Input shapes - x: {x.shape}, quant_weight: {quant_weight.shape}, quant_bias: {quant_bias.shape}")
        output_tensor = super().inner_forward_impl(x, quant_weight, quant_bias)
        print("Output shape of DebugQuantLinear:", output_tensor.shape)
        try:
            return output_tensor
        except Exception as e:
            print(f"Error occurred in inner_forward_impl: {e}")
            raise

class DebugGINConv(GINEConv):
    def forward(self, x, edge_index, edge_attr=None, size=None):
        try:
            # Add print or logging statements for debugging
            print("Debugging GINConv forward method...")
            # Call the original forward method from the superclass
            return super().forward(x, edge_index, edge_attr)
        except Exception as e:
            # Catch any exceptions and print or log the error message
            print(f"Error occurred in GINConv forward method: {e}")
            # Reraise the exception to propagate it further if needed
            raise

class GINe_FHE(torch.nn.Module):
    def __init__(self, num_features, num_gnn_layers, n_classes=2, 
                 n_hidden=100, edge_updates=False, residual=True, 
                 edge_dim=None, dropout=0.0, final_dropout=0.5,
                 weight_bit_width=16, accumulator_bit_width=16):  # Set desired bit widths
        super().__init__()
        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout

        # Quantized linear layers for node and edge embeddings
        print("Node embedding layer: ", num_features, n_hidden, weight_bit_width)
        self.node_emb = DebugQuantLinear(num_features, n_hidden, weight_bit_width=weight_bit_width, bias=True)
        print("Edge embedding layer: ", edge_dim, n_hidden, weight_bit_width)
        self.edge_emb = DebugQuantLinear(edge_dim, n_hidden, weight_bit_width=weight_bit_width, bias=True)

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        #Keep track of pruned layers
        self.pruned_layers = set()

        #Intialise quant identity layer
        self.quant_inp = qnn.QuantIdentity(return_quant_tensor=True)

        for _ in range(self.num_gnn_layers):
            # Quantized GINEConv layers
            print(f"Conv layer: {self.n_hidden}, {self.n_hidden}, {weight_bit_width}")
            conv = DebugGINConv(nn.Sequential(
                DebugQuantLinear(self.n_hidden, self.n_hidden, weight_bit_width=weight_bit_width, bias=True), 
                qnn.QuantReLU(bit_width=accumulator_bit_width),  # Set accumulator bit width
                DebugQuantLinear(self.n_hidden, self.n_hidden, weight_bit_width=weight_bit_width, bias=True)
            ), edge_dim=self.n_hidden)
            
            if self.edge_updates:
                # Quantized MLP layers for edge updates
                self.emlps.append(nn.Sequential(
                    DebugQuantLinear(3 * self.n_hidden, self.n_hidden, weight_bit_width=weight_bit_width, bias=True),
                    qnn.QuantReLU(bit_width=accumulator_bit_width),  # Set accumulator bit width
                    DebugQuantLinear(self.n_hidden, self.n_hidden, weight_bit_width=weight_bit_width, bias=True)
                ))
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))

        # Quantized MLP for final classification
        print(f"MLP layers: {n_hidden*3}, 50, {weight_bit_width}")
        self.mlp = nn.Sequential(
            DebugQuantLinear(n_hidden*3, 50, weight_bit_width=weight_bit_width, bias=True),
            qnn.QuantReLU(bit_width=accumulator_bit_width),  # Set accumulator bit widths
            nn.Dropout(self.final_dropout),
            DebugQuantLinear(50, 25, weight_bit_width=weight_bit_width, bias=True),
            qnn.QuantReLU(bit_width=accumulator_bit_width),  # Set accumulator bit width
            nn.Dropout(self.final_dropout),
            DebugQuantLinear(25, n_classes, weight_bit_width=weight_bit_width, bias=True)
        )

    def forward(self, x, edge_index, edge_attr):
        print("x initial:", x)
        print("edge_index initial:", edge_index)
        print("edge_attr initial:", edge_attr)
        src, dst = edge_index

        # Forward pass with quantized layers
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        try:
            for i in range(self.num_gnn_layers):
                x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
                if self.edge_updates:
                    edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2

        except Exception as e:
            print(f"An error occurred in the loop: {e}")

        print("x before reshape: ", x)
        try:
            x = x[edge_index.T]
            x = self.quant_inp(x)
            x = x.reshape(-1, 2 * self.n_hidden)
            x = self.quant_inp(x)
            quant_relu = qnn.QuantReLU()
            quant_relu(x)
        except Exception as e:
            print(f"Error: {e}")
        
        print("x after processing: ", x)
        try:
            x = self.quant_inp(x)
            y = self.quant_inp(edge_attr.view(-1, edge_attr.shape[1]))
            x = torch.cat((x, y), 1)
        except Exception as e:
            print(f"An error occurred in the cat line: {e}")
        out = x
        print("Before self.mlp(out): ", out)
        return self.mlp(out)
    
    def prune(self, threshold):
        # Linear layer weight has dimensions NumOutputs x NumInputs
        for name, layer in self.named_modules():
            if isinstance(layer, DebugQuantLinear):
                mask = torch.abs(layer.weight) < threshold  # Create mask of weights below threshold
                prune.custom_from_mask(layer, "weight", mask)  #weight in layer set to 0 where mask = True
                self.pruned_layers.add(name)

    def unprune(self):
        for name, layer in self.named_modules():
            if name in self.pruned_layers:
                prune.remove(layer, "weight")
                self.pruned_layers.remove(name)
    
class Model_Wrapper(torch.nn.Module):
    def __init__(self, original_model, x, edge_index, edge_attr):
        super(Model_Wrapper, self).__init__()
        self.original_model = original_model
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    def forward(self, x):
        # Assuming you have default values for edge_index and edge_attr
        x = self.x
        print("x:", x)
        print("edge_attr:", self.edge_attr)
        print("edge_attr.shape:", self.edge_attr.shape)

        try: 
            return self.original_model(x, self.edge_index, self.edge_attr)
        except Exception as e:
            print(f"Error occurred in forward call: {e}")
            raise
            
class GATe(torch.nn.Module):
    def __init__(self, num_features, num_gnn_layers, n_classes=2, n_hidden=100, n_heads=4, edge_updates=False, edge_dim=None, dropout=0.0, final_dropout=0.5):
        super().__init__()
        # GAT specific code
        tmp_out = n_hidden // n_heads
        n_hidden = tmp_out * n_heads

        self.n_hidden = n_hidden
        self.n_heads = n_heads
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.dropout = dropout
        self.final_dropout = final_dropout
        
        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)
        
        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(self.num_gnn_layers):
            conv = GATConv(self.n_hidden, tmp_out, self.n_heads, concat = True, dropout = self.dropout, add_self_loops = True, edge_dim=self.n_hidden)
            if self.edge_updates: self.emlps.append(nn.Sequential(nn.Linear(3 * self.n_hidden, self.n_hidden),nn.ReLU(),nn.Linear(self.n_hidden, self.n_hidden),))
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))
                
        self.mlp = nn.Sequential(Linear(n_hidden*3, 50), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(25, n_classes))
            
    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        
        for i in range(self.num_gnn_layers):
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
            if self.edge_updates:
                edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2
                    
        logging.debug(f"x.shape = {x.shape}, x[edge_index.T].shape = {x[edge_index.T].shape}")
        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        logging.debug(f"x.shape = {x.shape}")
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        logging.debug(f"x.shape = {x.shape}")
        out = x

        return self.mlp(out)
    
class PNA(torch.nn.Module):
    def __init__(self, num_features, num_gnn_layers, n_classes=2, 
                n_hidden=100, edge_updates=True,
                edge_dim=None, dropout=0.0, final_dropout=0.5, deg=None):
        super().__init__()
        n_hidden = int((n_hidden // 5) * 5)
        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            conv = PNAConv(in_channels=n_hidden, out_channels=n_hidden,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=n_hidden, towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            if self.edge_updates: self.emlps.append(nn.Sequential(
                nn.Linear(3 * self.n_hidden, self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_hidden),
            ))
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))

        self.mlp = nn.Sequential(Linear(n_hidden*3, 50), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
                              Linear(25, n_classes))

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
            if self.edge_updates: 
                edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2

        logging.debug(f"x.shape = {x.shape}, x[edge_index.T].shape = {x[edge_index.T].shape}")
        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        logging.debug(f"x.shape = {x.shape}")
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        logging.debug(f"x.shape = {x.shape}")
        out = x
        return self.mlp(out)
    
class RGCN(nn.Module):
    def __init__(self, num_features, edge_dim, num_relations, num_gnn_layers, n_classes=2, 
                n_hidden=100, edge_update=False,
                residual=True,
                dropout=0.0, final_dropout=0.5, n_bases=-1):
        super(RGCN, self).__init__()

        self.num_features = num_features
        self.num_gnn_layers = num_gnn_layers
        self.n_hidden = n_hidden
        self.residual = residual
        self.dropout = dropout
        self.final_dropout = final_dropout
        self.n_classes = n_classes
        self.edge_update = edge_update
        self.num_relations = num_relations
        self.n_bases = n_bases

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.mlp = nn.ModuleList()

        if self.edge_update:
            self.emlps = nn.ModuleList()
            self.emlps.append(nn.Sequential(
                nn.Linear(3 * self.n_hidden, self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_hidden),
            ))
        
        for _ in range(self.num_gnn_layers):
            conv = RGCNConv(self.n_hidden, self.n_hidden, num_relations, num_bases=self.n_bases)
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(self.n_hidden))

            if self.edge_update:
                self.emlps.append(nn.Sequential(
                    nn.Linear(3 * self.n_hidden, self.n_hidden),
                    nn.ReLU(),
                    nn.Linear(self.n_hidden, self.n_hidden),
                ))

        self.mlp = nn.Sequential(Linear(n_hidden*3, 50), nn.ReLU(), nn.Dropout(self.final_dropout), Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
                              Linear(25, n_classes))

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, RGCNConv):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        edge_type = edge_attr[:, -1].long()
        #edge_attr = edge_attr[:, :-1]
        src, dst = edge_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):
            x =  (x + F.relu(self.bns[i](self.convs[i](x, edge_index, edge_type)))) / 2
            if self.edge_update:
                edge_attr = (edge_attr + F.relu(self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)))) / 2
        
        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        x = self.mlp(x)
        out = x

        return x