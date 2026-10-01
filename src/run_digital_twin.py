import torch
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from src.temporal_simulator import load_and_simulate
from src.temporal_dataset import get_dataloaders
from src.models.temporal_bilstm import TemporalBiLSTM
from src.digital_twin import DigitalTwinNetwork, NodeState
import time

def run_simulation(model, test_loader, device='cuda', num_steps=1000):
    """
    Runs a digital twin simulation on test data sequences.
    """
    model.eval()
    
    # We will pretend the sequences arrive over time for a subset of virtual nodes
    node_ids = ['node_001', 'node_002', 'node_003']
    dt_network = DigitalTwinNetwork(node_ids)
    
    state_log = []
    
    step = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            # Predict
            outputs = model(batch_X)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            features = batch_X.cpu().numpy()
            targets = batch_y.cpu().numpy()
            
            for i in range(len(probs)):
                if step >= num_steps:
                    break
                    
                # Distribute traffic sequentially to nodes
                node_id = node_ids[step % len(node_ids)]
                
                # The "current" flow feature vector is the last one in the window
                current_flow_features = features[i, -1, :]
                attack_prob = probs[i]
                
                start_t = time.perf_counter()
                dt_network.update_network(node_id, current_flow_features, attack_prob)
                latency = (time.perf_counter() - start_t) * 1000 # ms
                
                node = dt_network.nodes[node_id]
                
                state_log.append({
                    'simulation_step': step,
                    'node_id': node_id,
                    'traffic_volume': node.traffic_volume,
                    'packet_count': node.packet_count,
                    'attack_probability': attack_prob,
                    'risk_score': node.risk_score,
                    'network_state': node.state.value,
                    'global_risk': dt_network.global_risk_score,
                    'true_sequence_label': targets[i],
                    'update_latency_ms': latency
                })
                
                step += 1
            if step >= num_steps:
                break
                
    log_df = pd.DataFrame(state_log)
    os.makedirs('results/tables', exist_ok=True)
    log_df.to_csv('results/tables/digital_twin_state_log.csv', index=False)
    
    return log_df

def plot_digital_twin(log_df, output_dir='results/figures'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Attack Probability Timeline
    plt.figure(figsize=(12, 6))
    for node_id in log_df['node_id'].unique():
        node_data = log_df[log_df['node_id'] == node_id]
        plt.plot(node_data['simulation_step'], node_data['attack_probability'], label=f'{node_id} Attack Prob', alpha=0.7)
    plt.axhline(0.5, color='r', linestyle='--', label='Suspicious Threshold')
    plt.xlabel('Simulation Step')
    plt.ylabel('Probability')
    plt.title('Attack Probability over Simulated Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/attack_probability_timeline.png')
    plt.close()
    
    # Network Risk Timeline
    plt.figure(figsize=(12, 6))
    plt.plot(log_df['simulation_step'], log_df['global_risk'], label='Global Risk Score', color='darkred', linewidth=2)
    plt.xlabel('Simulation Step')
    plt.ylabel('Risk Score')
    plt.title('Digital Twin Network Risk Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/network_risk_timeline.png')
    plt.close()
    
    # State Transitions
    plt.figure(figsize=(12, 6))
    state_map = {'NORMAL': 0, 'SUSPICIOUS': 1, 'RECOVERING': 2, 'UNDER_ATTACK': 3}
    for node_id in log_df['node_id'].unique():
        node_data = log_df[log_df['node_id'] == node_id]
        numeric_states = node_data['network_state'].map(state_map)
        plt.plot(node_data['simulation_step'], numeric_states, label=node_id, marker='o', markersize=2, linestyle='-')
    plt.yticks([0, 1, 2, 3], ['NORMAL', 'SUSPICIOUS', 'RECOVERING', 'UNDER_ATTACK'])
    plt.xlabel('Simulation Step')
    plt.ylabel('Network State')
    plt.title('Digital Twin Node State Transitions')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/network_state_transitions.png')
    plt.close()
    
    # Print latency stats
    mean_latency = log_df['update_latency_ms'].mean()
    median_latency = log_df['update_latency_ms'].median()
    p95_latency = log_df['update_latency_ms'].quantile(0.95)
    
    print("\nSimulation Latency Statistics (ms):")
    print(f"Mean: {mean_latency:.4f} ms")
    print(f"Median: {median_latency:.4f} ms")
    print(f"95th Percentile: {p95_latency:.4f} ms")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Loading test dataset sequences...")
    _, _, test_data, _ = load_and_simulate(window_size=20, strategy='any')
    
    from src.temporal_dataset import TemporalDataset
    from torch.utils.data import DataLoader
    test_dataset = TemporalDataset(test_data[0], test_data[1])
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    print("Loading Temporal BiLSTM...")
    model = TemporalBiLSTM(input_dim=70, hidden_dim=128, num_layers=2, dropout=0.3).to(device)
    model.load_state_dict(torch.load('results/temporal_model_best.pth', weights_only=True))
    
    print("Running Digital Twin Simulation (1000 steps)...")
    log_df = run_simulation(model, test_loader, device=device, num_steps=1000)
    
    print("Generating Visualizations...")
    plot_digital_twin(log_df)
    
    print("Simulation Complete!")
