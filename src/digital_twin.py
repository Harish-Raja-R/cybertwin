from enum import Enum
import collections

class NodeState(Enum):
    NORMAL = 'NORMAL'
    SUSPICIOUS = 'SUSPICIOUS'
    UNDER_ATTACK = 'UNDER_ATTACK'
    RECOVERING = 'RECOVERING'

class DigitalTwinNode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.state = NodeState.NORMAL
        
        # Traffic statistics (rolling window)
        self.history_len = 50
        self.attack_prob_history = collections.deque(maxlen=self.history_len)
        self.traffic_intensity_history = collections.deque(maxlen=self.history_len)
        
        # Current metrics
        self.current_attack_prob = 0.0
        self.risk_score = 0.0
        self.traffic_volume = 0
        self.packet_count = 0
        
    def update_metrics(self, flow_features, attack_prob):
        """
        flow_features is assumed to be a dictionary or a row array.
        For simulation, we approximate traffic intensity using total packets or bytes if available.
        """
        self.current_attack_prob = attack_prob
        self.attack_prob_history.append(attack_prob)
        
        # Simulate traffic volume (using feature indices: e.g. Total Packets is often engineered or Fwd Packets is at index 2)
        # We just use a placeholder scaled intensity based on the attack prob and random jitter if actual bytes aren't mapped.
        # But we must use actual available processed features!
        # Assuming flow_features is a vector of 70 features.
        # Fwd Packet Length Max, Total Length of Fwd Packets, etc.
        # We will just take the sum of the absolute feature vector as a proxy for 'traffic intensity' for the simulation.
        intensity = abs(sum(flow_features))
        self.traffic_intensity_history.append(intensity)
        self.traffic_volume += intensity
        self.packet_count += 1
        
        self._calculate_risk_score()
        self._update_state()
        
    def _calculate_risk_score(self):
        """
        Risk Score Formula:
        0.5 * Attack_Prob + 0.3 * (Recent_Suspicious_Ratio) + 0.2 * (Normalized_Traffic_Intensity)
        Note: The weights are simulation parameters rather than learned physical-system parameters.
        """
        if not self.attack_prob_history:
            self.risk_score = 0.0
            return
            
        recent_suspicious = sum(1 for p in self.attack_prob_history if p > 0.5)
        suspicious_ratio = recent_suspicious / len(self.attack_prob_history)
        
        avg_intensity = sum(self.traffic_intensity_history) / len(self.traffic_intensity_history)
        # Normalize intensity by dividing by max observed or a safe constant
        norm_intensity = min(avg_intensity / 100.0, 1.0)
        
        self.risk_score = (0.5 * self.current_attack_prob) + (0.3 * suspicious_ratio) + (0.2 * norm_intensity)
        self.risk_score = min(max(self.risk_score, 0.0), 1.0)
        
    def _update_state(self):
        """
        State Transitions:
        NORMAL -> SUSPICIOUS if attack_prob > 0.5 or risk > 0.4
        SUSPICIOUS -> UNDER_ATTACK if attack_prob > 0.8 or risk > 0.7
        UNDER_ATTACK -> RECOVERING if attack_prob < 0.2 and risk < 0.5
        RECOVERING -> NORMAL if risk < 0.3
        SUSPICIOUS -> NORMAL if risk < 0.3 and attack_prob < 0.2
        """
        if self.state == NodeState.NORMAL:
            if self.current_attack_prob > 0.8 or self.risk_score > 0.7:
                self.state = NodeState.UNDER_ATTACK
            elif self.current_attack_prob > 0.5 or self.risk_score > 0.4:
                self.state = NodeState.SUSPICIOUS
                
        elif self.state == NodeState.SUSPICIOUS:
            if self.current_attack_prob > 0.8 or self.risk_score > 0.7:
                self.state = NodeState.UNDER_ATTACK
            elif self.current_attack_prob < 0.2 and self.risk_score < 0.3:
                self.state = NodeState.NORMAL
                
        elif self.state == NodeState.UNDER_ATTACK:
            if self.current_attack_prob < 0.2 and self.risk_score < 0.5:
                self.state = NodeState.RECOVERING
                
        elif self.state == NodeState.RECOVERING:
            if self.current_attack_prob > 0.8 or self.risk_score > 0.7:
                self.state = NodeState.UNDER_ATTACK
            elif self.risk_score < 0.3:
                self.state = NodeState.NORMAL

class DigitalTwinNetwork:
    def __init__(self, node_ids):
        self.nodes = {nid: DigitalTwinNode(nid) for nid in node_ids}
        self.global_risk_score = 0.0
        
    def update_network(self, node_id, flow_features, attack_prob):
        if node_id in self.nodes:
            self.nodes[node_id].update_metrics(flow_features, attack_prob)
            self._update_global_state()
            
    def _update_global_state(self):
        # Global risk is the max risk of any node, plus a small penalty for multiple nodes under attack
        if not self.nodes:
            self.global_risk_score = 0.0
            return
            
        risks = [node.risk_score for node in self.nodes.values()]
        under_attack_count = sum(1 for node in self.nodes.values() if node.state == NodeState.UNDER_ATTACK)
        
        base_risk = max(risks)
        penalty = min(0.05 * under_attack_count, 0.2)
        
        self.global_risk_score = min(base_risk + penalty, 1.0)
