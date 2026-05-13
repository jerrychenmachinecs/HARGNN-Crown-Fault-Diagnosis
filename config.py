import torch
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class Config:
    def __init__(self):
        self.random_seed = 42
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 4
        
        self.data_root = './data'
        self.raw_data_path = os.path.join(self.data_root, 'raw')
        self.processed_data_path = os.path.join(self.data_root, 'processed')
        self.checkpoint_path = './checkpoints'
        self.log_path = './logs'
        self.result_path = './results'
        
        self.num_stands = 7
        self.num_params_per_stand = 15
        self.input_dim = 105
        self.stand_names = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']
        
        self.param_names = [
            'rolling_force', 'bending_force', 'cvc_position',
            'work_roll_diameter', 'work_roll_coils', 'backup_roll_diameter',
            'backup_roll_coils', 'rolling_speed', 'entry_temperature',
            'exit_temperature', 'entry_thickness', 'exit_thickness',
            'cooling_water_flow', 'cooling_water_temp', 'calculated_crown'
        ]
        
        self.num_classes = 3
        self.class_names = ['under_crown', 'normal', 'over_crown']
        self.crown_thresholds = [10.0, 50.0]
        
        self.batch_size = 64
        self.num_epochs = 200
        self.learning_rate = 0.001
        self.weight_decay = 1e-5
        self.early_stopping_patience = 20
        
        self.gat_num_layers = 3
        self.gat_num_heads = 8
        self.gat_hidden_dims = [64, 128, 256]
        self.gat_dropout = 0.3
        self.gat_alpha = 0.2
        self.gat_use_residual = True
        self.gat_use_gating = True
        
        self.gru_num_layers = 2
        self.gru_hidden_dim = 256
        self.gru_dropout = 0.2
        self.gru_bidirectional = True
        
        self.attention_dim = 128
        self.use_param_attention = True
        self.use_feature_attention = True
        self.use_stand_attention = True
        
        self.mlp_hidden_dim = 256
        self.mlp_dropout = 0.3
        
        self.use_cost_sensitive = True
        self.cost_beta = 0.9999
        self.cost_epsilon = 1e-7
        
        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        
        self.use_standardization = True
        self.use_time_features = True
        self.time_window = 5
        
        self.graph_construction = 'fully_connected'
        self.edge_threshold = 0.3
        self.knn_k = 10
        
        self.lr_scheduler = 'cosine'
        self.warmup_epochs = 10
        self.min_lr = 1e-6
        
        self.gradient_clip = 1.0
        self.use_amp = False
        
        self.save_interval = 10
        self.eval_interval = 5
        self.log_interval = 100
        
        self.causal_alpha = 0.01
        self.causal_max_condition_set_size = 4
        self.causal_ci_test = 'kci'
        self.causal_top_k = 30
        
        self.shap_num_samples = 1000
        self.shap_max_display = 20
        
        self.visualization_dpi = 300
        self.visualization_format = 'png'
        
        self.use_tensorboard = True
        self.tensorboard_log_dir = './runs'
        
        self.resume_training = False
        self.resume_checkpoint = None
        
        self.metrics = [
            'accuracy', 'precision', 'recall', 'f1', 
            'macro_f1', 'weighted_f1', 'mcc', 'kappa'
        ]
        
        self.class_weights = None
        self.pos_weight = None
        
        self.augmentation = False
        self.noise_std = 0.01
        
        self.debug_mode = False
        self.verbose = True
        
    def update_from_dict(self, config_dict: Dict):
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown config parameter: {key}")
    
    def create_directories(self):
        os.makedirs(self.processed_data_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.result_path, exist_ok=True)
        if self.use_tensorboard:
            os.makedirs(self.tensorboard_log_dir, exist_ok=True)
    
    def get_device_info(self):
        if self.device.type == 'cuda':
            return {
                'device': 'cuda',
                'device_name': torch.cuda.get_device_name(0),
                'device_count': torch.cuda.device_count(),
                'cuda_version': torch.version.cuda
            }
        else:
            return {'device': 'cpu'}
    
    def print_config(self):
        print("=" * 80)
        print("Configuration")
        print("=" * 80)
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                print(f"{key:30s}: {value}")
        print("=" * 80)
    
    def save_config(self, path: str):
        import json
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('_') and not isinstance(v, torch.device)}
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    @classmethod
    def load_config(cls, path: str):
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        config = cls()
        config.update_from_dict(config_dict)
        return config
