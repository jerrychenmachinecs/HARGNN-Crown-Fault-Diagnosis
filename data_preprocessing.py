import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os
from typing import List, Dict, Tuple, Optional
from config import Config


class DataPreprocessor:
    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.stand_feature_indices = {}
        
    def load_raw_data(self) -> pd.DataFrame:
        data_frames = []
        for stand in self.config.stand_names:
            file_path = os.path.join(
                self.config.raw_data_path, 
                f'stand_{stand}_data.csv'
            )
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['stand'] = stand
                data_frames.append(df)
        
        if not data_frames:
            raise FileNotFoundError("No data files found")
        
        combined_df = pd.concat(data_frames, ignore_index=True)
        return combined_df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_ratio = df.isnull().sum() / len(df)
        print(f"Missing value ratios:\n{missing_ratio[missing_ratio > 0]}")
        
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                if missing_ratio[col] < 0.05:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mean(), inplace=True)
        
        return df
    
    def detect_outliers(self, df: pd.DataFrame, 
                       columns: Optional[List[str]] = None,
                       method: str = 'iqr',
                       threshold: float = 3.0) -> pd.DataFrame:
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        mask = pd.Series([True] * len(df))
        
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                col_mask = z_scores < threshold
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
            
            mask = mask & col_mask
        
        outlier_count = (~mask).sum()
        print(f"Detected {outlier_count} outliers ({outlier_count/len(df)*100:.2f}%)")
        
        return df[mask].reset_index(drop=True)
    
    def create_time_features(self, df: pd.DataFrame, 
                            param_cols: List[str]) -> pd.DataFrame:
        if not self.config.use_time_features:
            return df
        
        time_window = self.config.time_window
        
        for col in param_cols:
            df[f'{col}_mean'] = df.groupby('coil_id')[col].transform(
                lambda x: x.rolling(window=time_window, min_periods=1).mean()
            )
            df[f'{col}_std'] = df.groupby('coil_id')[col].transform(
                lambda x: x.rolling(window=time_window, min_periods=1).std()
            ).fillna(0)
            df[f'{col}_diff'] = df.groupby('coil_id')[col].diff().fillna(0)
        
        return df
    
    def create_physical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for stand in self.config.stand_names:
            stand_mask = df['stand'] == stand
            
            if 'rolling_force' in df.columns and 'entry_thickness' in df.columns:
                df.loc[stand_mask, 'unit_rolling_force'] = (
                    df.loc[stand_mask, 'rolling_force'] / 
                    (df.loc[stand_mask, 'entry_thickness'] + 1e-7)
                )
            
            if 'bending_force' in df.columns and 'rolling_force' in df.columns:
                df.loc[stand_mask, 'bending_force_ratio'] = (
                    df.loc[stand_mask, 'bending_force'] / 
                    (df.loc[stand_mask, 'rolling_force'] + 1e-7)
                )
            
            if 'entry_thickness' in df.columns and 'exit_thickness' in df.columns:
                df.loc[stand_mask, 'reduction_rate'] = (
                    (df.loc[stand_mask, 'entry_thickness'] - 
                     df.loc[stand_mask, 'exit_thickness']) /
                    (df.loc[stand_mask, 'entry_thickness'] + 1e-7)
                )
            
            if 'entry_temperature' in df.columns and 'exit_temperature' in df.columns:
                df.loc[stand_mask, 'temperature_drop'] = (
                    df.loc[stand_mask, 'entry_temperature'] - 
                    df.loc[stand_mask, 'exit_temperature']
                )
        
        return df
    
    def create_crown_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        crown_col = 'exit_crown'
        if crown_col not in df.columns:
            crown_col = 'calculated_crown'
        
        if crown_col not in df.columns:
            raise ValueError(f"Crown column not found in dataframe")
        
        lower_threshold, upper_threshold = self.config.crown_thresholds
        
        conditions = [
            df[crown_col] < lower_threshold,
            (df[crown_col] >= lower_threshold) & (df[crown_col] <= upper_threshold),
            df[crown_col] > upper_threshold
        ]
        choices = [0, 1, 2]
        
        df['crown_label'] = np.select(conditions, choices, default=1)
        
        return df
    
    def normalize_features(self, train_df: pd.DataFrame, 
                          val_df: pd.DataFrame, 
                          test_df: pd.DataFrame,
                          feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.config.use_standardization:
            train_df[feature_cols] = self.scaler.fit_transform(train_df[feature_cols])
            val_df[feature_cols] = self.scaler.transform(val_df[feature_cols])
            test_df[feature_cols] = self.scaler.transform(test_df[feature_cols])
        
        return train_df, val_df, test_df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        unique_coils = df['coil_id'].unique()
        np.random.seed(self.config.random_seed)
        np.random.shuffle(unique_coils)
        
        train_size = int(len(unique_coils) * self.config.train_ratio)
        val_size = int(len(unique_coils) * self.config.val_ratio)
        
        train_coils = unique_coils[:train_size]
        val_coils = unique_coils[train_size:train_size + val_size]
        test_coils = unique_coils[train_size + val_size:]
        
        train_df = df[df['coil_id'].isin(train_coils)].reset_index(drop=True)
        val_df = df[df['coil_id'].isin(val_coils)].reset_index(drop=True)
        test_df = df[df['coil_id'].isin(test_coils)].reset_index(drop=True)
        
        print(f"Data split:")
        print(f"  Train: {len(train_coils)} coils, {len(train_df)} samples")
        print(f"  Val: {len(val_coils)} coils, {len(val_df)} samples")
        print(f"  Test: {len(test_coils)} coils, {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def compute_class_weights(self, train_df: pd.DataFrame) -> np.ndarray:
        label_counts = train_df['crown_label'].value_counts().sort_index()
        total = len(train_df)
        num_classes = self.config.num_classes
        
        if self.config.use_cost_sensitive:
            beta = self.config.cost_beta
            epsilon = self.config.cost_epsilon
            
            weights = []
            for i in range(num_classes):
                count = label_counts.get(i, 0)
                effective_num = 1.0 - np.power(beta, count)
                weight = (1.0 - beta) / (effective_num + epsilon)
                weights.append(weight)
            
            weights = np.array(weights)
        else:
            weights = total / (num_classes * label_counts.values)
        
        weights = weights / weights.sum() * num_classes
        
        print(f"Class weights: {weights}")
        return weights
    
    def create_graph_structure(self, num_params: int, 
                              method: str = 'fully_connected') -> Dict:
        if method == 'fully_connected':
            edge_index = []
            for i in range(num_params):
                for j in range(num_params):
                    if i != j:
                        edge_index.append([i, j])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_weight = torch.ones(edge_index.size(1))
        
        elif method == 'domain_knowledge':
            force_params = [0, 1]
            position_params = [2]
            thermal_params = [8, 9, 13]
            geometric_params = [3, 5, 10, 11]
            
            edge_index = []
            for i in range(num_params):
                for j in range(num_params):
                    if i != j:
                        if (i in force_params and j in geometric_params) or \
                           (i in thermal_params and j in force_params) or \
                           (i in position_params and j in geometric_params):
                            edge_index.append([i, j])
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_weight = torch.ones(edge_index.size(1))
        
        else:
            raise ValueError(f"Unknown graph construction method: {method}")
        
        graph_structure = {
            'edge_index': edge_index,
            'edge_weight': edge_weight,
            'num_nodes': num_params
        }
        
        return graph_structure
    
    def prepare_sequence_data(self, df: pd.DataFrame, 
                             feature_cols: List[str]) -> List[Dict]:
        sequences = []
        
        for coil_id in df['coil_id'].unique():
            coil_data = df[df['coil_id'] == coil_id].sort_values('stand')
            
            if len(coil_data) != self.config.num_stands:
                continue
            
            stand_features = []
            for stand in self.config.stand_names:
                stand_data = coil_data[coil_data['stand'] == stand]
                if len(stand_data) == 0:
                    continue
                
                features = stand_data[feature_cols].values[0]
                stand_features.append(features)
            
            if len(stand_features) != self.config.num_stands:
                continue
            
            label = coil_data['crown_label'].mode()[0]
            
            sequence = {
                'coil_id': coil_id,
                'features': np.array(stand_features),
                'label': int(label)
            }
            sequences.append(sequence)
        
        return sequences
    
    def save_processed_data(self, train_sequences: List[Dict],
                           val_sequences: List[Dict],
                           test_sequences: List[Dict]):
        data = {
            'train': train_sequences,
            'val': val_sequences,
            'test': test_sequences,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'config': self.config
        }
        
        save_path = os.path.join(
            self.config.processed_data_path, 
            'processed_data.pkl'
        )
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Processed data saved to {save_path}")
    
    def process_pipeline(self):
        print("Starting data preprocessing pipeline...")
        
        print("1. Loading raw data...")
        df = self.load_raw_data()
        print(f"   Loaded {len(df)} raw samples")
        
        print("2. Handling missing values...")
        df = self.handle_missing_values(df)
        
        print("3. Detecting and removing outliers...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = self.detect_outliers(df, columns=numeric_cols)
        
        print("4. Creating crown labels...")
        df = self.create_crown_labels(df)
        
        print("5. Creating time-based features...")
        param_cols = [col for col in self.config.param_names if col in df.columns]
        df = self.create_time_features(df, param_cols)
        
        print("6. Creating physical features...")
        df = self.create_physical_features(df)
        
        print("7. Splitting data...")
        train_df, val_df, test_df = self.split_data(df)
        
        print("8. Computing class weights...")
        class_weights = self.compute_class_weights(train_df)
        self.config.class_weights = class_weights
        
        feature_cols = [col for col in df.columns if col not in 
                       ['coil_id', 'stand', 'crown_label', 'exit_crown', 'calculated_crown']]
        self.feature_names = feature_cols
        
        print("9. Normalizing features...")
        train_df, val_df, test_df = self.normalize_features(
            train_df, val_df, test_df, feature_cols
        )
        
        print("10. Preparing sequence data...")
        train_sequences = self.prepare_sequence_data(train_df, feature_cols)
        val_sequences = self.prepare_sequence_data(val_df, feature_cols)
        test_sequences = self.prepare_sequence_data(test_df, feature_cols)
        
        print(f"   Train sequences: {len(train_sequences)}")
        print(f"   Val sequences: {len(val_sequences)}")
        print(f"   Test sequences: {len(test_sequences)}")
        
        print("11. Saving processed data...")
        self.save_processed_data(train_sequences, val_sequences, test_sequences)
        
        print("Data preprocessing completed!")
        
        return train_sequences, val_sequences, test_sequences


def main():
    config = Config()
    config.create_directories()
    
    preprocessor = DataPreprocessor(config)
    train_seq, val_seq, test_seq = preprocessor.process_pipeline()


if __name__ == '__main__':
    main()
