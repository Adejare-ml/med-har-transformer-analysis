import torch
from torch.utils.data import DataLoader, TensorDataset
from src.model import TimeSeriesTransformer
from src.pipeline import HARPipeline

def run_experiment():
    # Configuration
    DATA_PATH = "data/har70plus" # Update this to the real dataset path
    WINDOW_SIZE = 100
    STEP = 50
    EPOCHS = 10
    LR = 0.001
    
    pipeline = HARPipeline(window_size=WINDOW_SIZE, step=STEP)
    
    try:
        # 1. Data Prep
        X, y, class_names = pipeline.load_and_preprocess(DATA_PATH)
        
        # Split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)
        
        # 2. Initialize Model
        input_dim = X.shape[1] if len(X.shape) == 2 else X.shape[2]
        model = TimeSeriesTransformer(input_dim=input_dim, num_classes=len(class_names))
        
        # 3. Train
        pipeline.train(model, train_loader, epochs=EPOCHS, lr=LR)
        
        # 4. Evaluate
        pipeline.evaluate(model, test_loader, class_names)
        
    except Exception as e:
        print(f"Experiment failed: {e}")

if __name__ == "__main__":
    run_experiment()
