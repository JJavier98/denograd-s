import gc
import time

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from xgboost import XGBRegressor
from tabpfn import TabPFNRegressor
from TSFEDL import OhShuLih, OhShuLih_Forecaster
from src.models.dnn import TabularDNN
from src.models.lstm import MultivariateLSTM
from src.models.xlstm_adapter import xLSTMAdapter
from src.models.dlinear_adapter import DLinearAdapter
from src.profiling import benchmark_inference, model_size_bytes
from src.trainer import Trainer
from src.utils import extract_data_from_loader

class PermuteAndRun(nn.Module):
    """
    Wrapper to adapt (N, L, C) input to (N, C, L) for CNN-based models like OhShuLih
    inside a pipeline providing (N, L, C) data.
    Also ensures RNN weights are compacted in external models to avoid UserWarnings.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Compact weights for any RNN submodule inside the external model
        # This fixes: "UserWarning: RNN module weights are not part of single contiguous chunk of memory"
        # without modifying the external library code.
        for m in self.model.modules():
            if isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)):
                m.flatten_parameters()

        # Permute (N, L, C) -> (N, C, L)
        x = x.permute(0, 2, 1)
        return self.model(x)

def run_regression_benchmark(
    loaders,
    data,
    device,
    benchmark_cfg=None,
    artifact_cache=None,
    cache_key=None,
    reuse_cached=False,
    profile_models=False,
    return_metadata=False,
):
    train_loader, val_loader, _ = loaders
    (X_train_tensor, y_train_tensor), _, (X_test_tensor, y_test_tensor) = data

    artifact_name = f"regression_{cache_key}" if cache_key else "regression"
    if reuse_cached and artifact_cache is not None:
        cached = artifact_cache.load_json(artifact_name, kind="metrics")
        if cached is not None:
            if return_metadata:
                profile_cached = artifact_cache.load_json(f"{artifact_name}_profile", kind="metrics")
                return cached, (profile_cached or {})
            return cached

    results = {}
    metadata = {
        "timings_seconds": {},
        "model_stats": {},
        "inference_profile": {},
    }

    # Convert benchmark_cfg to dict if it is a DictConfig
    if benchmark_cfg is not None and hasattr(benchmark_cfg, 'keys'):
        benchmark_cfg = dict(benchmark_cfg)

    X_train = X_train_tensor.cpu().numpy()
    y_train = y_train_tensor.cpu().numpy()
    # X_val = X_val_tensor.cpu().numpy()
    # y_val = y_val_tensor.cpu().numpy()
    X_test = X_test_tensor.cpu().numpy()
    y_test = y_test_tensor.cpu().numpy()

    # 1. Ridge
    print("  -> Training Ridge...")
    t0 = time.perf_counter()
    ridge_params = benchmark_cfg.get("ridge", {}) if benchmark_cfg else {}
    ridge = Ridge(**ridge_params)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    results['Ridge'] = mean_squared_error(y_test, y_pred_ridge)
    metadata["timings_seconds"]["Ridge"] = time.perf_counter() - t0
    del ridge; gc.collect()

    # 2. KNN
    print("  -> Training KNN...")
    t0 = time.perf_counter()
    knn_params = benchmark_cfg.get("knn", {}) if benchmark_cfg else {}
    knn = KNeighborsRegressor(**knn_params)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    results['KNN'] = mean_squared_error(y_test, y_pred_knn)
    metadata["timings_seconds"]["KNN"] = time.perf_counter() - t0
    del knn; gc.collect()

    # 3. XGBoost
    print("  -> Training XGBoost...")
    t0 = time.perf_counter()
    xgb_params = benchmark_cfg.get("xgboost", {}) if benchmark_cfg else {}
    xgb = XGBRegressor(**xgb_params)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    results['XGBoost'] = mean_squared_error(y_test, y_pred_xgb)
    metadata["timings_seconds"]["XGBoost"] = time.perf_counter() - t0
    del xgb; gc.collect()

    # 4. TabPFN
    # TabPFN uses O(n²) attention — subsample for large datasets to avoid RAM OOM
    TABPFN_MAX_SAMPLES = 3_000
    print("  -> Training TabPFN...")
    t0 = time.perf_counter()
    tabpfn_params = benchmark_cfg.get("tabpfn", {}) if benchmark_cfg else {}

    # Ensure ignore_pretraining_limits is set to True to handle large datasets
    if hasattr(tabpfn_params, 'items'):
        tabpfn_params = dict(tabpfn_params)
    tabpfn_params.setdefault('ignore_pretraining_limits', True)
    # Force CPU to avoid CUDA OOM with large datasets
    tabpfn_params['device'] = 'cpu'
    # Limit estimators to reduce RAM usage
    tabpfn_params['n_estimators'] = 1

    # TabPFN expects 1D y for regression usually, or single output
    y_train_tpfn = y_train.ravel() if y_train.ndim > 1 and y_train.shape[1] == 1 else y_train

    # Subsample training data if too large for TabPFN
    if len(X_train) > TABPFN_MAX_SAMPLES:
        print(f"     (Subsampling {TABPFN_MAX_SAMPLES}/{len(X_train)} samples for TabPFN)")
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_train), TABPFN_MAX_SAMPLES, replace=False)
        X_train_tpfn = X_train[idx]
        y_train_tpfn = y_train_tpfn[idx]
    else:
        X_train_tpfn = X_train

    tpfn = TabPFNRegressor(**tabpfn_params)
    tpfn.fit(X_train_tpfn, y_train_tpfn)
    y_pred_tpfn = tpfn.predict(X_test)
    results['TabPFN'] = mean_squared_error(y_test, y_pred_tpfn)
    metadata["timings_seconds"]["TabPFN"] = time.perf_counter() - t0
    del tpfn, X_train_tpfn, y_train_tpfn; gc.collect()

    # 5. DNN
    print("  -> Training DNN...")
    t0 = time.perf_counter()
    # Determine input dim
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1

    dnn_params = benchmark_cfg.get("dnn", {}) if benchmark_cfg else {}
    hidden_dims = dnn_params.get("hidden_dims", [128, 64, 32])
    # Ensure hidden_dims is a list (OmegaConf ListConfig -> list)
    if hasattr(hidden_dims, 'split'): # it's a string?
        pass # assume list
    else:
        hidden_dims = list(hidden_dims)

    lr = dnn_params.get("lr", 0.001)
    patience = dnn_params.get("patience", 10)

    # Simple architecture for benchmark
    model = TabularDNN(
        input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = Trainer(
        model=model,
        train_generator=train_loader,
        val_generator=val_loader,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        epoch_scheduler=None,
        batch_scheduler=None,
        patience=patience,
        epochs=dnn_params.get("max_epochs", 50),
        checkpoints_path="checkpoints/dnn_benchmark.pth"
    )

    best_nn_model, _, _, _, _ = trainer.fit()

    # Evaluate DNN
    best_nn_model.eval()
    with torch.no_grad():
        # Ensure input is 2D for DNN if it expects it, or handled by model
        y_pred_dnn = best_nn_model(X_test_tensor.to(device)).cpu().numpy()

    results['DNN'] = mean_squared_error(y_test, y_pred_dnn)
    metadata["timings_seconds"]["DNN"] = time.perf_counter() - t0
    metadata["model_stats"]["DNN"] = model_size_bytes(best_nn_model)

    if profile_models:
        try:
            example_batch = X_test_tensor[: min(64, X_test_tensor.size(0))]
            metadata["inference_profile"]["DNN"] = benchmark_inference(
                best_nn_model,
                example_batch,
                device=device,
                warmup_runs=2,
                timed_runs=5,
            )
        except (RuntimeError, ValueError, AttributeError) as e:
            metadata["inference_profile"]["DNN"] = {"error": str(e)}

    # Free GPU and CPU memory after benchmark
    del best_nn_model, model, trainer, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if artifact_cache is not None:
        artifact_cache.save_json(results, artifact_name, kind="metrics")
        artifact_cache.save_json(metadata, f"{artifact_name}_profile", kind="metrics")
        artifact_cache.update_manifest(
            artifact_name,
            {
                "metrics": str(artifact_cache.json_path(artifact_name, kind="metrics")),
                "profile": str(artifact_cache.json_path(f"{artifact_name}_profile", kind="metrics")),
            },
        )

    if return_metadata:
        return results, metadata
    return results


def run_ts_benchmark(
    loaders,
    device,
    benchmark_cfg=None,
    artifact_cache=None,
    cache_key=None,
    reuse_cached=False,
    profile_models=False,
    return_metadata=False,
):
    """
    Benchmark for Time Series.
    Adapts 3D data (N, T, D) to 2D (N, T*D) for Sklearn models.
    """
    train_loader, val_loader, test_loader = loaders

    artifact_name = f"ts_{cache_key}" if cache_key else "ts"
    if reuse_cached and artifact_cache is not None:
        cached = artifact_cache.load_json(artifact_name, kind="metrics")
        if cached is not None:
            if return_metadata:
                profile_cached = artifact_cache.load_json(f"{artifact_name}_profile", kind="metrics")
                return cached, (profile_cached or {})
            return cached

    # Extract data to CPU tensors
    X_train, y_train = extract_data_from_loader(train_loader)
    X_test, y_test = extract_data_from_loader(test_loader)

    # Shapes: (N, T, D)
    N_train, _, _ = X_train.shape
    N_test, _, _ = X_test.shape

    # Flatten for Sklearn: (N, T*D)
    X_train_flat = X_train.reshape(N_train, -1).numpy()

    # Check y shape
    # If y is (N, T, D), reshape to (N, T*D)
    # If y is (N, T), reshape to (N, T) -> OK for Ridge?
    # Ridge expects (n_samples, n_targets). If T*D > 1, it's multi-output.
    y_train_flat = y_train.reshape(N_train, -1).numpy()

    X_test_flat = X_test.reshape(N_test, -1).numpy()
    y_test_flat = y_test.reshape(N_test, -1).numpy()

    results = {}
    metadata = {
        "timings_seconds": {},
        "model_stats": {},
        "inference_profile": {},
    }

    # Convert benchmark_cfg to dict
    if benchmark_cfg is not None and hasattr(benchmark_cfg, 'keys'):
        benchmark_cfg = dict(benchmark_cfg)
    else:
        benchmark_cfg = {}

    # 1. Ridge
    print("  -> Training Ridge (Flattened)...")
    t0 = time.perf_counter()
    ridge_params = benchmark_cfg.get("ridge", {})
    ridge = Ridge(**ridge_params)
    ridge.fit(X_train_flat, y_train_flat)
    y_pred_ridge = ridge.predict(X_test_flat)
    results['Ridge'] = mean_squared_error(y_test_flat, y_pred_ridge)
    metadata["timings_seconds"]["Ridge"] = time.perf_counter() - t0

    # 2. KNN
    print("  -> Training KNN (Flattened)...")
    t0 = time.perf_counter()
    knn_params = benchmark_cfg.get("knn", {})
    knn = KNeighborsRegressor(**knn_params)
    knn.fit(X_train_flat, y_train_flat)
    y_pred_knn = knn.predict(X_test_flat)
    results['KNN'] = mean_squared_error(y_test_flat, y_pred_knn)
    metadata["timings_seconds"]["KNN"] = time.perf_counter() - t0

    # 3. XGBoost
    print("  -> Training XGBoost (Flattened)...")
    t0 = time.perf_counter()
    xgb_params = benchmark_cfg.get("xgboost", {})
    xgb = XGBRegressor(**xgb_params)
    xgb.fit(X_train_flat, y_train_flat)
    y_pred_xgb = xgb.predict(X_test_flat)
    results['XGBoost'] = mean_squared_error(y_test_flat, y_pred_xgb)
    metadata["timings_seconds"]["XGBoost"] = time.perf_counter() - t0

    # 4. DNN (Flattened)
    print("  -> Training DNN (Flattened)...")
    t0 = time.perf_counter()

    # Extract VAL data for early stopping
    X_val, y_val = extract_data_from_loader(val_loader)
    N_val = X_val.shape[0]
    X_val_flat = X_val.reshape(N_val, -1).numpy()
    y_val_flat = y_val.reshape(N_val, -1).numpy()

    # Create tensors
    X_train_tensor = torch.tensor(X_train_flat, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_flat, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_flat, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_flat, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_flat, dtype=torch.float32)

    # Create Datasets and Loaders
    train_ds_flat = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_ds_flat = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)

    batch_size = train_loader.batch_size
    train_loader_flat = torch.utils.data.DataLoader(train_ds_flat, batch_size=batch_size,
                                                    shuffle=True)
    val_loader_flat = torch.utils.data.DataLoader(val_ds_flat, batch_size=batch_size)

    # DNN Params
    input_dim = X_train_flat.shape[1]
    output_dim = y_train_flat.shape[1]

    dnn_params = benchmark_cfg.get("dnn", {})
    hidden_dims = dnn_params.get("hidden_dims", [128, 64, 32])
    if not isinstance(hidden_dims, list):
        hidden_dims = list(hidden_dims)

    lr = dnn_params.get("lr", 0.001)
    patience = dnn_params.get("patience", 10)
    max_epochs = dnn_params.get("max_epochs", 50)

    model = TabularDNN(
        input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = Trainer(
        model=model,
        train_generator=train_loader_flat,
        val_generator=val_loader_flat,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        epoch_scheduler=None,
        batch_scheduler=None,
        patience=patience,
        epochs=max_epochs,
        checkpoints_path="checkpoints/dnn_ts_benchmark.pth"
    )

    best_nn_model, _, _, _, _ = trainer.fit()

    best_nn_model.eval()
    with torch.no_grad():
        y_pred_dnn = best_nn_model(X_test_tensor.to(device)).cpu().numpy()

    results['DNN'] = mean_squared_error(y_test_flat, y_pred_dnn)
    metadata["timings_seconds"]["DNN"] = time.perf_counter() - t0
    metadata["model_stats"]["DNN"] = model_size_bytes(best_nn_model)

    if profile_models:
        try:
            example_batch = X_test_tensor[: min(64, X_test_tensor.size(0))]
            metadata["inference_profile"]["DNN"] = benchmark_inference(
                best_nn_model,
                example_batch,
                device=device,
                warmup_runs=2,
                timed_runs=5,
            )
        except (RuntimeError, ValueError, AttributeError) as e:
            metadata["inference_profile"]["DNN"] = {"error": str(e)}

    # 5. LSTM (Deep Sequence Model)
    print("  -> Training LSTM (Deep Sequence Model)...")
    t0 = time.perf_counter()

    # LSTM expects (Batch, Seq, Feat), we use X_train (unflattened) and TensorDataset
    # Time series require keeping temporal dimension

    # Create Datasets with shapes (N, T, D)
    train_ds_seq = torch.utils.data.TensorDataset(X_train.float(), y_train.float())
    val_ds_seq = torch.utils.data.TensorDataset(X_val.float(), y_val.float())

    train_loader_seq = torch.utils.data.DataLoader(train_ds_seq, batch_size=batch_size, shuffle=True)
    val_loader_seq = torch.utils.data.DataLoader(val_ds_seq, batch_size=batch_size)

    input_dim_seq = X_train.shape[2] # Feat dimension
    seq_len = X_train.shape[1]

    if y_train.dim() == 2:
        pred_len = 1
        output_dim_seq = y_train.shape[1]
    else:
        pred_len = y_train.shape[1]
        output_dim_seq = y_train.shape[2]

    lstm_params = benchmark_cfg.get("lstm", {})
    hidden_dim = lstm_params.get("hidden_dim", 64)
    num_layers = lstm_params.get("num_layers", 2)
    dropout = lstm_params.get("dropout", 0.1)
    lr = lstm_params.get("lr", 0.001)
    patience = lstm_params.get("patience", 10)
    max_epochs = lstm_params.get("max_epochs", 50)

    # Calculate output size for LSTM
    # If pred_len=1, usually flattened output (N, Out)
    # If pred_len>1, usually (N, T_out, Out)
    # MultivariateLSTM implementation returns (N, output_size)
    # We set output_size to total flattened size to match y_test_flat
    lstm_output_size = output_dim_seq * pred_len

    model_lstm = MultivariateLSTM(
        input_size=input_dim_seq,
        hidden_size=hidden_dim,
        output_size=lstm_output_size, # Assuming last step projection
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_lstm.parameters(), lr=lr)

    trainer_lstm = Trainer(
        model=model_lstm,
        train_generator=train_loader_seq,
        val_generator=val_loader_seq,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        epoch_scheduler=None,
        batch_scheduler=None,
        patience=patience,
        epochs=max_epochs,
        checkpoints_path="checkpoints/lstm_ts_benchmark.pth"
    )

    best_lstm_model, _, _, _, _ = trainer_lstm.fit()
    best_lstm_model.eval()
    with torch.no_grad():
        y_pred_lstm = best_lstm_model(X_test.float().to(device)).cpu().numpy()

    # Ensure shape match
    if y_pred_lstm.ndim != y_test_flat.ndim:
        y_pred_lstm = y_pred_lstm.reshape(y_test_flat.shape)

    results['LSTM'] = mean_squared_error(y_test_flat, y_pred_lstm)
    metadata["timings_seconds"]["LSTM"] = time.perf_counter() - t0
    metadata["model_stats"]["LSTM"] = model_size_bytes(best_lstm_model)

    if profile_models:
        try:
            example_batch = X_test[: min(64, X_test.size(0))].float()
            metadata["inference_profile"]["LSTM"] = benchmark_inference(
                best_lstm_model,
                example_batch,
                device=device,
                warmup_runs=2,
                timed_runs=5,
            )
        except (RuntimeError, ValueError, AttributeError) as e:
            metadata["inference_profile"]["LSTM"] = {"error": str(e)}

    # 6. xLSTM
    print("  -> Training xLSTM...")
    t0 = time.perf_counter()

    xlstm_params = benchmark_cfg.get("xlstm", {})
    hidden_dim_xs = xlstm_params.get("hidden_dim", 64)
    num_layers_xs = xlstm_params.get("num_layers", 2)
    lr_xs = xlstm_params.get("lr", 0.001)

    output_dim_total = output_dim_seq * pred_len

    # Check if xLSTMAdapter supports sequence output or flat
    model_xlstm = xLSTMAdapter(
        input_dim=input_dim_seq,
        output_dim=output_dim_total,
        seq_len=seq_len,
        hidden_dim=hidden_dim_xs,
        num_layers=num_layers_xs,
        dropout=dropout
    ).to(device)

    optimizer_xs = torch.optim.Adam(model_xlstm.parameters(), lr=lr_xs)

    # We use flat y for xLSTM if it predicts flat head
    train_ds_flat_y = torch.utils.data.TensorDataset(X_train.float(), y_train.reshape(X_train.shape[0], -1).float())
    val_ds_flat_y = torch.utils.data.TensorDataset(X_val.float(), y_val.reshape(X_val.shape[0], -1).float())

    train_loader_xs = torch.utils.data.DataLoader(train_ds_flat_y, batch_size=batch_size, shuffle=True)
    val_loader_xs = torch.utils.data.DataLoader(val_ds_flat_y, batch_size=batch_size)

    trainer_xlstm = Trainer(
        model=model_xlstm,
        train_generator=train_loader_xs,
        val_generator=val_loader_xs,
        device=device,
        criterion=criterion,
        optimizer=optimizer_xs,
        epoch_scheduler=None,
        batch_scheduler=None,
        patience=patience,
        epochs=max_epochs,
        checkpoints_path="checkpoints/xlstm_ts_benchmark.pth"
    )

    try:
        best_xlstm_model, _, _, _, _ = trainer_xlstm.fit()
        best_xlstm_model.eval()
        with torch.no_grad():
            y_pred_xlstm = best_xlstm_model(X_test.float().to(device)).cpu().numpy()
        results['xLSTM'] = mean_squared_error(y_test_flat, y_pred_xlstm)
        metadata["timings_seconds"]["xLSTM"] = time.perf_counter() - t0
        metadata["model_stats"]["xLSTM"] = model_size_bytes(best_xlstm_model)

        if profile_models:
            try:
                example_batch = X_test[: min(64, X_test.size(0))].float()
                metadata["inference_profile"]["xLSTM"] = benchmark_inference(
                    best_xlstm_model,
                    example_batch,
                    device=device,
                    warmup_runs=2,
                    timed_runs=5,
                )
            except (RuntimeError, ValueError, AttributeError) as e:
                metadata["inference_profile"]["xLSTM"] = {"error": str(e)}
    except (RuntimeError, ValueError, AttributeError) as e:
        print(f"    [Warning] xLSTM failed: {e}")
        results['xLSTM'] = float('inf')
        metadata["timings_seconds"]["xLSTM"] = time.perf_counter() - t0
        metadata["inference_profile"]["xLSTM"] = {"error": str(e)}

    # 7. OhShuLih (TSFEDL)
    print("  -> Training OhShuLih (TSFEDL)...")
    t0 = time.perf_counter()
    oh_params = benchmark_cfg.get("ohshulih", {})

    try:
        # Replace top_module
        top_module = OhShuLih_Forecaster(
            out_features=output_dim_seq,
            n_pred=pred_len
        )

        # Instantiate base model
        model_oh = OhShuLih(
            in_features=input_dim_seq,
            top_module=top_module,
            loss=criterion,
            dropout=oh_params.get("dropout", 0.1)
        )

        model_oh_adapted = PermuteAndRun(model_oh).to(device)

        # Use Trainer
        optimizer_oh = torch.optim.Adam(model_oh_adapted.parameters(), lr=lr)

        trainer_oh = Trainer(
            model=model_oh_adapted,
            train_generator=train_loader_seq,
            val_generator=val_loader_seq,
            device=device,
            criterion=criterion,
            optimizer=optimizer_oh,
            epoch_scheduler=None,
            batch_scheduler=None,
            patience=patience,
            epochs=max_epochs,
            checkpoints_path="checkpoints/ohshulih_ts_benchmark.pth"
        )

        best_oh, _, _, _, _ = trainer_oh.fit()
        best_oh.eval()
        with torch.no_grad():
            y_pred_oh = best_oh(X_test.float().to(device)).cpu().numpy()

        if y_pred_oh.ndim != y_test_flat.ndim:
            y_pred_oh = y_pred_oh.reshape(y_test_flat.shape)

        results['OhShuLih'] = mean_squared_error(y_test_flat, y_pred_oh)
        metadata["timings_seconds"]["OhShuLih"] = time.perf_counter() - t0
        metadata["model_stats"]["OhShuLih"] = model_size_bytes(best_oh)

        if profile_models:
            try:
                example_batch = X_test[: min(64, X_test.size(0))].float()
                metadata["inference_profile"]["OhShuLih"] = benchmark_inference(
                    best_oh,
                    example_batch,
                    device=device,
                    warmup_runs=2,
                    timed_runs=5,
                )
            except (RuntimeError, ValueError, AttributeError) as e:
                metadata["inference_profile"]["OhShuLih"] = {"error": str(e)}

    except (RuntimeError, ValueError, AttributeError) as e:
        print(f"    [Warning] OhShuLih failed: {e}")
        results['OhShuLih'] = float('inf')
        metadata["timings_seconds"]["OhShuLih"] = time.perf_counter() - t0
        metadata["inference_profile"]["OhShuLih"] = {"error": str(e)}

    # 8. DLinear
    print("  -> Training DLinear...")
    t0 = time.perf_counter()

    dlinear_params = benchmark_cfg.get("dlinear", {})

    model_dlinear = DLinearAdapter(
        seq_len=seq_len,
        pred_len=pred_len,
        input_dim=input_dim_seq,
        output_dim=output_dim_total,
        moving_avg=dlinear_params.get("moving_avg", 25),
        individual=dlinear_params.get("individual", False)
    ).to(device)

    # Wrap DLinear to flatten output for compatibility with simple Trainer/MSE
    class FlattenWrapper(nn.Module):
        def __init__(self, model, target_dim):
            super().__init__()
            self.model = model
            self.target_dim = target_dim

        def forward(self, x):
            out = self.model(x) # (N, PredLen, InputDim)
            # Select only target channels (first target_dim channels)
            # as our dataset puts y first.
            out = out[:, :, :self.target_dim]
            return out.reshape(out.size(0), -1)

    model_dlinear_flat = FlattenWrapper(model_dlinear, target_dim=output_dim_seq).to(device)
    optimizer_dl = torch.optim.Adam(model_dlinear_flat.parameters(), lr=lr)
    trainer_dl = Trainer(
        model=model_dlinear_flat,
        train_generator=train_loader_xs, # reusing flat y loader
        val_generator=val_loader_xs,
        device=device,
        criterion=criterion,
        optimizer=optimizer_dl,
        epoch_scheduler=None,
        batch_scheduler=None,
        patience=patience,
        epochs=max_epochs,
        checkpoints_path="checkpoints/dlinear_ts_benchmark.pth"
    )

    try:
        best_dl, _, _, _, _ = trainer_dl.fit()
        best_dl.eval()
        with torch.no_grad():
            y_pred_dl = best_dl(X_test.float().to(device)).cpu().numpy()
        results['DLinear'] = mean_squared_error(y_test_flat, y_pred_dl)
        metadata["timings_seconds"]["DLinear"] = time.perf_counter() - t0
        metadata["model_stats"]["DLinear"] = model_size_bytes(best_dl)

        if profile_models:
            try:
                example_batch = X_test[: min(64, X_test.size(0))].float()
                metadata["inference_profile"]["DLinear"] = benchmark_inference(
                    best_dl,
                    example_batch,
                    device=device,
                    warmup_runs=2,
                    timed_runs=5,
                )
            except (RuntimeError, ValueError, AttributeError) as e:
                metadata["inference_profile"]["DLinear"] = {"error": str(e)}
    except (RuntimeError, ValueError, AttributeError) as e:
        print(f"    [Warning] DLinear failed: {e}")
        results['DLinear'] = float('inf')
        metadata["timings_seconds"]["DLinear"] = time.perf_counter() - t0
        metadata["inference_profile"]["DLinear"] = {"error": str(e)}

    if artifact_cache is not None:
        artifact_cache.save_json(results, artifact_name, kind="metrics")
        artifact_cache.save_json(metadata, f"{artifact_name}_profile", kind="metrics")
        artifact_cache.update_manifest(
            artifact_name,
            {
                "metrics": str(artifact_cache.json_path(artifact_name, kind="metrics")),
                "profile": str(artifact_cache.json_path(f"{artifact_name}_profile", kind="metrics")),
            },
        )

    if return_metadata:
        return results, metadata
    return results
