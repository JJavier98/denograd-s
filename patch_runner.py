import re

with open("src/libs/experiment_runner.py", "r") as f:
    content = f.read()

old_str = """                sparse_backbone, _, _, _, _ = sparse_trainer.fit()
                sparse_report = {
                    "method": sparse_method,
                    "requested_sparsity": sparse_ratio,
                    "training_controller": sparse_controller.report(),
                    "final_stats": summarize_sparsity(
                        sparse_backbone,
                        include_bias=include_bias,
                    ),
                }"""

new_str = """                sparse_backbone, _, _, _, _ = sparse_trainer.fit()

                from src.libs.sparsity import apply_variance_threshold_compact_sparsification
                sparse_backbone, compact_report = apply_variance_threshold_compact_sparsification(
                    sparse_backbone, variance_pct=1e-8, include_bias=include_bias, inplace=True
                )

                ft_epochs = 15
                ft_lr = model_cfg.get("lr", 0.001) / 10.0
                ft_optimizer = torch.optim.Adam(sparse_backbone.parameters(), lr=ft_lr)
                ft_trainer = Trainer(
                    model=sparse_backbone,
                    train_generator=loaders_noisy[0],
                    val_generator=loaders_noisy[1],
                    device=device,
                    criterion=criterion,
                    optimizer=ft_optimizer,
                    epoch_scheduler=None,
                    batch_scheduler=None,
                    patience=5,
                    epochs=ft_epochs,
                    checkpoints_path=str(sparse_model_ckpt),
                    verbose=False,
                    sparsity_controller=None,
                )
                sparse_backbone, _, _, _, _ = ft_trainer.fit()

                sparse_report = {
                    "method": sparse_method,
                    "requested_sparsity": sparse_ratio,
                    "training_controller": sparse_controller.report(),
                    "compact_report": compact_report,
                    "final_stats": summarize_sparsity(
                        sparse_backbone,
                        include_bias=include_bias,
                    ),
                }"""

new_content = content.replace(old_str, new_str)
with open("src/libs/experiment_runner.py", "w") as f:
    f.write(new_content)
