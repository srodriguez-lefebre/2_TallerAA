# Source state manifest

Estados fuente usados para el fine-tuning noisy.

| Branch | Weight | Checkpoint SHA256 | Metadata SHA256 | Submission SHA256 | Config |
|---|---:|---|---|---|---|
| separable_headsep | 0.250 | `a6e22d4fd20457a4` | `43112e44d2ee56a8` | `b1702710324a5405` | architecture=separable_residual, activation=relu, initializer=he_normal, head_hidden=256, head_dropout=0.3, optimizer=adam, scheduler=multistep, full_train=True, seed=42, n_mels=128, frames=512, cache_tag=None, best_epoch=100, best_lwlrap=None |
| globalmel_sep_temporal | 0.375 | `b3908c3ad004db0f` | `08fcfa42c71d8249` | `3fa123950778271a` | architecture=separable_temporal_bigru, activation=silu, initializer=he_normal, head_hidden=0, head_dropout=0.3, optimizer=adamw, scheduler=multistep, full_train=True, seed=42, n_mels=128, frames=512, cache_tag=globalmel, best_epoch=100, best_lwlrap=None |
| sep_temporal_f1024 | 0.375 | `80b31e7c22277052` | `b69848f02bbd0dc7` | `69796aeb96f7e006` | architecture=separable_temporal_bigru, activation=silu, initializer=he_normal, head_hidden=0, head_dropout=0.3, optimizer=adamw, scheduler=multistep, full_train=True, seed=42, n_mels=128, frames=1024, cache_tag=None, best_epoch=100, best_lwlrap=None |
