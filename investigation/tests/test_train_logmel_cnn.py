import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "torch is only installed in the project virtualenv")
class TrainLogmelCnnTests(unittest.TestCase):
    def test_model_supports_separable_residual_architecture(self) -> None:
        from scripts.train_logmel_cnn import SmallLogmelCnn

        model = SmallLogmelCnn(
            num_classes=4,
            architecture="separable_residual",
            activation="relu",
        )
        output = model(torch.randn(2, 1, 128, 512))
        depthwise_convolutions = [
            module
            for module in model.modules()
            if isinstance(module, torch.nn.Conv2d)
            and module.groups == module.in_channels
            and module.in_channels > 1
        ]

        self.assertEqual(tuple(output.shape), (2, 4))
        self.assertGreaterEqual(len(depthwise_convolutions), 4)

    def test_model_supports_temporal_bigru_architecture(self) -> None:
        from scripts.train_logmel_cnn import SmallLogmelCnn

        model = SmallLogmelCnn(
            num_classes=4,
            architecture="temporal_bigru",
            activation="silu",
        )
        output = model(torch.randn(2, 1, 128, 512))
        recurrent_layers = [
            module
            for module in model.modules()
            if isinstance(module, torch.nn.GRU)
        ]

        self.assertEqual(tuple(output.shape), (2, 4))
        self.assertEqual(len(recurrent_layers), 1)
        self.assertTrue(recurrent_layers[0].bidirectional)

    def test_model_supports_separable_temporal_bigru_architecture(self) -> None:
        from scripts.train_logmel_cnn import SmallLogmelCnn

        model = SmallLogmelCnn(
            num_classes=4,
            architecture="separable_temporal_bigru",
            activation="silu",
        )
        output = model(torch.randn(2, 1, 128, 512))
        recurrent_layers = [
            module
            for module in model.modules()
            if isinstance(module, torch.nn.GRU)
        ]
        depthwise_convolutions = [
            module
            for module in model.modules()
            if isinstance(module, torch.nn.Conv2d)
            and module.groups == module.in_channels
            and module.in_channels > 1
        ]

        self.assertEqual(tuple(output.shape), (2, 4))
        self.assertEqual(len(recurrent_layers), 1)
        self.assertTrue(recurrent_layers[0].bidirectional)
        self.assertGreaterEqual(len(depthwise_convolutions), 4)

    def test_model_supports_separable_residual_se_architecture(self) -> None:
        from scripts.train_logmel_cnn import SmallLogmelCnn, SqueezeExcitation

        model = SmallLogmelCnn(
            num_classes=4,
            architecture="separable_residual_se",
            activation="relu",
            head_hidden=256,
        )
        output = model(torch.randn(2, 1, 128, 512))
        se_layers = [module for module in model.modules() if isinstance(module, SqueezeExcitation)]

        self.assertEqual(tuple(output.shape), (2, 4))
        self.assertGreaterEqual(len(se_layers), 3)

    def test_time_reversal_only_flips_frame_axis(self) -> None:
        from scripts.train_logmel_cnn import _augment_image

        image = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        augmented = _augment_image(
            image.clone(),
            apply_spec_augment=False,
            time_reverse_probability=1.0,
        )

        torch.testing.assert_close(augmented, image.flip(dims=(2,)))

    def test_contrast_scaling_preserves_mean_and_shape(self) -> None:
        from scripts.train_logmel_cnn import _scale_contrast

        image = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        augmented = _scale_contrast(image, factor=1.5)

        self.assertEqual(tuple(augmented.shape), tuple(image.shape))
        self.assertTrue(torch.isfinite(augmented).all())
        torch.testing.assert_close(augmented.mean(), image.mean())
        self.assertGreater(float(augmented.std()), float(image.std()))

    def test_apply_he_initialization_resets_supported_layers(self) -> None:
        from scripts.train_logmel_cnn import SmallLogmelCnn, apply_he_initialization

        model = SmallLogmelCnn(num_classes=4, head_hidden=256)
        for parameter in model.parameters():
            parameter.data.zero_()

        model.apply(apply_he_initialization)

        first_conv = model.features[0][0]
        first_batch_norm = model.features[0][1]
        hidden_layer = model.classifier[1]
        output_layer = model.classifier[-1]

        self.assertGreater(float(first_conv.weight.std()), 0.0)
        torch.testing.assert_close(first_batch_norm.weight, torch.ones_like(first_batch_norm.weight))
        torch.testing.assert_close(first_batch_norm.bias, torch.zeros_like(first_batch_norm.bias))
        self.assertGreater(float(hidden_layer.weight.std()), 0.085)
        self.assertGreater(float(output_layer.weight.std()), 0.0)
        torch.testing.assert_close(output_layer.bias, torch.zeros_like(output_layer.bias))

    def test_build_scheduler_supports_plateau_on_max_metric(self) -> None:
        from scripts.train_logmel_cnn import build_scheduler

        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)

        scheduler = build_scheduler(
            optimizer,
            scheduler_name="plateau",
            epochs=20,
            plateau_patience=3,
            plateau_factor=0.5,
            lr_milestones=[],
        )

        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        self.assertEqual(scheduler.mode, "max")
        self.assertEqual(scheduler.patience, 3)
        self.assertEqual(scheduler.factor, 0.5)

    def test_build_scheduler_supports_replayed_lr_milestones(self) -> None:
        from scripts.train_logmel_cnn import build_scheduler

        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=5e-4)

        scheduler = build_scheduler(
            optimizer,
            scheduler_name="multistep",
            epochs=50,
            plateau_patience=2,
            plateau_factor=0.5,
            lr_milestones=[33, 40, 46],
        )

        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.MultiStepLR)
        self.assertEqual(sorted(scheduler.milestones.elements()), [33, 40, 46])
        self.assertEqual(scheduler.gamma, 0.5)

    def test_model_supports_relu_and_block_dropout(self) -> None:
        from scripts.train_logmel_cnn import SmallLogmelCnn

        model = SmallLogmelCnn(
            num_classes=4,
            activation="relu",
            block_dropout=0.2,
        )

        activations = [
            module
            for module in model.modules()
            if isinstance(module, torch.nn.ReLU)
        ]
        block_dropouts = [
            module
            for module in model.features.modules()
            if isinstance(module, torch.nn.Dropout2d)
        ]

        self.assertEqual(len(activations), 8)
        self.assertEqual(len(block_dropouts), 4)
        self.assertTrue(all(module.p == 0.2 for module in block_dropouts))

    def test_model_supports_batch_normalized_hidden_head(self) -> None:
        from scripts.train_logmel_cnn import SmallLogmelCnn

        model = SmallLogmelCnn(
            num_classes=4,
            head_hidden=256,
            head_dropout=0.3,
        )

        self.assertIsInstance(model.classifier[1], torch.nn.Linear)
        self.assertEqual(model.classifier[1].out_features, 256)
        self.assertIsInstance(model.classifier[2], torch.nn.BatchNorm1d)
        self.assertEqual(model.classifier[4].p, 0.3)
        self.assertIsInstance(model.classifier[-1], torch.nn.Linear)
        self.assertEqual(model.classifier[-1].out_features, 4)

    def test_build_optimizer_supports_adam_without_weight_decay(self) -> None:
        from scripts.train_logmel_cnn import build_optimizer

        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = build_optimizer(
            [parameter],
            optimizer_name="adam",
            learning_rate=5e-4,
            weight_decay=1e-4,
        )

        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.0)


if __name__ == "__main__":
    unittest.main()
