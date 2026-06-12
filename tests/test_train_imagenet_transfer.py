import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "torch is only installed in the project virtualenv")
class TrainImagenetTransferTests(unittest.TestCase):
    def test_full_train_scheduler_replays_learning_rate_milestones(self) -> None:
        from scripts.train_imagenet_transfer import build_transfer_scheduler

        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        scheduler = build_transfer_scheduler(
            optimizer,
            full_train=True,
            epochs=19,
            lr_milestones=[17],
        )

        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.MultiStepLR)
        self.assertEqual(sorted(scheduler.milestones.elements()), [17])

    def test_prepare_imagenet_batch_resizes_and_normalizes_logmels(self) -> None:
        from scripts.train_imagenet_transfer import prepare_imagenet_batch

        images = torch.linspace(-3.0, 4.0, steps=2 * 128 * 512).reshape(2, 1, 128, 512)
        prepared = prepare_imagenet_batch(images)

        self.assertEqual(tuple(prepared.shape), (2, 3, 224, 224))
        self.assertTrue(torch.isfinite(prepared).all())

    def test_resnet50_transfer_freezes_backbone_and_trains_head(self) -> None:
        from scripts.train_imagenet_transfer import build_resnet50_transfer

        model = build_resnet50_transfer(num_classes=80, weights=None)
        output = model(torch.randn(2, 3, 224, 224))

        self.assertEqual(tuple(output.shape), (2, 80))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.backbone.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.classifier.parameters()))


if __name__ == "__main__":
    unittest.main()
