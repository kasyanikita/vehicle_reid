import torchreid
import examples.vric_dataset
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("model_name")
    parser.add_argument("--log_dir")
    parser.add_argument("--bs", default=16, type=int)
    parser.add_argument("--loss", choices=["softmax", "triplet"], default="softmax")
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--max_epoch", default=60, type=int)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    datamanager = torchreid.data.ImageDataManager(
        sources="vric", # name of dataset set in the example/vric_dataset.py file "torchreid.data.register_image_dataset('vric', VRICDataset)"
        targets="vric",
        height=208,
        width=208,
        batch_size_train=args.bs,
        batch_size_test=args.bs,
        transforms=["random_flip", "random_crop"]
    )

    if args.loss == "softmax":
        model = torchreid.models.build_model(
            name=args.model_name,
            num_classes=datamanager.num_train_pids,
            loss="softmax",
            pretrained=True
        )
    else:
        model = torchreid.models.build_model(
            name=args.model_name,
            num_classes=datamanager.num_train_pids,
            loss="triplet",
            pretrained=True
        )

    model = model.cuda()

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim="adam",
        lr=args.lr
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler="single_step",
        stepsize=20
    )

    if args.loss == "softmax":
        engine = torchreid.engine.ImageSoftmaxEngine(
            datamanager,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            label_smooth=True
        )
    else:
        engine = torchreid.engine.ImageTripletEngine(
            datamanager,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            label_smooth=True
        )

    engine.run(
        save_dir=args.log_dir,
        max_epoch=args.max_epoch,
        eval_freq=5,
        print_freq=200,
        test_only=False
    )