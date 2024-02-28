import torchreid
import vric_dataset
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("model_name")
    parser.add_argument("--log_dir")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    datamanager = torchreid.data.ImageDataManager(
        sources="vric",
        targets="vric",
        height=208,
        width=208,
        batch_size_train=16,
        batch_size_test=32,
        transforms=["random_flip", "random_crop"]
    )

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
        lr=0.0003
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler="single_step",
        stepsize=20
    )

    engine = torchreid.engine.ImageTripletEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

    engine.run(
        save_dir=args.log_dir,
        max_epoch=60,
        eval_freq=5,
        print_freq=200,
        test_only=False
    )