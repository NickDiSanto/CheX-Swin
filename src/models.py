from timm import create_model


def build_model(args):
    pretrained = args.init == "ImageNet"
    print(f"Creating model {args.model_name} with {args.init} weights...")

    model_map = {
        "resnet18": args.model_name,
        "resnet50": args.model_name,
        "swin_tiny": "swin_tiny_patch4_window7_224",
        "swin_base": "swin_base_patch4_window7_224",
    }

    if args.model_name in model_map:
        model = create_model(
            model_name=model_map[args.model_name],
            pretrained=pretrained,
            num_classes=args.num_classes,
            in_chans=args.in_chans,
        )
    else:
        raise Exception(f"Model {args.model_name} is not implemented. Please provide a valid model name.")

    return model
