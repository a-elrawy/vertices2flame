from .dataset import get_dataloader_ted, get_dataloader_vox

def get_dataloaders(args):
    ## get dataloaders
    if args.dataset == "ted":
        return get_dataloader_ted(args)
    elif args.dataset == "vox":
        return get_dataloader_vox(args)


if __name__ == "__main__":
    get_dataloaders()