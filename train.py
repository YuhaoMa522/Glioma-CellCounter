from util import *

def main():
    set_seed(1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dataset = SegmentationDataset('data/train_images', 'data/train_labels', transform_image=transforms.ToTensor(), joint_transform=random_joint_transform)
    test_dataset = SegmentationDataset('data/test_images', 'data/test_labels', transform_image=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    class_weights = compute_class_weights(train_dataset).to(device)
    model = NestedUNet(in_ch=3, out_ch=3, deep_supervision=False).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    print("Training...")
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=50)
    print("Testing on labeled test set...")
    test_model(model, test_loader, criterion, device)
    torch.save(model.state_dict(), "model.pth")
    save_final_labeled_results_and_counts(model, test_dataset, device)
    compute_and_plot_train_roc(model, train_loader, device)

if __name__ == '__main__':
    main()