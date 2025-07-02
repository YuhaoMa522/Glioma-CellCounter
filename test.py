from util import *
import csv

def main():
    set_seed(1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_dataset = SegmentationDataset('data/test_images', 'data/test_labels', transform_image=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    unlabeled = None
    if os.path.exists('data/unlabeled_images'):
        unlabeled = UnlabeledDataset('data/unlabeled_images', transform=transforms.ToTensor())
    weights = compute_class_weights(test_dataset).to(device)
    model = NestedUNet(in_ch=3, out_ch=3, deep_supervision=False).to(device)
    if os.path.exists('model.pth'):
        model.load_state_dict(torch.load('model.pth', map_location=device))
    else:
        print('Model file not found.'); return
    criterion = nn.CrossEntropyLoss(weight=weights)
    print("Testing on labeled test set...")
    test_model(model, test_loader, criterion, device)
    print("Saving visualized results...")
    if unlabeled:
        print("Processing unlabeled...")
        results = save_final_unlabeled_results_and_counts(model, unlabeled, device)
        csv_path = os.path.join('final_result_labeled', 'unlabeled_results.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['img_idx','cellA','cellB'])
            writer.writerows(results)
        print(f'Unlabeled CSV saved to {csv_path}')
    print("Computing macro recall...")
    names, recalls = compute_macro_recall_for_cells(model, test_dataset, device)
    plot_macro_recall_bar_chart(names, recalls, os.path.join('final_result_labeled','macro_recall_barplot.png'))
    print("Plotting ROC combined...")
    roc_per_sample_and_plot(model, test_dataset, device)

if __name__ == '__main__':
    main()
