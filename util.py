
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_joint_transform(image, label):
    if random.random() > 0.5:
        image = TF.hflip(image)
        label = TF.hflip(label)
    angle = random.uniform(-15, 15)
    image = TF.rotate(image, angle, interpolation=Image.BILINEAR)
    label = TF.rotate(label, angle, interpolation=Image.NEAREST)
    image = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)(image)
    return image, label


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform_image=None, transform_label=None, joint_transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform_image = transform_image
        self.transform_label = transform_label
        self.joint_transform = joint_transform
        self.image_names = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))])
    def __len__(self):
        return len(self.image_names)
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert('RGB')
        label = Image.open(os.path.join(self.label_dir, img_name)).convert('L')
        if self.joint_transform:
            image, label = self.joint_transform(image, label)
        if self.transform_image:
            image = self.transform_image(image)
        else:
            image = transforms.ToTensor()(image)
        label_np = np.array(label, dtype=np.uint8)
        mapping = {0: 0, 120: 1, 240: 2}
        for old_val, new_val in mapping.items():
            label_np[label_np == old_val] = new_val
        if self.transform_label:
            label_img = Image.fromarray(label_np)
            label_tensor = self.transform_label(label_img).long()
        else:
            label_tensor = torch.from_numpy(label_np).long()
        return image, label_tensor

class UnlabeledDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))])
    def __len__(self):
        return len(self.image_names)
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        return image, img_name


def upsample(x, target):
    return F.interpolate(x, size=target.size()[2:], mode='bilinear', align_corners=True)

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class NestedUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        n1, n2, n3, n4, n5 = 64, 128, 256, 512, 1024
        self.pool = nn.MaxPool2d(2, 2)
        self.conv0_0 = conv_block(in_ch, n1)
        self.conv1_0 = conv_block(n1, n2)
        self.conv2_0 = conv_block(n2, n3)
        self.conv3_0 = conv_block(n3, n4)
        self.conv4_0 = conv_block(n4, n5)
        self.conv0_1 = conv_block(n1 + n2, n1)
        self.conv1_1 = conv_block(n2 + n3, n2)
        self.conv2_1 = conv_block(n3 + n4, n3)
        self.conv3_1 = conv_block(n4 + n5, n4)
        self.conv0_2 = conv_block(n1 * 2 + n2, n1)
        self.conv1_2 = conv_block(n2 * 2 + n3, n2)
        self.conv2_2 = conv_block(n3 * 2 + n4, n3)
        self.conv0_3 = conv_block(n1 * 3 + n2, n1)
        self.conv1_3 = conv_block(n2 * 3 + n3, n2)
        self.conv0_4 = conv_block(n1 * 4 + n2, n1)
        if self.deep_supervision:
            self.final1 = nn.Conv2d(n1, out_ch, kernel_size=1)
            self.final2 = nn.Conv2d(n1, out_ch, kernel_size=1)
            self.final3 = nn.Conv2d(n1, out_ch, kernel_size=1)
            self.final4 = nn.Conv2d(n1, out_ch, kernel_size=1)
        else:
            self.final = nn.Conv2d(n1, out_ch, kernel_size=1)
    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, upsample(x1_0, x0_0)], dim=1))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, upsample(x2_0, x1_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, upsample(x1_1, x0_0)], dim=1))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, upsample(x3_0, x2_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, upsample(x2_1, x1_0)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, upsample(x1_2, x0_0)], dim=1))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, upsample(x4_0, x3_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, upsample(x3_1, x2_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, upsample(x2_2, x1_0)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, upsample(x1_3, x0_0)], dim=1))
        if self.deep_supervision:
            return [self.final1(x0_1), self.final2(x0_2), self.final3(x0_3), self.final4(x0_4)]
        else:
            return self.final(x0_4)

def compute_class_weights(dataset):
    class_counts = np.zeros(3, dtype=np.float64)
    total_pixels = 0
    for _, label in dataset:
        arr = label.numpy()
        total_pixels += arr.size
        for c in range(3):
            class_counts[c] += np.sum(arr == c)
    eps = 1e-6
    weights = total_pixels / (3.0 * (class_counts + eps))
    return torch.tensor(weights, dtype=torch.float32)

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=50):
    model.train()
    epoch_losses, epoch_accuracies = [], []
    for epoch in range(num_epochs):
        running_loss, total_correct, total_pixels = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_pixels += labels.numel()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = total_correct / total_pixels
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_acc)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    plt.figure(); plt.plot(range(1, num_epochs+1), epoch_losses); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.savefig('final_result_labeled/training_loss.png'); plt.close()
    plt.figure(); plt.plot(range(1, num_epochs+1), epoch_accuracies); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.savefig('final_result_labeled/training_accuracy.png'); plt.close()
    return epoch_losses, epoch_accuracies

def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss, total_correct, total_pixels = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_pixels += labels.numel()
    test_loss /= len(test_loader.dataset)
    test_acc = total_correct / total_pixels
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

def color_map(seg):
    h, w = seg.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    palette = {0: (0,0,0), 1: (255,0,0), 2: (0,255,0)}
    for c, clr in palette.items():
        colored[seg == c] = clr
    return colored


def filter_small_components(binary_mask, min_size=20):
    kernel = np.ones((3,3), np.uint8)
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    mask = np.zeros_like(opened, dtype=np.uint8)
    count = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            mask[labels == i] = 1
            count += 1
    return mask, count

def save_final_labeled_results_and_counts(model, dataset, device, save_dir='final_result_labeled', min_size=20):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    results = []
    for idx in range(len(dataset)):
        image, label = dataset[idx]
        name = os.path.splitext(dataset.image_names[idx])[0]
        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(device)).argmax(dim=1).squeeze(0).cpu().numpy()
        A_mask = (pred == 1).astype(np.uint8)
        B_mask = (pred == 2).astype(np.uint8)
        A_f, A_count = filter_small_components(A_mask, min_size)
        B_f, B_count = filter_small_components(B_mask, min_size)
        combined = A_f * 1 + B_f * 2
        results.append((name, A_count, B_count))
        img_np = image.permute(1,2,0).cpu().numpy()
        gt_col = color_map(label.numpy())
        pred_col = color_map(combined)
        fig, axes = plt.subplots(1,3,figsize=(15,5))
        for ax, im, ttl in zip(axes, [img_np, gt_col, pred_col], ['Input','GT','Filtered']):
            ax.imshow(im); ax.set_title(ttl); ax.axis('off')
        plt.savefig(os.path.join(save_dir, f"{name}.png")); plt.close(fig)

    names = [r[0] for r in results]
    A_counts = [r[1] for r in results]
    B_counts = [r[2] for r in results]
    x = np.arange(len(names))
    width = 0.35
    plt.figure(figsize=(10,6))
    plt.bar(x - width/2, A_counts, width, label='A')
    plt.bar(x + width/2, B_counts, width, label='B')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cell_count_barplot.pdf'))
    plt.close()

def save_final_unlabeled_results_and_counts(model, dataset, device, save_dir='final_result_unlabeled', min_size=20):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    results = []
    for img, name in dataset:
        base = os.path.splitext(name)[0]
        with torch.no_grad():
            p = model(img.unsqueeze(0).to(device)).argmax(dim=1).squeeze(0).cpu().numpy()
        A, cntA = filter_small_components((p==1).astype(np.uint8), min_size)
        B, cntB = filter_small_components((p==2).astype(np.uint8), min_size)
        results.append((base, cntA, cntB))
        img_np = img.permute(1,2,0).cpu().numpy()
        pc = color_map(A*1 + B*2)
        fig, ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].imshow(img_np); ax[0].set_title('Input'); ax[0].axis('off')
        ax[1].imshow(pc); ax[1].set_title(f'Filtered A={cntA}, B={cntB}'); ax[1].axis('off')
        plt.savefig(os.path.join(save_dir, f"{base}_filtered.png")); plt.close(fig)
    return results

def compute_and_plot_train_roc(model, train_loader, device, num_classes=3,
                                save_csv_dir='auc_results'):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            probs = probs.permute(0, 2, 3, 1).reshape(-1, num_classes).cpu().numpy()  # [N, C]
            labels = labels.view(-1).cpu().numpy()  # [N]

            all_probs.append(probs)
            all_labels.append(labels)


    all_probs = np.concatenate(all_probs, axis=0)  # [N, C]
    all_labels = np.concatenate(all_labels, axis=0)  # [N]
    labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))  # [N, C]

    if not os.path.exists(save_csv_dir):
        os.makedirs(save_csv_dir)


    np.savetxt(os.path.join(save_csv_dir, "train_probs.csv"), all_probs, delimiter=",")
    np.savetxt(os.path.join(save_csv_dir, "train_labels_bin.csv"), labels_bin, delimiter=",")
    print(f"Saved train_probs.csv and train_labels_bin.csv for further use.")


    plt.figure(figsize=(8, 6))
    auc_list = []
    interp_tprs = []
    mean_fpr = np.linspace(0, 1, 1000)


    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        auc_list.append(roc_auc)


        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)


        df = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
        csv_path = os.path.join(save_csv_dir, f'class_{i}_roc.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved ROC data for class {i} to {csv_path} (AUC = {roc_auc:.4f})")


        plt.plot(fpr, tpr, lw=1.5, label=f'Class {i} (AUC = {roc_auc:.2f})')


    macro_auc_numeric = np.mean(auc_list)
    print(f"Macro-Averaged AUC (simple mean of per-class AUC): {macro_auc_numeric:.4f}")


    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[-1] = 1.0
    macro_auc_geom = auc(mean_fpr, mean_tpr)
    np.savetxt(os.path.join(save_csv_dir, "mean_fpr.csv"), mean_fpr, delimiter=",")
    np.savetxt(os.path.join(save_csv_dir, "mean_tpr.csv"), mean_tpr, delimiter=",")


def compute_macro_recall_for_cells(model, dataset, device, eps=1e-6):
    model.eval()
    recalls = []
    names = []
    with torch.no_grad():
        for img, lbl in dataset:
            names.append(dataset.image_names[len(names)])
            out = model(img.unsqueeze(0).to(device)).argmax(dim=1).squeeze(0).cpu().numpy()
            gtA = (lbl.numpy()==1).astype(np.uint8)
            prA = (out==1).astype(np.uint8)
            TP_A = np.logical_and(gtA==1, prA==1).sum()
            FN_A = np.logical_and(gtA==1, prA==0).sum()
            recA = TP_A/(TP_A+FN_A+eps)
            gtB = (lbl.numpy()==2).astype(np.uint8)
            prB = (out==2).astype(np.uint8)
            TP_B = np.logical_and(gtB==1, prB==1).sum()
            FN_B = np.logical_and(gtB==1, prB==0).sum()
            recB = TP_B/(TP_B+FN_B+eps)
            recalls.append((recA+recB)/2)
    for i, r in enumerate(recalls[:5]):
        print(f"Sample {i+1} Macro Recall = {r:.4f}")
    return names, recalls

def plot_macro_recall_bar_chart(names, recalls, save_path):
    x = np.arange(len(names))
    plt.figure(figsize=(8,6))
    plt.bar(x, recalls)
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel('Macro Recall')
    plt.title('Macro Recall per Sample')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def roc_per_sample_and_plot(model, dataset, device, train_csv_dir='auc_results', num_classes=3, save_path='final_result_labeled/roc_combined.pdf'):
    mean_fpr = np.loadtxt(os.path.join(train_csv_dir, 'mean_fpr.csv'), delimiter=',')
    mean_tpr = np.loadtxt(os.path.join(train_csv_dir, 'mean_tpr.csv'), delimiter=',')
    plt.figure(figsize=(10,7))
    plt.plot(mean_fpr, mean_tpr, 'k-', lw=3, label=f'Train Macro ROC (AUC={auc(mean_fpr, mean_tpr):.2f})')
    model.eval()
    with torch.no_grad():
        for idx in range(min(5, len(dataset))):
            img, lbl = dataset[idx]
            prob = F.softmax(model(img.unsqueeze(0).to(device)), dim=1).squeeze(0).cpu().numpy()
            prob_flat = prob.transpose(1,2,0).reshape(-1, num_classes)
            lbl_flat = lbl.numpy().reshape(-1)
            lbl_bin = label_binarize(lbl_flat, classes=list(range(num_classes)))
            fpr, tpr, _ = roc_curve(lbl_bin.ravel(), prob_flat.ravel())
            auc_val = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Test {idx+1} (AUC={auc_val:.2f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlim([0,1]); plt.ylim([0,1])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right'); plt.tight_layout()
    plt.savefig(save_path); plt.close()
