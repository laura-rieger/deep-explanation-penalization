from torch.utils.data import TensorDataset, ConcatDataset
import numpy as np
from os.path import join as oj
import torch.utils.data as utils
from torch.utils.data import DataLoader
from sklearn.metrics import auc,average_precision_score, roc_curve,roc_auc_score,precision_recall_curve, f1_score
from torch.utils.data import TensorDataset, ConcatDataset
import numpy as np
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize
import torch
from os.path import join as oj
def load_precalculated_dataset(path):
    with open(oj(path, "cancer.npy"), 'rb') as f:
        cancer_features = np.load(f)
    with open(oj(path, "not_cancer.npy"), 'rb') as f:
        not_cancer_features = np.load(f)
    with open(oj(path, "not_cancer_cd.npy"), 'rb') as f:
        not_cancer_cd= np.load(f)   
    cancer_targets = np.ones((cancer_features.shape[0])).astype(np.int64)
    not_cancer_targets = np.zeros((not_cancer_features.shape[0])).astype(np.int64)
    not_cancer_dataset = TensorDataset(torch.from_numpy(not_cancer_features).float(), torch.from_numpy(not_cancer_targets),torch.from_numpy(not_cancer_cd).float())
    cancer_dataset = TensorDataset(torch.from_numpy(cancer_features).float(), torch.from_numpy(cancer_targets),torch.from_numpy(-np.ones((len(cancer_features), 2, 25088))).float())
    complete_dataset = ConcatDataset((cancer_dataset, not_cancer_dataset))
    num_total = len(complete_dataset)
    num_train = int(0.8 * num_total)
    num_val = int(0.1 * num_total)
    num_test = num_total - num_train - num_val
    torch.manual_seed(0); #reproducible splitting
    train_dataset, test_dataset, val_dataset= torch.utils.data.random_split(complete_dataset, [num_train, num_test, num_val])
    train_filtered_dataset =torch.utils.data.Subset(complete_dataset, [idx for idx in train_dataset.indices if complete_dataset[idx][2][0,0] ==-1])
    test_filtered_dataset = torch.utils.data.Subset(complete_dataset, [idx for idx in test_dataset.indices if complete_dataset[idx][2][0,0] ==-1])
    val_filtered_dataset = torch.utils.data.Subset(complete_dataset, [idx for idx in test_dataset.indices if complete_dataset[idx][2][0,0] ==-1])
    datasets = {'train': train_dataset,'train_no_patches': train_filtered_dataset, 'val':val_dataset ,'val_no_patches':val_filtered_dataset ,'test':test_dataset, 'test_no_patches':test_filtered_dataset }
    return datasets
def get_output(model, dataset):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16,
                                             shuffle=False, num_workers=4)
    model = model.eval()
    y = []
    y_hat = []
    softmax= torch.nn.Softmax()
    with torch.no_grad() :
        for inputs, labels, cd in data_loader:
            y_hat.append((labels).cpu().numpy())
            y.append(torch.nn.Softmax(dim=1)( model(inputs.cuda()))[:,0].detach().cpu().numpy())
    y_hat = np.concatenate( y_hat, axis=0 )
    y = np.concatenate( y, axis=0 )
    return 1-y, y_hat # in the training set the values were switched
def get_auc_f1(model, fname, dataset):
    with open(fname, 'rb') as f:
        weights = torch.load(f)
    if "classifier.0.weight" in weights.keys(): #for the gradient models we unfortunately saved all of the weights
        model.load_state_dict(weights)
    else:
        model.classifier.load_state_dict(weights)
    y, y_hat = get_output(model.classifier, dataset)
    auc =roc_auc_score(y_hat, y)
    f1 = np.asarray([f1_score(y_hat, y > x) for x in np.linspace(0.1,1, num = 10) if (y >x).any() and (y<x).any()]).max()
    return auc, f1
def load_img_dataset(path):
    img_path_nocancer = oj(path, "not_cancer")
    img_path_cancer = oj(path, "cancer")
    seg_path  = oj(path, "../segmentation")
    mean = np.asarray([0.485, 0.456, 0.406])
    std = np.asarray([0.229, 0.224, 0.225])
    complete_dataset = torchvision.datasets.ImageFolder(path, transform=Compose([ToTensor(), Normalize(mean, std)]))
    num_total = len(complete_dataset)
    num_train = int(0.8 * num_total)
    num_val = int(0.1 * num_total)
    num_test = num_total - num_train - num_val
    torch.manual_seed(0);
    train_dataset, test_dataset, val_dataset= torch.utils.data.random_split(complete_dataset, [num_train, num_test, num_val])
    return test_dataset  