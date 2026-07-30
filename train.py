import sys
import time
import os
import argparse
import gc
import numpy as np
import pandas as pd
from torch.autograd import Variable
from sklearn import metrics
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from data_generator import *
from EGNN_model import *
from final_model import *
from GraphTransformer_Block import *


def compute_class_weights(dataframe):
    """Compute class weights from a dataframe's label column.
    Uses sqrt(neg/pos) heuristic — softer than the raw ratio, prevents the model
    from over-swinging to predict-all-positive while still breaking mode collapse.
    Returns a float32 tensor of shape [2] for nn.CrossEntropyLoss(weight=...).
    """
    all_labels = np.concatenate([np.array(lbl) for lbl in dataframe['label'].values])
    pos = (all_labels == 1).sum()
    neg = (all_labels == 0).sum()
    # sqrt(neg/pos): e.g. neg/pos=5.52 → weight=2.35 instead of 5.52
    pos_weight = float(neg / pos) ** 0.5 if pos > 0 else 1.0
    pos_weight = min(pos_weight, 5.0)  # clamp
    weights = torch.tensor([1.0, pos_weight], dtype=torch.float32)
    print(f"Class weights: neg=1.00, pos={pos_weight:.2f}  (neg={neg}, pos={pos}, raw_ratio={neg/pos:.2f})")
    return weights

parser = argparse.ArgumentParser()
parser.add_argument('--fusion_mode', type=str, default='none', choices=['none', 'concat', 'gated', 'cross_attn'])
parser.add_argument('--d_proj', type=int, default=128)
parser.add_argument('--model_time', type=str, default=None)
parser.add_argument('--smoke_test', action='store_true')
args = parser.parse_args()

FUSION_MODE = args.fusion_mode
D_PROJ = args.d_proj
model_time = args.model_time

if args.smoke_test:
    NUMBER_EPOCHS = 1

Dataset_Path = "./Dataset/"
Model_Path = "./Model/"
Log_path = "./Log/"


# class EarlyStopping:
#     def __init__(self, patience=10, delta=0, path='checkpoint.pt'):
#         self.patience = patience
#         self.delta = delta
#         self.path = path
#         self.best_score = None
#         self.early_stop = False
#         self.counter = 0

#     def __call__(self, val_loss, model):
#         score = -val_loss

#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(model)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.save_checkpoint(model)
#             self.counter = 0

#     def save_checkpoint(self, model):
#         torch.save(model.state_dict(), self.path)





def train_one_epoch(model, data_loader):
    epoch_loss_train = 0.0
    n = 0
    for data in data_loader:
        model.optimizer.zero_grad()
        _, _, labels, node_features, G_batch, adj_matrix, xyz_feats, edges, edge_att, edge_feat, plm_features = data


        if torch.cuda.is_available():
            node_features = Variable(node_features.cuda().float())
            plm_features = Variable(plm_features.cuda().float())
            G_batch.edata['ex'] = Variable(G_batch.edata['ex'].float())
            G_batch = G_batch.to(torch.device('cuda:0'))
            adj_matrix = Variable(adj_matrix.cuda())
            xyz_feats = Variable(xyz_feats.cuda().float())
            edges = Variable(edges.cuda())
            edge_att = Variable(edge_att.cuda().float())
            edge_feat = Variable(edge_feat.cuda().float())
            y_true = Variable(labels.cuda())
        else:
            node_features = Variable(node_features.float())
            plm_features = Variable(plm_features.float())
            G_batch.edata['ex'] = Variable(G_batch.edata['ex'].float())
            adj_matrix = Variable(adj_matrix)
            xyz_feats = Variable(xyz_feats.float())
            edges = Variable(edges)
            edge_att = Variable(edge_att.float())
            edge_feat = Variable(edge_feat.float())
            y_true = Variable(labels)

        adj_matrix = torch.squeeze(adj_matrix)
        y_true = torch.squeeze(y_true)
        y_true = y_true.long()

        y_pred = model(node_features, xyz_feats, edges, edge_att, edge_feat, adj_matrix, plm_features=plm_features)

        # calculate loss
        loss = model.criterion(y_pred, y_true)

        # backward gradient
        loss.backward()

        # clip gradients to prevent explosions/NaNs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # update all parameters
        model.optimizer.step()

        epoch_loss_train += loss.item()
        n += 1

    epoch_loss_train_avg = epoch_loss_train / n
    return epoch_loss_train_avg


def evaluate(model, data_loader):
    model.eval()
    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []
    pred_dict = {}

    for data in data_loader:
        with torch.no_grad():
            sequence_names, _, labels, node_features, G_batch, adj_matrix, xyz_feats, edges, edge_att, edge_feat, plm_features = data

            if torch.cuda.is_available():
                node_features = Variable(node_features.cuda().float())
                plm_features = Variable(plm_features.cuda().float())
                adj_matrix = Variable(adj_matrix.cuda())
                G_batch.edata['ex'] = Variable(G_batch.edata['ex'].float())
                G_batch = G_batch.to(torch.device('cuda:0'))
                xyz_feats = Variable(xyz_feats.cuda().float())
                edges = Variable(edges.cuda())
                edge_att = Variable(edge_att.cuda().float())
                edge_feat = Variable(edge_feat.cuda().float())
                y_true = Variable(labels.cuda())

            else:
                node_features = Variable(node_features.float())
                plm_features = Variable(plm_features.float())
                adj_matrix = Variable(adj_matrix)
                xyz_feats = Variable(xyz_feats.float())
                edges = Variable(edges)
                edge_att = Variable(edge_att.float())
                edge_feat = Variable(edge_feat.float())
                y_true = Variable(labels)
                G_batch.edata['ex'] = Variable(G_batch.edata['ex'].float())

            adj_matrix = torch.squeeze(adj_matrix)
            y_true = torch.squeeze(y_true)
            y_true = y_true.long()

            y_pred = model(node_features, xyz_feats, edges, edge_att, edge_feat, adj_matrix, plm_features=plm_features)
            
            loss = model.criterion(y_pred, y_true)
            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred)
            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
            valid_pred += [pred[1] for pred in y_pred]
            valid_true += list(y_true)
            pred_dict[sequence_names[0]] = [pred[1] for pred in y_pred]

            epoch_loss += loss.item()
            n += 1
    epoch_loss_avg = epoch_loss / n

    return epoch_loss_avg, valid_true, valid_pred, pred_dict


def analysis(y_true, y_pred, best_threshold=None):
    if best_threshold is None:
        best_f1 = 0
        best_threshold = 0.5  # sensible default if no threshold found
        # Start from 0.01 — threshold=0.0 maps every softmax output to positive
        # (a degenerate solution that trivially maximises recall at the cost of precision)
        for threshold in range(1, 100):
            threshold = threshold / 100
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            f1 = metrics.f1_score(binary_true, binary_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]
    binary_true = y_true

    # binary evaluate
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    AUC = metrics.roc_auc_score(binary_true, y_pred)
    precisions, recalls, thresholds = metrics.precision_recall_curve(binary_true, y_pred)
    AUPRC = metrics.auc(recalls, precisions)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)

    results = {
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'AUC': AUC,
        'AUPRC': AUPRC,
        'mcc': mcc,
        'threshold': best_threshold
    }
    return results


class EarlyStopping:
    """Stop training when validation AUPRC has not improved for `patience` epochs."""
    def __init__(self, patience=10, path='checkpoint.pkl'):
        self.patience = patience
        self.path = path
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score, model):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            torch.save(model.state_dict(), self.path)
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping: {self.counter}/{self.patience} epochs without improvement.")
            if self.counter >= self.patience:
                self.early_stop = True


def train(model, train_dataframe, valid_dataframe, fold=0):
    train_loader = DataLoader(dataset=ProDataset(train_dataframe, fusion_mode=FUSION_MODE),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=8,
                              collate_fn=graph_collate)
    valid_loader = DataLoader(dataset=ProDataset(valid_dataframe, fusion_mode=FUSION_MODE),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=8,
                              collate_fn=graph_collate)

    best_epoch = 0
    best_val_auc = 0
    best_val_aupr = 0

    ckpt_path = os.path.join(Model_Path, f'Fold{fold}_best_model.pkl')
    early_stopping = EarlyStopping(patience=15, path=ckpt_path)

    for epoch in range(NUMBER_EPOCHS):
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        model.train()

        epoch_loss_train_avg = train_one_epoch(model, train_loader)
        print("========== Evaluate Train set ==========")
        _, train_true, train_pred, _ = evaluate(model, train_loader)
        # Use best-threshold search (not hardcoded 0.5) so metrics are meaningful
        # even when the model's confidence for positives is below 0.5.
        result_train = analysis(train_true, train_pred)
        print("Train loss: ", epoch_loss_train_avg)
        print("Train binary acc: ", result_train['binary_acc'])
        print("Train AUC: ", result_train['AUC'])
        print("Train AUPRC: ", result_train['AUPRC'])
        print("Train F1: ", result_train['f1'])
        print("Train best threshold: ", result_train['threshold'])
        print("Current LR: ", model.optimizer.param_groups[0]['lr'])

        print("========== Evaluate Valid set ==========")
        epoch_loss_valid_avg, valid_true, valid_pred, _ = evaluate(model, valid_loader)
        # Also use best-threshold for validation — this is the honest metric
        result_valid = analysis(valid_true, valid_pred)
        print("Valid loss: ", epoch_loss_valid_avg)
        print("Valid binary acc: ", result_valid['binary_acc'])
        print("Valid precision: ", result_valid['precision'])
        print("Valid recall: ", result_valid['recall'])
        print("Valid f1: ", result_valid['f1'])
        print("Valid AUC: ", result_valid['AUC'])
        print("Valid AUPRC: ", result_valid['AUPRC'])
        print("Valid mcc: ", result_valid['mcc'])
        print("Valid best threshold: ", result_valid['threshold'])

        # Track best by AUPRC (area under precision-recall curve)
        if best_val_aupr < result_valid['AUPRC']:
            best_epoch = epoch + 1
            best_val_auc = result_valid['AUC']
            best_val_aupr = result_valid['AUPRC']

        # Early stopping monitors validation AUPRC and saves best checkpoint
        early_stopping(result_valid['AUPRC'], model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

        # Step CosineAnnealingLR each epoch
        model.scheduler.step()

    return best_epoch, best_val_auc, best_val_aupr


def cross_validation(all_dataframe, fold_number=5):
    print("Random seed:", SEED)
    print("The base Model type:", BASE_MODEL_TYPE)
    print("Add node features:", ADD_NODEFEATS)
    print("Map cutoff:", MAP_CUTOFF)
    print("Use edge features or not while using GAT model:", USE_EFEATS)
    print("The parameter of normalizing the distance:", DIST_NORM)
    print("Feature dim:", INPUT_DIM)
    print("Hidden dim:", HIDDEN_DIM)
    print("Layer:", LAYER)
    print("Dropout:", DROPOUT)
    print("Alpha:", ALPHA)
    print("Lambda:", LAMBDA)
    print("Learning rate:", LEARNING_RATE)
    print("Training epochs:", NUMBER_EPOCHS)
    print()

    # 取出dataframe中的值
    sequence_names = all_dataframe['ID'].values
    sequence_labels = all_dataframe['label'].values
    
    if args.smoke_test:
        print("\n\n========== Smoke Test Fold 1 ==========")
        train_dataframe = all_dataframe.iloc[:2]
        valid_dataframe = all_dataframe.iloc[2:4]
        class_weights = compute_class_weights(train_dataframe)
        model = FinalModel(INPUT_DIM, HIDDEN_DIM, FLITER_DIM, OUTPUT_SIZE, DROPOUT, LAYER,
                           fusion_mode=FUSION_MODE, d_proj=D_PROJ, class_weights=class_weights)
        if torch.cuda.is_available():
            model.cuda()
        best_epoch, valid_auc, valid_aupr = train(model, train_dataframe, valid_dataframe, fold=1)
        return 1

    kfold = KFold(n_splits=fold_number, shuffle=True)
    fold = 0
    best_epochs = []
    valid_aucs = []
    valid_auprs = []

    for train_index, valid_index in kfold.split(sequence_names, sequence_labels):
        print("\n\n========== Fold " + str(fold + 1) + " ==========")
        train_dataframe = all_dataframe.iloc[train_index, :]
        valid_dataframe = all_dataframe.iloc[valid_index, :]
        print("Train on", str(train_dataframe.shape[0]), "samples, validate on",
              str(valid_dataframe.shape[0]), "samples")

        # Compute class weights from THIS fold's training set only (no data leakage)
        class_weights = compute_class_weights(train_dataframe)

        model = FinalModel(INPUT_DIM, HIDDEN_DIM, FLITER_DIM, OUTPUT_SIZE, DROPOUT, LAYER,
                           fusion_mode=FUSION_MODE, d_proj=D_PROJ, class_weights=class_weights)

        if torch.cuda.is_available():
            model.cuda()

        best_epoch, valid_auc, valid_aupr = train(model, train_dataframe, valid_dataframe, fold + 1)
        best_epochs.append(str(best_epoch))
        valid_aucs.append(valid_auc)
        valid_auprs.append(valid_aupr)
        fold += 1

        # Clean up memory to avoid fragmentation
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n\nBest epoch: " + " ".join(best_epochs))
    print("Average AUC of {} fold: {:.4f}".format(fold_number, sum(valid_aucs) / fold_number))
    print("Average AUPR of {} fold: {:.4f}".format(fold_number, sum(valid_auprs) / fold_number))
    return round(sum([int(epoch) for epoch in best_epochs]) / fold_number)


def train_full_model(all_dataframe, aver_epoch):
    if args.smoke_test:
        all_dataframe = all_dataframe.iloc[:2]
    print("\n\nTraining a full model using all training data...\n")

    # Compute class weights from the full training set
    class_weights = compute_class_weights(all_dataframe)
    model = FinalModel(INPUT_DIM, HIDDEN_DIM, FLITER_DIM, OUTPUT_SIZE, DROPOUT, LAYER,
                       fusion_mode=FUSION_MODE, d_proj=D_PROJ, class_weights=class_weights)
    if torch.cuda.is_available():
        model.cuda()

    train_loader = DataLoader(dataset=ProDataset(all_dataframe, fusion_mode=FUSION_MODE),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=8,
                              collate_fn=graph_collate)

    for epoch in range(NUMBER_EPOCHS):
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        model.train()

        epoch_loss_train_avg = train_one_epoch(model, train_loader)
        print("========== Evaluate Train set ==========")
        _, train_true, train_pred, _ = evaluate(model, train_loader)
        result_train = analysis(train_true, train_pred)
        print("Train loss: ", epoch_loss_train_avg)
        print("Train binary acc: ", result_train['binary_acc'])
        print("Train AUC: ", result_train['AUC'])
        print("Train AUPRC: ", result_train['AUPRC'])
        print("Train F1: ", result_train['f1'])

        model.scheduler.step()

        if args.smoke_test:
            torch.save(model.state_dict(), os.path.join(Model_Path, 'Full_model_1.pkl'))
            break

        if epoch + 1 in [aver_epoch, 45]:
            torch.save(model.state_dict(), os.path.join(Model_Path, 'Full_model_{}.pkl'.format(epoch + 1)))  # save model


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, 'ab', buffering=0)

    def write(self, message):
        self.terminal.write(message)
        try:
            self.log.write(message.encode('utf-8'))
        except ValueError:
            pass

    def close(self):
        self.log.close()
        sys.stdout = self.terminal

    def flush(self):
        pass


def main():
    if not os.path.exists(Log_path): os.makedirs(Log_path)

    with open(Dataset_Path + "Train_335.pkl", "rb") as f:
        Train_335 = pickle.load(f)
        Train_335.pop('2j3rA')  # remove the protein with error sequence in the train dataset
    IDs, sequences, labels = [], [], []

    for ID in Train_335:
        IDs.append(ID)
        item = Train_335[ID]
        sequences.append(item[0])
        labels.append(item[1])

    train_dic = {"ID": IDs, "sequence": sequences, "label": labels}
    train_dataframe = pd.DataFrame(train_dic)
    aver_epoch = cross_validation(train_dataframe, fold_number=5)
    train_full_model(train_dataframe, aver_epoch)

if __name__ == "__main__":

    if model_time is not None:
        checkpoint_path = os.path.normpath(Log_path +"/"+ model_time)
    else:
        localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        checkpoint_path = os.path.normpath(Log_path + f"/fusion_{FUSION_MODE}_d{D_PROJ}_" + localtime)
        os.makedirs(checkpoint_path)
    Model_Path = os.path.normpath(checkpoint_path + '/model')
    if not os.path.exists(Model_Path): os.makedirs(Model_Path)

    sys.stdout = Logger(os.path.normpath(checkpoint_path + '/training.log'))
    main()
    sys.stdout.log.close()
