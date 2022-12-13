import torch
import csv
from torch.utils.data import random_split
from tokenized_smiles import FullData
from torch.utils.data import DataLoader
from feedforward import FeedForward
# Mol2VecFingerprint
# import seaborn as sns
# sns.set_theme()
# GET DATA
data = FullData(r'data\full_data_tokenized.csv', label='HL_Gap')
train, test = random_split(data, [60665, 26000])
train_loader = DataLoader(train, batch_size=16, shuffle=True)
test_loader = DataLoader(test, batch_size=16, shuffle=True)

model = FeedForward(input_dim=512, hidden_dim=256, hidden_dim_2=64, output_dim=1)

opt = torch.optim.Adam(model.parameters(), lr=0.00005)

loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

epochs = 50
train_loss_list = []
val_loss_list = []

for i in range(1, epochs+1):
    print("EPOCH {}".format(i))

    train_loss_run = 0

    for j, data in enumerate(train_loader):
        x, y = data
        x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        opt.zero_grad()

        y_hat = model(x)
        y_hat = torch.squeeze(y_hat)
        y = torch.squeeze(y)

        train_loss = loss(y_hat, y)
        train_loss.backward()

        train_loss_run += train_loss

        opt.step()

    print(j)
    train_loss_list.append(train_loss_run / j)
    print(train_loss_list)

    val_loss_run, val_acc_run = 0, 0

    y_hats = list()
    y_trues = list()

    # validate
    with torch.no_grad():

        model.eval()

        for k, data in enumerate(test_loader):

            x_val, y_val = data
            x_val, y_val = torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)

            y_hat_val = model(x_val)
            y_hat_val = torch.squeeze(y_hat_val)
            y_val = torch.squeeze(y_val)
            y_hats.append(y_hat_val.tolist())
            y_trues.append(y_val.tolist())

            val_loss = loss(y_hat_val, y_val)
            val_loss_run += val_loss
            val_loss_run += val_loss
            print("PREDICTED", y_hat_val)
            print("TRUE", y_val)

        print(val_loss_run / k)
        val_loss_list.append(val_loss_run / k)
        print(val_loss_list)
        with open(r'data\predicted_hl_epoch{}'.format(i), 'w') as f:
            write = csv.writer(f)
            write.writerows(y_hats)
        with open(r'data\true_hl_epoch{}'.format(i), 'w') as f:
            write = csv.writer(f)
            write.writerows(y_trues)

print(val_loss_list)
