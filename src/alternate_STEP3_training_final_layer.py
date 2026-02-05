import torch
import pandas as pd
import os
import matplotlib.pyplot as plt

regularization_type = 'L1'  # or 'L2'
lambda_reg = 0 # regularization strength

class EmbedingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings_path, labels_path,split ='train'):
        if split == 'train':
            df0 = pd.read_excel(labels_path, sheet_name="PB").dropna(subset=["Patient"])
            self.dict_labels = {int(df0.at[i, "Patient"]): int(df0.at[i,"Récidive avant 2 ans"]) for i in range(len(df0)) if int(df0.at[i, "Patient"])<50}
        elif split == 'val':
            df0 = pd.read_excel(labels_path, sheet_name="PB").dropna(subset=["Patient"])
            self.dict_labels = {int(df0.at[i, "Patient"]): int(df0.at[i,"Récidive avant 2 ans"]) for i in range(len(df0)) if int(df0.at[i, "Patient"])>=50}
        elif split == 'test':
            df1 = pd.read_excel(labels_path, sheet_name="HMN").dropna(subset=["Patient"])
            df2 = pd.read_excel(labels_path, sheet_name="BJN").dropna(subset=["Patient"])
            self.dict_labels = {int(df1.at[i, "Patient"]): int(df1.at[i,"Récidive avant 2 ans"]) for i in range(len(df1))}
            self.dict_labels.update({int(df2.at[i, "Patient"]): int(df2.at[i,"Récidive avant 2 ans"]) for i in range(len(df2))})
        self.pt_files = {f.split('_')[0] : os.path.join(embeddings_path, f) for f in os.listdir(embeddings_path) if f.endswith('_features.pt')}
        # filter pt_files to keep only those in dict_labels
        self.pt_files = {k: v for k, v in self.pt_files.items() if int(k[:-1]) in self.dict_labels}
        self.slide_ids = list(self.pt_files.keys())
        
    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]
        embedding = torch.load(self.pt_files[slide_id])['last_layer_embed']
        label = self.dict_labels[int(slide_id[:-1])]
        target = torch.tensor(label, dtype=torch.float32)
        return embedding.to('cuda'), target.to('cuda')

last_layer = torch.nn.Sequential(torch.nn.Dropout(0.2),torch.nn.Linear(in_features=768, out_features=1)).to('cuda')

trainDataset = EmbedingDataset(embeddings_path='/media/eve/My Passport/data_hcc/features', labels_path="/home/eve/Desktop/papier_aymen/sample_dataset/Label_slides.xlsx", split='train')
traindataloader = torch.utils.data.DataLoader(trainDataset, batch_size=12, shuffle=True)

valDataset = EmbedingDataset(embeddings_path='/media/eve/My Passport/data_hcc/features', labels_path="/home/eve/Desktop/papier_aymen/sample_dataset/Label_slides.xlsx", split='val')
valdataloader = torch.utils.data.DataLoader(valDataset, batch_size=2, shuffle=False)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(last_layer.parameters(), lr=1e-5)


l_losses = []
v_losses = []
min_cumulative_loss = float('inf')
# Training loop for the final layer
for epoch in range(200):  # number of epochs
    print(f"Epoch {epoch+1}")
    cumulative_loss = 0.0
    
    # train
    last_layer.train()
    for embeds,labels in traindataloader:
        # forward pass
        outputs = last_layer(embeds)
        loss = loss_fn(outputs.squeeze(), labels)
        optimizer.zero_grad()
        cumulative_loss += loss.item()
        # Apply L1 regularization
        if regularization_type == 'L1':
            l1_norm = sum(p.abs().sum() for p in last_layer.parameters())
            loss += lambda_reg * l1_norm
        # backward pass and optimization
        loss.backward()
        optimizer.step()
    l_losses.append(cumulative_loss / len(traindataloader))

    print(f"Loss: {cumulative_loss / len(traindataloader)}")
    
    # validation
    last_layer.eval()
    cumulative_loss = 0.0
    with torch.no_grad():
        for embeds,labels in valdataloader:
            outputs = last_layer(embeds)
            loss = loss_fn(outputs.squeeze(), labels.squeeze())
            cumulative_loss += loss.item()
        if cumulative_loss < min_cumulative_loss:
            min_cumulative_loss = cumulative_loss
            os.makedirs('data/models', exist_ok=True)
            torch.save(last_layer.state_dict(), f'data/models/last_layer_best.pth')
            print("Saved best model with loss:", min_cumulative_loss/len(valdataloader))
    print(f"Validation Loss: {cumulative_loss / len(valdataloader)}")
    v_losses.append(cumulative_loss / len(valdataloader))

plt.plot(l_losses, label='Train Loss')
plt.plot(v_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()