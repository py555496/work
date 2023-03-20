from sklearn.model_selection import train_test_split


feature_arr = []
len_set = set()
def get_data(file_name):
    x = []
    y = []
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip()
            line_arr = line.split(" ")
            label = line_arr[0] 
            feature = line_arr[1:]
            y.append(int(label))
            feature = [float(x.split(":")[1]) for x in feature]
            x.append(feature)
    return x, y 
x_train, y_train = get_data("./sample_data/sample_train")
x_test, y_test = get_data("./sample_data/sample_test")
for x in x_train:
    len_set.add(len(x))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# 定义DeepFM模型
class DeepFM_v2(nn.Module):
    def __init__(self, feature_dim, embedding_dim, hidden_dim):
        super(DeepFM, self).__init__()
        
        # FM部分
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(feature_dim, embedding_dim)
        self.linear = nn.Linear(embedding_dim, 1)
        
        # DNN部分
        self.fc1 = nn.Linear(feature_dim * embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

        # total dense         
        self.final_fc = nn.Linear(3, 1)
        
    def forward(self, x):
        # FM部分
        emb = self.embedding(x)
        linear_part = self.linear(torch.sum(emb, dim=1))
        interaction_part = 0.5 * torch.sum(torch.pow(torch.sum(emb, dim=1), 2) - torch.sum(torch.pow(emb, 2), dim=1), dim=1)
        
        # DNN部分
        dnn_input = emb.view(-1, x.shape[1] * self.embedding_dim)
        dnn_output = self.relu(self.fc1(dnn_input))
        dnn_output = self.fc2(dnn_output)
        
        # 合并FM和DNN的输出
        #print(linear_part, interaction_part.reshape(-1,1), dnn_output.squeeze().reshape(-1,1))
        #output = linear_part + interaction_part + dnn_output.squeeze()
        #return self.final_fc(torch.cat([linear_output, fm_output, deep_output], dim=1))
        return self.final_fc(torch.cat((linear_part, interaction_part.reshape(-1,1), dnn_output.squeeze().reshape(-1,1)), dim=1))
        
        #return output


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.long), torch.tensor(float(self.y[idx]), dtype=torch.float)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = DeepFM(feature_size=137, embedding_size=10, hidden_size=64).to(device)
#model = DeepFM_v2(len(x_train[0]), embedding_dim=32, hidden_dim=32).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_dataset = MyDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch))