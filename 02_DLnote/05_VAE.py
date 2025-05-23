import torch
from torch import nn, optim
import torch.utils.data
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


## MNIST 데이터셋 로드 & 전처리
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))  # MNIST 평균, 표준편차
# ])

# train_dataset = datasets.MNIST(
#     root='data', train=True, download=True, transform=transform
# )
# test_dataset = datasets.MNIST(
#     root='data', train=False, download=True, transform=transform
# )

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform = transforms.ToTensor()),
    batch_size=128, shuffle=True, num_workers=4) #, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True,
                   transform= transforms.ToTensor()),
    batch_size=128, shuffle=False, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, z_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        """
        x: (batch_size, 784) 형태 (MNIST 이미지를 28x28 -> 784로 flatten)
        return: (mu, log_var)
        """
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, z_dim=20, hidden_dim=400, output_dim=784):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        """
        z: (batch_size, z_dim) 형태
        return: (batch_size, 784)
        """
        h = torch.relu(self.fc1(z))
        x_recon = torch.sigmoid(self.fc2(h))  # 픽셀값 0~1 범위로
        return x_recon

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, z_dim=20):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, z_dim)
        self.decoder = Decoder(z_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization Trick:
        z = mu + sigma * eps,
        where eps ~ N(0, I)
        """
        std = torch.exp(0.5 * logvar)   # logvar = log(sigma^2)
        eps = torch.randn_like(std)     # std와 같은 shape의 표준정규 샘플
        return mu + eps * std

    def forward(self, x):
        """
        x: (batch_size, 784)
        return: x_recon, mu, logvar
        """
        mu, logvar = self.encoder(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

def vae_loss_function(x_recon, x, mu, logvar):
    """
    x_recon: Decoder가 출력한 복원 이미지 (batch_size, 784)
    x: 실제 입력 이미지 (batch_size, 784)
    mu, logvar: Encoder에서 나온 z 분포 파라미터
    """
    # 1) Reconstruction Loss
    #   - x_recon은 0~1 범위의 확률처럼 해석 가능
    #   - x는 (0~1 범위로 정규화된) 실제 픽셀값
    recon_loss = F.binary_cross_entropy(
        x_recon, x.view(-1, 784), reduction='sum')

    # 2) KL Divergence: D_KL(q(z|x) || p(z))
    #    (정규분포 p(z)=N(0,I)로 가정)
    #    = 0.5 * sum(logvar.exp() + mu^2 - 1 - logvar)
    kl_div = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # 최종 loss = 재구성 오차 + KL
    return recon_loss + kl_div


def train(epoch, num_epochs):
    model.train()  # 학습 모드
    train_loss = 0.0
    for idx, (x_batch, _) in enumerate(train_loader):
        # 1) 데이터 준비
        x_batch = x_batch.to(device)  # (batch_size, 784)
        optimizer.zero_grad()
        # 2) 순전파 (Forward)
        x_recon, mu, logvar = model(x_batch)
        # 3) 손실 계산 (ELBO의 음수 방향)
        loss = vae_loss_function(x_recon, x_batch, mu, logvar)
        # 4) 역전파 및 최적화
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    # 배치 평균 또는 전체 샘플 수로 나눈 값으로 스케일 조정
    avg_loss = train_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}")

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for idx, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon, mu, log_var = model(data)
            test_loss += vae_loss_function(recon, data, mu, log_var).item()
            if idx == 0:
                n = min(data.size(0), 10)
                comparison = torch.cat([data[:n],
                                        recon.view(-1, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           './results/epoch_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    # 하이퍼파라미터 설정
    batch_size = 128
    learning_rate = 1e-3
    num_epochs = 10
    z_dim = 20  # 잠재변수 차원

    model = VAE(input_dim=784, hidden_dim=400, z_dim=z_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 생성을 위한 랜덤 잠재차원 샘플
    s = torch.randn(64, 20).to(device)

    for epoch in range(0, num_epochs):
        train(epoch,num_epochs)
        if epoch % 10 == 0:
            test(epoch)
            sample = model.decoder(s).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_{}_result.png'.format(epoch))
