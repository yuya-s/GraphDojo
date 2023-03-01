# GAT

$\def\bm{\boldsymbol}$**GAT**は隣接する節点の中でも重要なものと重要ではないものがいるという考えのもと，どれが重要なのかも学習するGraph neural networkです．基本的な部分はGCNと大きな差はありません．

## GATの理論

GATでは隣接節点からの特徴量を集約する際に重要度$\alpha$(アテンション)を掛けます．

$\bm{H}_v^{(l+1)}=\sigma(\sum_{u \in \mathbb{N}_v\cup v}\alpha_{vu} \bm{H}_u \bm{W}^{(l)})$

$\alpha_{vu}$がどういう計算になるかというと，

$\alpha_{vu}=\frac{exp(LeakyReLU(\bm{a}(\bm{W}\bm{H}_v||\bm{W}\bm{H}_u)))}{exp(\sum_{m\in \mathbb{N}_v}LeakyReLU(\bm{a}(\bm{W}\bm{H}_v||\bm{W}\bm{H}_m)))}$

となります．

ポイントは$LeakyReLU(\bm{a}(\bm{W}\bm{H}_v||\bm{W}\bm{H}_u))$部分です．分子と分母がややこしいですが，ソフトマックス関数を用いて，節点$v$にとっての節点$u$の重要度を[0,1]にしているだけです．

$LeakyReLU(\bm{a}(\bm{W}\bm{H}_v||\bm{W}\bm{H}_u))$を解説すると，$\bm{a}$はベクトルをスカラーに変更する学習パラメータで，$\bm{H}_v$と$\bm{H}_u$を学習パラメータ$\bm{W}$により線形変換しています．これにより，**よしなに**節点の重要度を学習します．

さらに，GATでは複数のアテンションを用いるマルチヘッドアテンションが採用されています．

$\bm{H}_v^{(l+1)}=||_{k=1}^K\sigma(\sum_{u \in \mathbb{N}_v\cup v}\alpha_{vu} \bm{H}_u \bm{W}^{(l)})$

これにより，複数種類の重要度を学習することを狙っています．
一点注意点として，$K$個の特徴量を結合しているためl+1番目の層の特徴量のサイズが大きくなります．

## GATの実装

GAT層の実装例は以下となります．

```
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
```


```
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
```

## GATのポイント

周囲の節点の特徴量の重要度（アテンション）を学習することで，一様な集約ではない特徴量の集約を可能とする．

## GATの文献

論文：Graph Attention Networks 

著者：Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio 

会議：ICLR 2018 https://arxiv.org/abs/1609.02907 
