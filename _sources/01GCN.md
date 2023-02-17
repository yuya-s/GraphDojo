# GCN

**GCN**は最も基礎的かつ直観的に理解しやすいGraph Neural Networkです．非常に端的に言うと，自身の特徴量と枝で繋がってる節点の特徴量を平均化して，自身の特徴量とするだけです．

## GCNの理論

$H^{l+1} = \sigma((I+D^{-\frac{1}{2}}AD^{-\frac{1}{2})}H^{l}W) $

あるひとつの節点$v$の特徴量を考えると，

$h_v^{l+1}=\sigma(h_v^l+\sum_{u \in N_v}\frac{1}{\sqrt{d_vd_u}}h_u W_v)$

## GCNの実装