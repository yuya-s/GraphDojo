# SGC

**SGC**はGCNの処理を簡易にすることで学習の効率化を目指したGNNです．

## SGCの理論

GCNでは隣接する節点の集約という処理を逐一実行しなければいけませんでした．逐一処理しなければいけない原因は活性化関数の存在です．活性化関数があるせいで集約の事前計算ができませんでした．

SGCでは活性化関数を全て取り除き，事前に特徴量を集約し，線形層での学習を行うという処理を行います．

SGCは下記の数式で表すことができます．

$\bm{H}=\bm{A}^k\bm{X}\bm{W}$

$k$はGCNにおける畳込み回数に相当します．


非常に簡単な式ですが，学習も高速，精度もGCNとそれほど変わらないという結果が報告されています．

## SGCの実装

準備中です．

## SGCの文献

論文：Simplifying Graph Convolutional Networks 

著者：Felix Wu, Tianyi Zhang, Amauri Holanda de Souza Jr., Christopher Fifty, Tao Yu, Kilian Q. Weinberger 

会議：ICML 2019 https://arxiv.org/abs/1902.07153