# 1. 機械学習の目的と確率論の復習
近年の深層学習(Deep Learning, DL)の発展には驚かされます. 深層学習は GPU(Graphics Processing Unit) による並列化処理やオンラインデータの充実、効率的な最適化手法やアルゴリズムの提案など様々な要因が複雑に絡み合っており, 一言で「こういうものだ」と説明するのは難しいのですが、誤解を恐れず、あえて簡単化して言えば, **深層ニューラルネットワークをもちいた機械学習の方法** ということです。

### 基礎研究について
現在、世界中で機械学習の研究や開発が行われています。研究論文は雑誌に論文を投稿し出版されるケースもありますが、主に国際会議に投稿され出版されるケースのほうが多い気がします。人気のある国際会議の参加チケットはレジストレーション開始から僅か数分で売り切れてしまうといいます。機械学習の研究では、

- [Conference on Neural Information Processing Systems (NIPS または NeurIPS)](https://nips.cc)
- [International Conference on Machine Learning (ICML)](https://icml.cc)
- [International Conference on Learning Representations (ICLR)](https://iclr.cc)

などが特に有名です。ICLR は特に深層学習に特化した会議で、深層学習の良い論文が多い印象を受けます。また、より広い人工知能の分野での会議には

- [Conference on AI sponsored by the Association for the Advancement of Artificial Intelligence (AAAI)](https://www.aaai.org/Conferences/AAAI/aaai.php)
- [International Conference on Artificial Intelligence and Statistics (AISTATS)](https://www.aistats.org)
- [International Joint Conferences on Artificial Intelligence Organization (IJCAI)](https://www.ijcai.org)

などが挙げられます。この他にも画像関連に特化したCVPRやロボティクス等に特化した ICRAなどより専門分野を絞った国際会議も多くあります。どの会議でも発表したい人は、まず論文を投稿します。論文はレフェリーからの査読を受け、採択されれば発表させてもらえ ます。これらの会議は非常に人気が高く、投稿数も多い(オーダー $10^3$ 件程らしいです)ので採択率も低く、およそ20%程だと思われます。また、レビューは公正を期すためダブルブラインド方式、すなわち著者もレフェリーも共に匿名で行われます。
> ただし、プライオリティ保護のためにプレプリント・サーバー (arXiv) への投稿は認められているように思います。従って厳密な意味でのダブルブラインドではないです。

レビューの結果は採択/不採択にかかわらず
- https://openreview.net

にて開示されることが多いようです。採択された論文の著者の所属を眺めていると、大学や研究所の研究者も沢山いらっしゃいますが、同等かそれ以上に企業所属の研究者が多いように感じます。研究者を志しても、必ずしも大学に残らなくても良いところが他の分野と大きく違う、良いところだと思います。また、論文の多くがソースコードごと公開されるのが 普通です。公開にはgithubがよく使われます。文献目録は[dblp](https://dblp.uni-trier.de/)というのが網羅的で良いです。

### コンペティション
学術研究の一方で、機械学習の分野では様々なコンペティション (性能比べ大会?)が開催されています。有名なのは

- https://www.kaggle.com/competitions

です。ここにアカウントを登録すると機械学習コンペに参加することができます。他の有名 なコンペに、ImageNet と呼ばれる巨大なラベル付き画像データアーカイブを用いた画像認 識のコンペ(Large Scale Visual Recognition Challenge, ILSVRC)が 2010 年から 2017 年 まで開催されていました。開催当初は誤認率 26%程度で数%を争っていたところに現れたの が Hinton 率いるトロント大の深層学習アーキテクチャ、AlexNet([論文リンク](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networ))、です。これは 2012 年に誤認率 16% で他を圧倒し研究者を驚かせました。この「事件」が深層学習の方法の流行 の始まりとされています。


### 計算環境
なぜかはよくわかりませんが、フロントエンドのプログラミング言語には python が用いられることが多いです。python の深層学習ライブラリには

ライブラリ名|[TensorFlow](https://www.tensorflow.org/)|[PyTorch](https://pytorch.org/)|[MXNet](https://mxnet.apache.org)|[CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/)|...
---|---|---|---|---|---
サポート|Google|Facebook|AWS|Microsoft|...

などがあります。これらをより容易に使うためのインターフェースとして F. Chollet によるKerasというライブラリもあり、これも良いです(tensorflowに標準装備です)。また、よく使うサブルーチンや簡単な機械学習アルゴリズムをまとめたライブラリに [scikit-learn](https://scikit-learn.org/stable/) があります。これらの計算環境を自前で整えるには、それなりに面倒な手続きが必要であり、それが参入障壁になっているような気もします。しかし最近ではGoogle Colaboratoryという、Google のサーバー上で既に整えられた python 環境を利用するサービスが無料で提供されており、誰でも気軽に試せるようになっています。この講義ノートは基本的に全て Google Colaboratory 上で開けるようにしました。驚くことに GPU や 深層学習特化型ユニットの TPU(Tensor Processing Unit) までも無料で使えてしまいます。 このサービスでは連続 12 時間までの計算を実行してくれますが、それ以上長い時間がかかるような処理をさせたい場合は自前の計算環境を作る必要があります。

### 機械学習で現在できること
深層学習の方法は
- 画像に関する処理(主に畳み込みニューラルネットによる)
- 時系列データに対する処理(主にLSTMやGRU等の再帰的ニューラルネットによる) 
- その組み合わせ処理

において様々なタスクで高い汎用性を示しています。具体的にどのようなことができるのか 気になる方は[TensorFlowのチュートリアル](https://www.tensorflow.org/tutorials?hl=ja)や[PyTorchのチュートリアル](https://pytorch.org/tutorials/)を見てみるのがいいでしょう。