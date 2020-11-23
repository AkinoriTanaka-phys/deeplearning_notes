# Python周辺の入門

## 深層学習に限らず有用なもの

- [Python入門](https://colab.research.google.com/github/AkinoriTanaka-phys/deeplearning_notes/blob/master/appendix/intro2python.ipynb)
>深層学習のフレームワークの多くはプログラミング言語pythonから使うのが普通です。そこで、まずはpythonの文法から説明します。
- [NumPy入門](https://colab.research.google.com/github/AkinoriTanaka-phys/deeplearning_notes/blob/master/appendix/intro2np.ipynb)
>実はpythonだけでは、計算速度が遅いため数値計算などには向いていません。数値計算したい場合はnumpyを使うのをオススメします。ちなみにcupyというのもあり、numpyの文法でほぼそのまま、GPUでの並列計算の恩恵を受けられてとても便利です。
- [Matplotlib入門](https://colab.research.google.com/github/AkinoriTanaka-phys/deeplearning_notes/blob/master/appendix/intro2plt.ipynb)
>pythonで図を描く場合はだいたいこれを使います。matplotlibのラッパーでseabornというのもありますが、ここでは説明しません。

## 深層学習ライブラリ

- [TensorFlow入門](https://colab.research.google.com/github/AkinoriTanaka-phys/deeplearning_notes/blob/master/appendix/intro2tf.ipynb)
>TensorFlow中のKerasを用いて、ニューラルネットワークの勾配学習を行うためのミニマムな説明。
- PyTorch入門
>近いうち書く予定…

---

**ノートの記法について**

文法について記述した箇所は、以下のように書くことにします。

> [**文法名(該当する公式ドキュメントの該当する箇所へのリンクつき)**](https://docs.python.org/ja/3/)
>
> <font color=dodgerblue>**文法名(リンクなし)**</font>
> ```python
> syntax
> ```
> <font color="gray">補足事項

最新版？のドキュメントへのリンクが貼ってあるので、詳細が知りたい場合は[**アンダーラインがついている部分**](https://docs.python.org/ja/3/tutorial/appetite.html)をクリックしてみてください。
 
**Google Colaboratoryを用いたプログラム実行について**
    
以下のような部分（セルと呼びます）に書いた/書かれたプログラムは実行可能です。
```
[ ]
```
    
セルを実行するには、まず実行したいセルを選択（クリック）します。その後で、

- セルを選択しつつ[shift]+[enter]とタイプするか、
- セルにカーソルを合わせると出現する再生マーク（$\triangleright$）をクリックする

上記のどちらかの操作で、対象のセルに書かれているプログラムが実行されます。あるセルで定義した値や関数は、他のセルでも使うことができます。

**Google Colaboratoryノートブックの実行について**

上のリンクから飛んだ先のノートブックは、Googleアカウントにログインした状態であれば自由に実行できますが、そのまま実行しようとすると、
    
> 警告: このノートブックは Google が作成したものではありません。
    
上記のような警告を出されると思います。警告を無視してそのまま実行しても問題ありませんが、警告が嫌な方は、ご自身のGoogle driveに、このノートブックをコピーしておけば良いと思います。以下のようにすればコピーできるはずです。

- 左上のメニューの[ファイル] $\to$ [ドライブにコピーを保存]

警告について「このまま実行」とした結果を保存しようとすると、ドライブにコピーするところまで誘導されるので、必ずしも最初にコピーしなければならないわけではありません。
    
---
