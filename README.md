# integrated_I2A
an I2A agent with integrated envModel

これは私が修論に使うコードを保存するためのリポジトリです。

## 参考にしたリポジトリ
- [FlorianKlemt/gym-minipacman](https://github.com/FlorianKlemt/gym-minipacman)
    - これはnips2017のワークショップのコードをもとにしたminipacmanの実装になります
    - ワークショップ版よりの改良
        - gymの形式に書き換えた
        - もとのPillEaterではseedを固定できなかった、こっちはできるようになった
    - 問題点
        - Florianの実装はgym 0.10.xに基づいているので、`ImportError`が発生する。解決法としては、そのリポジトリの[Pull Request](https://github.com/FlorianKlemt/gym-minipacman/pull/2/commits/33bdffb1cec53bbab18f5384a1a7a55c59ec5cc1)通りにやれば解決できる。
        - `env.reset()`の`return`は`(0,1)`か？`(0,255)`か？
- [higgsfield/Imagination-Augmented-Agents](https://github.com/higgsfield/Imagination-Augmented-Agents)
    - これは上で言ったnips2017ワークショップの人が実装したpytorchによるI2Aの実装
    - このリポジトリ内のコードはいくつかの問題が存在している
        - I2Aの学習中、`distil_optimizer.step()`ではなく`optimizer.step()`を二回実行していた
        - 元論文ではすべてのネットワークは`LeakyReLU()`を使っていると述べたが、この実装では`ReLU()`を使っていた
- [higgsfield/RL-Adventure](https://github.com/higgsfield/RL-Adventure)
    - DQN系列の強化学習手法の実装
    - [Policy network](https://github.com/higgsfield/RL-Adventure-2)の方もある

## 参考にした記事
- [Reproducibility issues using OpenAI Gym](https://harald.co/2019/07/30/reproducibility-issues-using-openai-gym/)
    - これはOpenAI gymを使用する時のseed固定がうまく行かなかったときの解決法についての記事
    - TLDR: シードを決めるときは`env.action_space.seed(RANDOM_SEED)`もする
        - `env.action_space.sample()`のランダムシードは`env.seed(RANDOM_SEED)`に管理されてないのが問題
- [PyTorchは誤差逆伝播とパラメータ更新をどうやって行っているのか？](https://ohke.hateblo.jp/entry/2019/12/07/230000)
    - `optimizer.zero_grad()`->`loss.backward()`->`optimizer.step()`がどういう意味かについて解説
- 

## TODO
- modelの可視化
- tensorboardのparameterグリッド
- integrated-i2aを実装
- integrated-i2aのA2C実験
- integrated-i2aのDQN実験
- t-SNEでFCの最後の出力のウェイトを可視化
