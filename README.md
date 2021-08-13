# live-competition-2021
対話システムライブコンペティションのリポジトリ

## 実装のメモ

### Reference Here

[Reference](https://python-telegram-bot.readthedocs.io/en/latest/index.html)

[Transition-Guide](https://github.com/python-telegram-bot/python-telegram-bot/wiki/Transition-guide-to-Version-12.0)

### .telegramについて

~~~
TOKEN=------------
~~~

この形式で書いたものをTOKENとして読み込むことにする．

それ用の関数も作る．

実際にデプロイする際には，os.environ.get()の形式で読み込むようにする．多分Heroku?

### システム構成

System: 清水(男性)

User: 湯川(男性)

## 概要

シチュエーショントラックのシステム仕様・評価について
シチュエーショントラックでは，オープンな雑談とは異なり，設定された状況の中で状況にあった人らしい対話を行う能力を競います．
システム発話はテキスト，絵文字，顔文字を利用可能とします．STICKERS(LINEでいうスタンプ)は不可とします．
シチュエーションはライブコンペ3のものと異なります．
今回用いるシチュエーションについては シチュエーション を参照ください．
参加者はシチュエーションに適した対話を行うシステムを作成します．
予選では，上記の評価基準を用い，クラウドワーカーによる評価を実施します．
本選では，オープントラックと同様，参加者全員が評価します．
システム仕様・評価基準は本ページに則るものとします。(オープントラックは こちらのページ をご覧ください)
満たすべきシステムの仕様
評価の観点から，ボットは以下の仕様を満たすように作成してください．

後述の シチュエーション で行われる会話（テキストチャット）であること．
評価の都合上，システムからの発話は１ターンあたり１発話とし，発話内には改行を含めないようにしてください．
対話以外の要素に評価が左右されることを防ぐために，Telegram上のアイコン画像や発話外でのプロフィール提示は使用しないでください．
発話として入力できるのはテキストに加え，Telegramで利用可能な絵文字・顔文字 です．画像は不可とします．STICKERS(LINEでいうスタンプ)の利用も不可とします．

16発話以上システム発話が継続するようにしてください．
　 ※オーガナイザの判断により，継続すべき発話数は変更となる場合があります．
ユーザからの 15発話目を受け取り，システムが 16発話目を発話した後に，
　 "_FINISHED_:"に続けてユニークIDを発話として出力してください．ユニークIDは "unixtime:ユーザID:ボットのusername"から生成する文字列とします．
　 （例：_FINISHED_:1536932911:654708492:LiveCompetition2018_bot）
　 ※"unixtime:ユーザID"は IDをユニークにするために，またボットのusernameは対話システムを識別するために用います．
"_FINISHED_:[unixtime]:[ユーザID]:[ボットのusername]"を出力後，アノテータへの指示として以下の発話を出力してください．
　 "対話終了です．エクスポートした「messages.html」ファイルを，フォームからアップロードしてください．"
シチュエーション
背景
システム側

清水（システム）は，次の週末に，会社の仲の良い同期とオンライン飲み会を企画している． 今のところ，声がけしたメンバーの佐藤，鈴木，高橋，渡辺，小林は，全員参加の予定だ． コロナ禍の昨今，居酒屋で酒を飲むことはできないため，以前に上司の湯川（ユーザ）がオンライン飲み会を企画したことがあり，これが結構盛り上がった． 前回とは違って今回は，オンライン飲み会のために酒やおつまみを，当日に各自の自宅に届けてくれるサービスも使うので，更に楽しくなりそうだ． ところで，同期のメンバーらが，上司である湯川も誘ってはどうかと言い出した． 確かに湯川は，年も近いし，話が合うこともあって，私たちとプライベートでも仲が良い． 会社では上司であるが，同時に公私にわたって良き先輩であるともいえる． 週末に特に予定が無いことを「暇だ」とぼやいていたし，もともと飲み会が好きな人なので，誘えばきっと参加してくれるだろう． オンラインサービスの予約期限が迫っており，メンバーを確定しなくてはならないので，早速連絡しよう．
ユーザ側

部下の清水（システム）から，オンライン飲み会の誘いがきた． 私（湯川＝ユーザ）は飲み会は好きだし，以前に自分が企画したオンライン飲み会は結構楽しかったので，参加したい気持ちは正直ある． しかし，清水の同期ということは，小林も参加するということだ． ここだけの話だが，実は，小林のことがあまり好きではない．というより，はっきり言って嫌いだ． 前回は部署全体の飲み会だったので仕方がなかったが，今回は内輪での集まりのようだし，無理してまで参加する必要もない． 小林がいないのであれば参加したいのだけれども，それが我儘なのもよくわかっている．我慢して参加しても楽しくないので，今回は断ろう．

インストラクション
システム側

・システム（清水）は，ユーザ（湯川）を誘ってください．あなたの作るシステムはユーザを誘い約束を取り付けるシステムです．
・開発者はシステムの性別を決め，その性別同士（男性同士，もしくは，女性同士）の対話ができるようにしてください．
・開発者はシステム側の≪背景≫のみを考慮してください．ユーザ側の≪背景≫を清水（システム）は知らないものとします．
・システム（清水）は，≪背景≫の「早速連絡しよう」という台詞が示すように，連絡の開始部（例：挨拶）から会話を始めてください．
・会話はシステム（清水）側から始めます．（ユーザがenterキーを押すとシステムが会話を始めます）
ユーザ側

・ユーザ（湯川）はシステム（清水）の誘いを出来るだけ婉曲に断り続けてください。理由は、苦手な小林が参加することです．
・ユーザ（湯川）側の≪背景≫は，システム（清水）からの誘いの後，会話が展開とともに明らかになっていくシチュエーションです．まずはシステム（清水）から，連絡の開始部（例：挨拶）が行われるものとします．
・会話はシステム（清水）側から始めます．（ユーザがenterキーを押すとシステムが会話を始めます
評価方法・基準
対話システムは，「どれくらいシチュエーションに適しており，かつ，人らしい会話か」という1つの評価軸を用いて5段階で総合的に評価されます．「シチュエーションに適している」とは，所定の状況に鑑み「人らしい会話」であると直感的に思えることです．
「人らしい会話」とは，具体的には以下のような特徴を含みます．
・言いにくいことを言わなければならない場合は，相手との社会的な関係性を考慮して，相手に失礼にならないように内容を伝えられること．
・適当な「間」や「あいづち」，「フィラー」，「言い淀み」などが用いられていること．
・一つの話題に固執することなく，会話の流れに沿って，別の話題に自然に推移できること．
これらは，「シチュエーションに適しており，かつ，人らしい会話」というもののイメージを喚起する参考であり，すべてを満たす必要があるということではありません．

予選では，上記の評価基準を用い，クラウドワーカーによる評価を実施します．
本選では，オーガナイザが指定する対話者が，システムと会話をし，参加者全員で評価します．この対話の点数をシステムの点数とします．
なお，今年度は，オーガナイザが指定した対話者以外の対話による評価も検討しています．詳細が決まりましたら追ってお知らせいいたします．
評価の流れ
評価者には，対話の相手がシステムであることはあらかじめ通知されます．対話はシステム発話から始まり，システムとユーザは交互に発話するものとし，それぞれ15発話ずつ（※発話数については変更の可能性があります）行った時点で対話は終了することとします．対話システムはトラックそれぞれの評価基準・手順に基づき評価されます．
予選では，クラウドソーシングを用いて，50人程度のワーカーにより主観評価されます．予選で高い評価を得たシステムが，ライブイベントに参加できます．ライブイベントでは対話システムがシンポジウム参加者と対話し，その状況をシンポジウムの参加者全員でそれぞれのトラックの基準により鑑賞・評価します．なお，今年度は，オーガナイザが選定した対話者以外の対話による評価も検討しています．また，予選の前に疎通に問題ないか，最低限の対話ができるかなどを確認するためのスクリーニングを，オーガナイザと数名のクラウドワーカーにより実施します．本スクリーニングを通過しなかったシステムはその時点で評価の対象外となります．
本選の実施の方法については， 評価方法・基準 をご覧ください．