牛の固体番号管理アプリの説明
1. 実行方法
$ python3 cowTagMain.py

2. 内容
csvファイルに各月の牛の個体番号のリストを作成(1)
txtファイルにCowDataFile.javaのコンストラクタに適用可能なソースプログラムを生成(2)

3. ルール
Record/IORecord.csvに牛の入場退出を規定のフォーマットに記載
各月1日の牛のリストをRecord/CowTagLog.csvに保存する(4)
第一放牧場に存在したことのある牛の各月のリストをRecord/ExistRecord.csvに保存する
(1),(2)をする際にはこのCowTagLog.csvの該当月からIOFileの変更にしたがって牛のリストを管理
(例) 10月25日の牛のリストを求める際にはCowTagLog.csvの10月1日の牛のリストから変更分を作成している
IOFileに抜け漏れなどがあった場合はCowTagLog.csvの更新を行う(3)→その後(1),(2)を忘れずに

4. ファイルの保存場所
Recordフォルダ:CowTagLog, IORecord, ExistRecord
csvフォルダ:牛の個体番号リスト
txt:CowDataFileList用のソースコード


GPSデータベース作成アプリの説明
1. 実行方法
$ python3 makeDB.py yymmdd yymmdd

2. 内容
第一引数から第二引数までのGPSデータのデータベースを作成する
データベースは日付 (主キー)・緯度・経度・速さを格納している
手順は
CSVファイルよりその日の牛のリストを取得→牛ごとのIDの対応データベース (日付 (主キー)・GPSID・MobiID)に照合してGPSIDを取得→データのテキストファイルを読み込み・変換
の手順を踏んでいる

3. 注意点
作成する前にその日の牛のリスト (csv/YYYY-mm.csv) が完成しており，その日のIDの対応データベース (DB/TagDB/個体番号.db)が作成されていなければ正しいデータベースは得られない
(optionDB.py)

4. データテーブル等の形式
TagDB:
牛の個体番号.dbにそれぞれ１つずつデータテーブルが存在する
Tableの名前はTagInfo
PosDB:
日付.dbに各牛ごとのデータテーブルが存在する
Tableの名前は牛の個体番号 (5桁) 