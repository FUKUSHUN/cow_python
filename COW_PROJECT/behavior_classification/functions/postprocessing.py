"""
このコードは圧縮後のデータ復元に必要な処理をまとめたものである
特徴出力後のcsvファイルからの復元時に活用する
"""

import sys
import numpy as np

def make_labels(results):
	""" 分類結果のリストを時間のリストに合う形に整形する
	Parameter
		results	: 分類結果のリスト """
	new_results = []
	for r in results:
		new_results.append(r)
		new_results.append(r)
	return new_results


def decompress(t_list, zipped_t_list, zipped_l_list):
	""" 特徴抽出したCSVファイル (圧縮済み) から元の時系列データを作成 (展開) する
	Parameter
		t_list    : 元の時系列データ
		zipped_t_list	: 圧縮後の時系列データ
		zipped_l_list	: 圧縮データから得られるラベルのデータ
	Return
		new_t_list  : 新しい時間のリスト（削除部分との整合性をとるため）
		l_list  : ラベルのリスト
	圧縮後の時系列データは (start-end) を休息 | 歩行の形で記載している
	例) 2018/12/30 13:00:30-2018/12/30 13:00:30 | 2018/12/30 13:00:35-2018/12/30 13:00:45 """
	print(sys._getframe().f_code.co_name, "実行中")
	idx = 0
	new_t_list = []
	l_list = []
	start = zipped_t_list[0][0]
	end = zipped_t_list[0][1]
	label = zipped_l_list[0]
	final = zipped_t_list[len(zipped_t_list)-1][1]
	for time in t_list:
		if (start <= time and time <= end):
			new_t_list.append(time)
			l_list.append(label)
		if (final <= time):
			break
		if (end <= time):
			idx += 1
			start = zipped_t_list[idx][0]
			end = zipped_t_list[idx][1]
			label = zipped_l_list[idx]
	print(sys._getframe().f_code.co_name, "正常終了\n")
	return new_t_list, l_list

def process_result(S,i):
	""" Sの結果からk番目のクラスタに所属するデータをk+iとラベル付けしデータを1次元化する
		Parameter
			S	: K*N行列の形で1ofK表現で書かれている
			i	: ラベル付けの始まりの数字. k番目のクラスタにk+iのラベルを付与する
		Return
			new_S	: ndarray: N次元ベクトル """
	K = len(S) # クラスタ数
	N = len(S.T) # データ数
	new_S = np.zeros(N)
	for n in range(N):
		for k in range(K):
			if (S[k, n] == 1):
				new_S[n] = k + i
	return new_S

def make_new_list(old_t_list, new_t_list, something):
	""" 古い時間のリストのうち，新しい時間のリストに含まれている部分のみを切り取る
	Parameter
		old_t_list  : 古い時間のリスト
		new_t_list  : 新しい時間のリスト
		something   : 何かのリスト（古い時間のリストと一対一対応） """
	idx = 0
	something_new = []
	start = new_t_list[0]
	end = new_t_list[len(new_t_list) - 1]
	for time in old_t_list:
		if (start <= time and time <= end):
			something_new.append(something[idx])
		idx += 1
	return something_new
