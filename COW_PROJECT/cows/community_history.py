#-*- encoding:utf-8 -*-
import datetime

class CommunityMembers:
    """ キーとなる時刻とその時のコミュニティ（個体番号 (int) のリスト） を持つクラス"""
    _time:datetime.datetime
    _community_members:list


    def __init__(self, dt:datetime, community_members):
        """ クラス作成時に二つの要素を登録 """
        self._time = dt
        self._community_members = community_members


    def get_time(self):
        return self._time


    def get_community_members(self):
        return self._community_members


    def confirm_if_inside_times(self, start, end):
        """ 引数として与えられた2つの時刻間にこのクラスのフィールドであるtimeが入っているかを確認する """
        if (start <= self._time and self._time < end):
            return True
        else:
            return False