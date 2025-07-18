#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import argparse
import time
import csv
import pickle
import operator
import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose/sample')
opt = parser.parse_args()
print(opt)

dataset = 'sample_train-item-views.csv'
if opt.dataset == 'diginetica':
    dataset = '/kaggle/input/diginetica-dataset/train-item-views.csv'
elif opt.dataset == 'yoochoose':
    dataset = '/kaggle/input/recsys-challenge-2015/yoochoose-clicks.dat'

print("-- Starting @ %ss" % datetime.datetime.now())

# خواندن فایل بدون هدر با مشخص کردن نام فیلدها
with open(dataset, "r") as f:
    reader = csv.DictReader(f, 
                          delimiter=',',
                          fieldnames=['session_id', 'timestamp', 'item_id', 'category'])
    
    sess_clicks = {}
    sess_date = {}
    ctr = 0
    curid = -1
    curdate = None
    
    for data in reader:
        sessid = data['session_id']
        if curdate and not curid == sessid:
            date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
            sess_date[curid] = date
        
        curid = sessid
        item = data['item_id']  # فقط از item_id استفاده می‌کنیم
        curdate = data['timestamp']

        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
    
    # پردازش آخرین session
    date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
    sess_date[curid] = date

print("-- Reading data @ %ss" % datetime.datetime.now())

# حذف sessionهای با طول 1
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]

# شمارش تعداد دفعات مشاهده هر آیتم
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

# فیلتر کردن آیتم‌های با تعداد کمتر از 5
length = len(sess_clicks)
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
    if len(filseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq

# تقسیم داده به train و test بر اساس تاریخ
dates = list(sess_date.items())
maxdate = dates[0][1]

for _, date in dates:
    if maxdate < date:
        maxdate = date

# 1 روز برای تست (مطابق با yoochoose)
splitdate = maxdate - 86400 * 1
print('Splitting date', splitdate)

tra_sess = filter(lambda x: x[1] < splitdate, dates)
tes_sess = filter(lambda x: x[1] > splitdate, dates)

# مرتب‌سازی sessionها بر اساس تاریخ
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))

print(len(tra_sess))  # تعداد sessionهای train
print(len(tes_sess))  # تعداد sessionهای test
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

# ایجاد دیکشنری برای نگاشت item_id به شناسه‌های جدید
item_dict = {}
item_ctr = 1

def obtian_tra():
    global item_ctr
    train_ids = []
    train_seqs = []
    train_dates = []
    for s, date in tra_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2:
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
    print("تعداد آیتم‌های منحصر به فرد:", item_ctr)
    return train_ids, train_dates, train_seqs

def obtian_tes():
    test_ids = []
    test_seqs = []
    test_dates = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
    return test_ids, test_dates, test_seqs

tra_ids, tra_dates, tra_seqs = obtian_tra()
tes_ids, tes_dates, tes_seqs = obtian_tes()

def process_seqs(iseqs, idates):
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [id]
    return out_seqs, out_dates, labs, ids

tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)

tra = (tr_seqs, tr_labs)
tes = (te_seqs, te_labs)

print("تعداد دنباله‌های train:", len(tr_seqs))
print("تعداد دنباله‌های test:", len(te_seqs))

# محاسبه میانگین طول sessionها
all = 0
for seq in tra_seqs:
    all += len(seq)
for seq in tes_seqs:
    all += len(seq)
print('میانگین طول sessionها:', all/(len(tra_seqs) + len(tes_seqs) * 1.0))

# ذخیره نتایج
if opt.dataset == 'diginetica':
    if not os.path.exists('diginetica'):
        os.makedirs('diginetica')
    pickle.dump(tra, open('diginetica/train.txt', 'wb'))
    pickle.dump(tes, open('diginetica/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('diginetica/all_train_seq.txt', 'wb'))
elif opt.dataset == 'yoochoose':
    if not os.path.exists('yoochoose1_4'):
        os.makedirs('yoochoose1_4')
    if not os.path.exists('yoochoose1_64'):
        os.makedirs('yoochoose1_64')
    pickle.dump(tes, open('yoochoose1_4/test.txt', 'wb'))
    pickle.dump(tes, open('yoochoose1_64/test.txt', 'wb'))

    split4, split64 = int(len(tr_seqs) / 4), int(len(tr_seqs) / 64)
    print("اندازه train برای yoochoose1_4:", len(tr_seqs[-split4:]))
    print("اندازه train برای yoochoose1_64:", len(tr_seqs[-split64:]))

    tra4, tra64 = (tr_seqs[-split4:], tr_labs[-split4:]), (tr_seqs[-split64:], tr_labs[-split64:])
    seq4, seq64 = tra_seqs[tr_ids[-split4]:], tra_seqs[tr_ids[-split64]:]

    pickle.dump(tra4, open('yoochoose1_4/train.txt', 'wb'))
    pickle.dump(seq4, open('yoochoose1_4/all_train_seq.txt', 'wb'))

    pickle.dump(tra64, open('yoochoose1_64/train.txt', 'wb'))
    pickle.dump(seq64, open('yoochoose1_64/all_train_seq.txt', 'wb'))
else:
    if not os.path.exists('sample'):
        os.makedirs('sample')
    pickle.dump(tra, open('sample/train.txt', 'wb'))
    pickle.dump(tes, open('sample/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('sample/all_train_seq.txt', 'wb'))

print('پردازش با موفقیت انجام شد.')