#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import csv
import pickle
import operator
import datetime
import os

# --- بخش تعریف آرگومان‌ها ---
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='نام دیتاست: diginetica/yoochoose/sample')
parser.add_argument('--sample_portion', type=float, default=1.0, help='کسری از داده‌های آموزش برای تست سریع (مثلا 0.1 برای ۱۰ درصد)')
opt = parser.parse_args()
print(opt)

# --- تعیین نام فایل دیتاست بر اساس آرگومان ورودی ---
dataset_file = 'sample_train-item-views.csv'
if opt.dataset == 'diginetica':
    dataset_file = 'diginetica/train-item-views.csv' # مسیر را در صورت نیاز تغییر دهید
elif opt.dataset == 'yoochoose':
    dataset_file = '/kaggle/input/recsys-challenge-2015/yoochoose-clicks.dat' # مسیر را در صورت نیاز تغییر دهید

print(f"-- شروع پردازش @ {datetime.datetime.now()}")

# --- خواندن داده‌ها و استخراج جلسات (sessions) ---
with open(dataset_file, "r") as f:
    # برای yoochoose که هدر ندارد و با کاما جدا شده
    if opt.dataset == 'yoochoose':
        reader = csv.reader(f, delimiter=',')
        # sessionId, timestamp, itemId, category
        fieldnames = ['session_id', 'timestamp', 'item_id', 'category']
        reader = csv.DictReader(f, fieldnames=fieldnames, delimiter=',')
    else: # برای diginetica و sample که هدر دارند
        reader = csv.DictReader(f, delimiter=',')

    sess_clicks = {}
    sess_date = {}
    ctr = 0
    curid = -1
    curdate = None
    for data in reader:
        sessid = data['session_id']
        if curdate and not curid == sessid:
            # فرمت تاریخ برای دیتاست‌های مختلف متفاوت است
            if opt.dataset == 'yoochoose':
                date = time.mktime(time.strptime(curdate, '%Y-%m-%dT%H:%M:%S.%fZ'))
            else: # diginetica
                 date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            sess_date[curid] = date
        
        curid = sessid
        item = data['item_id']
        curdate = data['timestamp']

        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
    
    # پردازش آخرین جلسه
    if opt.dataset == 'yoochoose':
        date = time.mktime(time.strptime(curdate, '%Y-%m-%dT%H:%M:%S.%fZ'))
    else: # diginetica
        date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
    sess_date[curid] = date

print(f"-- خواندن داده‌ها تمام شد @ {datetime.datetime.now()}")

# --- فیلتر کردن داده‌ها ---
# حذف جلسات با طول ۱
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]

# شمارش تعداد تکرار هر آیتم
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        iid_counts[iid] = iid_counts.get(iid, 0) + 1

# حذف آیتم‌های با تکرار کمتر از ۵ و جلساتی که بعد از آن کوتاه می‌شوند
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
    if len(filseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq

# --- تقسیم داده به آموزش و تست ---
dates = list(sess_date.items())
maxdate = max(v for k, v in dates)
splitdate = maxdate - 86400  # ۱ روز آخر برای تست

tra_sess = [(k, v) for k, v in dates if v < splitdate]
tes_sess = [(k, v) for k, v in dates if v >= splitdate]

# مرتب‌سازی بر اساس تاریخ
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))

# --- بخش نمونه‌برداری از داده‌های آموزش ---
if opt.sample_portion < 1.0:
    print(f'!!! هشدار: فقط از {opt.sample_portion * 100:.0f}% داده‌های آموزش برای این اجرا استفاده می‌شود !!!')
    sample_size = int(len(tra_sess) * opt.sample_portion)
    tra_sess = tra_sess[:sample_size]

print(f"تعداد جلسات آموزش: {len(tra_sess)}")
print(f"تعداد جلسات تست: {len(tes_sess)}")
print(f"-- تقسیم داده تمام شد @ {datetime.datetime.now()}")

# --- ساخت دیکشنری آیتم و پردازش نهایی دنباله‌ها ---
item_dict = {}
item_ctr = 1

def obtian_tra():
    global item_ctr
    train_seqs = []
    for s, date in tra_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i not in item_dict:
                item_dict[i] = item_ctr
                item_ctr += 1
            outseq.append(item_dict[i])
        if len(outseq) >= 2:
            train_seqs.append(outseq)
    print(f"تعداد آیتم‌های منحصر به فرد: {item_ctr}")
    return train_seqs

def obtian_tes():
    test_seqs = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq = [item_dict[i] for i in seq if i in item_dict]
        if len(outseq) >= 2:
            test_seqs.append(outseq)
    return test_seqs

tra_seqs = obtian_tra()
tes_seqs = obtian_tes()

def process_seqs(iseqs):
    out_seqs = []
    labs = []
    for seq in iseqs:
        for i in range(1, len(seq)):
            labs.append(seq[-i])
            out_seqs.append(seq[:-i])
    return out_seqs, labs

tr_seqs, tr_labs = process_seqs(tra_seqs)
te_seqs, te_labs = process_seqs(tes_seqs)

tra = (tr_seqs, tr_labs)
tes = (te_seqs, te_labs)

print(f"تعداد دنباله‌های آموزش: {len(tr_seqs)}")
print(f"تعداد دنباله‌های تست: {len(te_seqs)}")
all_len = sum(len(s) for s in tr_seqs) + sum(len(s) for s in te_seqs)
print(f'میانگین طول دنباله‌ها: {all_len / (len(tr_seqs) + len(te_seqs)):.2f}')

# --- ذخیره نتایج ---
output_dir = opt.dataset
if 'yoochoose' in opt.dataset:
    # برای yoochoose، خروجی‌ها را در پوشه‌های 1/4 و 1/64 نیز ذخیره می‌کنیم
    output_dir_4 = 'yoochoose1_4'
    output_dir_64 = 'yoochoose1_64'
    if not os.path.exists(output_dir_4): os.makedirs(output_dir_4)
    if not os.path.exists(output_dir_64): os.makedirs(output_dir_64)
    
    pickle.dump(tes, open(os.path.join(output_dir_4, 'test.txt'), 'wb'))
    pickle.dump(tes, open(os.path.join(output_dir_64, 'test.txt'), 'wb'))

    split4 = int(len(tr_seqs) * 0.25)
    split64 = int(len(tr_seqs) * (1/64))

    tra4 = (tr_seqs[-split4:], tr_labs[-split4:])
    seq4 = [s for s, l in zip(tr_seqs[-split4:], tr_labs[-split4:])]

    tra64 = (tr_seqs[-split64:], tr_labs[-split64:])
    seq64 = [s for s, l in zip(tr_seqs[-split64:], tr_labs[-split64:])]
    
    pickle.dump(tra4, open(os.path.join(output_dir_4, 'train.txt'), 'wb'))
    pickle.dump(seq4, open(os.path.join(output_dir_4, 'all_train_seq.txt'), 'wb'))
    pickle.dump(tra64, open(os.path.join(output_dir_64, 'train.txt'), 'wb'))
    pickle.dump(seq64, open(os.path.join(output_dir_64, 'all_train_seq.txt'), 'wb'))
else:
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    pickle.dump(tra, open(os.path.join(output_dir, 'train.txt'), 'wb'))
    pickle.dump(tes, open(os.path.join(output_dir, 'test.txt'), 'wb'))
    pickle.dump(tra_seqs, open(os.path.join(output_dir, 'all_train_seq.txt'), 'wb'))

print('--- پردازش با موفقیت انجام شد ---')