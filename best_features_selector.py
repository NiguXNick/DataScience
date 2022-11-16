import pandas as pd
import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt
import csv
from itertools import count
import sys

def a (permission,class_id,ds):
    ds_class = ds[ds["class"] == class_id]
    return len(ds_class[permission][ds_class[permission] != 0])

def b (permission, class_id,ds):
    ds_class = ds[ds["class"] == class_id]
    return len(ds_class[permission][ds_class[permission] != 1])

def c (permission_id,class_id,ds):
    ds_class = ds[ds["class"] != class_id]
    return len(ds_class[permission_id][ds_class[permission_id] != 0])

def d (permission_id,class_id,ds):
    ds_class = ds[ds["class"] != class_id]
    return len(ds_class[permission_id][ds_class[permission_id] != 1])

def N (ds):
    return ds.shape[0]

def n1(class_id, ds):
    ds_class = ds[ds['class'] == class_id]
    return len(ds_class)

def n2(class_id, ds):
    ds_class = ds[ds['class'] != class_id]
    return len(ds_class)

def d1(permission, ds):
    ds_permission = ds[ds[permission] != 0]
    return len(ds_permission)

def d2(permission, ds):
    ds_permission = ds[ds[permission] != 1]
    return len(ds_permission)

best_features = []

def od(permission_id,class_id,ds):
    return (a(permission_id,class_id,ds) * d(permission_id,class_id,ds)) / (b(permission_id,class_id,ds) * c(permission_id,class_id,ds))

def get_od_list(class_id, ds):
    od_list = []
    for permission in ds.columns[:-1]:
        od_list.append([permission, od(permission, class_id, ds)])
    return sorted(od_list, key=lambda x: x[1], reverse=True)


def chi_squared(permission_id,class_id,ds):
    aa = a(permission_id,class_id,ds)
    bb = b(permission_id,class_id,ds)
    cc = c(permission_id,class_id,ds)
    dd = d(permission_id,class_id,ds)
    return (N(ds)*((aa*dd - bb*cc)**2) / ((aa+bb)*(cc+dd)*(aa+cc)*(bb+dd)))
    

def get_chi_squared_list(class_id, ds):
    chi_squared_list = []
    for permission in ds.columns[:-1]:
        chi_squared_list.append([permission, chi_squared(permission, class_id, ds)])
    return sorted(chi_squared_list, key=lambda x: x[1], reverse=True)

def idf(permission_id,class_id,ds):
    return np.log2(N(ds) / (a(permission_id,class_id,ds) + c(permission_id,class_id,ds)))

def get_idf_list(class_id, ds):
    idf_list = []
    for permission in ds.columns[:-1]:
        idf_list.append([permission, idf(permission, class_id, ds)])
    return sorted(idf_list, key=lambda x: x[1], reverse=True)

def dft(permission_id,class_id,ds):
    return a(permission_id,class_id,ds) + c(permission_id,class_id,ds)

def get_dft_list(class_id, ds):
    dft_list = []
    for permission in ds.columns[:-1]:
        dft_list.append([permission, dft(permission, class_id, ds)])
    return sorted(dft_list, key=lambda x: x[1], reverse=True)

def acc(permission_id,class_id,ds):
    return a(permission_id,class_id,ds) - c(permission_id,class_id,ds)

def acc2(permission_id,class_id,ds):
    return (a(permission_id,class_id,ds)/n1(class_id,ds)) - (c(permission_id,class_id,ds)/n2(class_id,ds))

def get_acc_list(class_id, ds):
    acc_list = []
    for permission in ds.columns[:-1]:
        acc_list.append([permission, acc(permission, class_id, ds)])
    return sorted(acc_list, key=lambda x: x[1], reverse=True)

def get_acc2_list(class_id, ds):
    acc2_list = []
    for permission in ds.columns[:-1]:
        acc2_list.append([permission, acc2(permission, class_id, ds)])
    return sorted(acc2_list, key=lambda x: x[1], reverse=True)

def m2(permission_id,class_id,ds):
    return dft(permission_id,class_id,ds) * np.absolute((a(permission_id,class_id,ds)/d1(permission_id,ds)) - (b(permission_id,class_id,ds)/d2(permission_id, ds)))

def get_m2_list(class_id, ds):
    m2_list = []
    for permission in ds.columns[:-1]:
        m2_list.append([permission, m2(permission, class_id, ds)])
    return sorted(m2_list, key=lambda x: x[1], reverse=True)

def rffs(permission_id,class_id,ds):
    return dft(permission_id,class_id,ds) * np.absolute(np.log2(2 + (a(permission_id,class_id,ds)/c(permission_id,class_id,ds))))

def get_rffs_list(class_id, ds):
    rffs_list = []
    for permission in ds.columns[:-1]:
        rffs_list.append([permission, rffs(permission, class_id, ds)])
    return sorted(rffs_list, key=lambda x: x[1], reverse=True)

def pci(class_id,ds):
    return len(ds[ds["class"] == class_id])/len(ds)

def pp(feature, ds, is_used = True):
    value = 1 if is_used else 0
    return len(ds[ds[feature] == value])/len(ds)

def pcip(feature, class_id, ds, is_used = True):
    value = 1 if is_used else 0
    ds_feature = ds[ds[feature] == value]
    return len(ds_feature[ds_feature['class'] == class_id])/len(ds_feature)

def ig(permission_id,ds):
    class_distribution = ds["class"].value_counts()
    a = 0.0
    b = 0.0
    c = 0.0
    for i in class_distribution.index:
        f_value = pci(i,ds)
        if f_value > 0.0:
            a += f_value * np.log2(f_value)
        f_value = pcip(permission_id,i,ds)
        if f_value > 0.0:
            b += f_value * np.log2(f_value)
        f_value = pcip(permission_id,i,ds,False)
        if f_value > 0.0:
            c += f_value * np.log2(f_value)
    b *= pp(permission_id,ds)
    c *= pp(permission_id,ds, False)
    return -1.0 * (a + b + c)

def get_ig_list(class_id, ds):
    ig_list = []
    for permission in ds.columns[:-1]:
        ig_list.append([permission, ig(permission, ds)])
    return sorted(ig_list, key=lambda x: x[1], reverse=True)

def best_features(class_id, ds):
    best_features_set = set()
    best_features_set.update([x[0] for x in get_acc_list(class_id, ds)[:10]])
    best_features_set.update([x[0] for x in get_acc2_list(class_id, ds)[:10]])
    best_features_set.update([x[0] for x in get_m2_list(class_id, ds)[:10]])
    best_features_set.update([x[0] for x in get_rffs_list(class_id, ds)[:10]])
    best_features_set.update([x[0] for x in get_ig_list(class_id, ds)[:10]])
    best_features_set.update([x[0] for x in get_od_list(class_id, ds)[:10]])
    best_features_set.update([x[0] for x in get_chi_squared_list(class_id, ds)[:10]])
    best_features_set = set(best_features_set)
    return list(best_features_set)

def get_best_features_ds(ds):
    best_features_set = best_features(1, ds)
    best_features_set.append("class")
    return ds[best_features_set]


def generate_best_features_ds(ds,name):
    best_features_ds = get_best_features_ds(ds)

    with open(name + '.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(best_features_ds.columns)
        writer.writerows(best_features_ds.values)

def main(ds,name):
    dsa = pd.read_csv(ds)
    generate_best_features_ds(dsa,name)

if __name__ == "__main__":
    ds = sys.argv[1]
    name = sys.argv[2]
    main(ds,name)