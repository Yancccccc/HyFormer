import numpy as np

def splite_classes(txt_path):
    data_xy = np.loadtxt(txt_path, dtype=np.int32)
    y = data_xy[:,0:3]
    y[:,[1,2]]=y[:,[2,1]]
    b = len(y)
    n = []
    c1, c2, c3, c4, c5, c6, c7, c8 = [],[],[],[],[],[],[],[]
    for i in range(b):
        if y[i,0] == 1:
            n.append(i)
    for i in range(b):
        if (i >= n[0]) & (i < n[1]):
            y[i,0] = 1
        if (i >= n[1]) & (i < n[2]):
            y[i,0] = 2
        if (i >= n[2]) & (i < n[3]):
            y[i,0] = 3
        if (i >= n[3]) & (i < n[4]):
            y[i,0] = 4
        if (i >= n[4]) & (i < n[5]):
            y[i,0] = 5
        if (i >= n[5]) & (i < n[6]):
            y[i,0] = 6
        if (i >= n[6]) & (i < n[7]):
            y[i,0] = 7
        if (i >= n[7]):
            y[i,0] = 8
    for i in range(b):
        if y[i,0] == 1:
            c1.append(y[i,:])
        if y[i,0] == 2:
            c2.append(y[i,:])
        if y[i,0] == 3:
            c3.append(y[i,:])
        if y[i,0] == 4:
            c4.append(y[i,:])
        if y[i,0] == 5:
            c5.append(y[i,:])
        if y[i,0] == 6:
            c6.append(y[i,:])
        if y[i,0] == 7:
            c7.append(y[i,:])
        if y[i,0] == 8:
            c8.append(y[i,:])
    c1 = np.array(c1)
    c2 = np.array(c2)
    c3 = np.array(c3)
    c4 = np.array(c4)
    c5 = np.array(c5)
    c6 = np.array(c6)
    c7 = np.array(c7)
    c8 = np.array(c8)
    return c1, c2, c3, c4, c5, c6, c7, c8

def random_split(input, ratio):
    #data_xy = np.loadtxt(post_path, dtype=np.int32)
    y = input
    #y = data_xy[:,0:3]
    #y[:,[1,2]]=y[:,[2,1]]
    b = len(y)
    train_dataset, test_dataset, val_dataset = [], [] ,[]
    indices = list(range(b))
    offset1 = int(np.floor(ratio * b))
    np.random.shuffle(indices)
    test_and_val_indices, train_indices = indices[offset1:], indices[:offset1]
    c = len(test_and_val_indices)
    indices_test_and_val = list(range(c))
    offset2 = int(np.floor(0.5 * c))
    np.random.shuffle(indices_test_and_val)
    test_indices, val_indices = indices_test_and_val[offset2:], indices_test_and_val[:offset2]
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    val_indices = np.array(val_indices)
    for i in range(len(train_indices)):
        train_dataset.append(y[train_indices[i],:])
    train_dataset = np.array(train_dataset)

    for i in range(len(test_indices)):
        test_dataset.append(y[test_indices[i]])
    test_dataset = np.array(test_dataset)

    for i in range(len(val_indices)):
        val_dataset.append(y[val_indices[i]])
    val_dataset = np.array(val_dataset)

    return train_dataset, test_dataset, val_dataset

def creat_dataset(txt_path):
    c1, c2, c3, c4, c5, c6, c7, c8 = splite_classes(txt_path)
    c1_train, c1_test, c1_val = random_split(c1, ratio=0.6)
    c2_train, c2_test, c2_val = random_split(c2, ratio=0.6)
    c3_train, c3_test, c3_val = random_split(c3, ratio=0.6)
    c4_train, c4_test, c4_val = random_split(c4, ratio=0.6)
    c5_train, c5_test, c5_val = random_split(c5, ratio=0.6)
    c6_train, c6_test, c6_val = random_split(c6, ratio=0.6)
    c7_train, c7_test, c7_val = random_split(c7, ratio=0.6)
    c8_train, c8_test, c8_val = random_split(c8, ratio=0.6)
    train_data =  np.vstack((c1_train, c2_train, c3_train, c4_train, c5_train, c6_train, c7_train, c8_train))
    test_data = np.vstack((c1_test, c2_test, c3_test, c4_test, c5_test, c6_test, c7_test, c8_test))
    val_data = np.vstack((c1_val, c2_val, c3_val, c4_val, c5_val, c6_val, c7_val, c8_val))
    train_xy = train_data[:, 1:]
    train_label = train_data[:, 0]
    test_xy = test_data[:, 1:]
    test_label = test_data[:, 0]
    val_xy = val_data[:, 1:]
    val_label = val_data[:, 0]

    i, j = train_xy.shape
    a, b = test_xy.shape
    m, n = val_xy.shape
    train_xy1 = np.zeros([i, j])
    test_xy1 = np.zeros([a, b])
    val_xy1 = np.zeros([m, n])

    for i1 in range(i):
        for j1 in range(j):
            train_xy1[i1, j1] = train_xy[i1, j1] - 1

    for a1 in range(a):
        for b1 in range(b):
            test_xy1[a1, b1] = test_xy[a1, b1] - 1

    for m1 in range(m):
        for n1 in range(n):
            val_xy1[m1, n1] = val_xy[m1, n1] - 1

    return train_xy1, train_label, test_xy1, test_label, val_xy1, val_label


if __name__ == "__main__":
    name = '2020'
    txt_path = 'xy/changxing/2020/20200801_3.txt'
    train_xy_correct, train_label, test_xy_correct, test_label, val_xy_correct, val_label = creat_dataset(txt_path)
    np.save('xy/changxing/{}/train_xy_2.npy'.format(name), train_xy_correct)
    np.save('xy/changxing/{}/train_label_2.npy'.format(name), train_label)
    np.save('xy/changxing/{}/test_xy_2.npy'.format(name), test_xy_correct)
    np.save('xy/changxing/{}/test_label_2.npy'.format(name), test_label)
    np.save('xy/changxing/{}/val_xy_2.npy'.format(name), val_xy_correct)
    np.save('xy/changxing/{}/val_label_2.npy'.format(name), val_label)
