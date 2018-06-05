import numpy as np

prob_file = "prob.npy"
labels_file = "labels.npy"

all_prob = np.load(prob_file)
all_labels = np.load(labels_file)

bag_size, num_class = all_prob.shape # bag level predict
mask = np.ones([num_class])
mask[0]=0
mask_prob = all_prob*mask
mask_prob = mask_prob.flatten()
idx_prob = mask_prob.argsort()

one_hot_labels = np.zeros([bag_size, num_class])
one_hot_labels[np.arange(bag_size), all_labels] = 1
one_hot_labels = one_hot_labels.flatten()


idx = idx_prob[-100:][::-1]
p100 = np.mean(one_hot_labels[idx])
idx = idx_prob[-200:][::-1]
p200 = np.mean(one_hot_labels[idx])
idx = idx_prob[-500:][::-1]
p500 = np.mean(one_hot_labels[idx])

print("p@100: %.3f p@200: %.3f p@500: %.3f" % (p100, p200, p500))

# all_prob = all_prob*mask
# prob_and_labels = list(zip(all_prob.flatten(), one_hot_labels))
# prob_and_labels.sort(reverse = True)
# correct = 0.0
# total_relation_facts = np.sum(all_labels)

# for i in range(500):
#   prob, label = prob_and_labels[i]
#   label = int(label)
#   if label == 1:
#     correct += 1
#   precision = correct / (i + 1)
#   recall = correct / total_relation_facts
#   if i==100-1:
#     print('p@100: %.3f' % precision)
#   if i==200-1:
#     print('p@200: %.3f' % precision)
#   if i==500-1:
#     print('p@500: %.3f' % precision)
#   # f.write('{0:.6f}\t{1:.6f}\t{2:.6f}\t{3}\n'.format(precision, recall, score, label))
    