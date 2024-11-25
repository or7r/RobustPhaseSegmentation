#!/usr/bin/python3.6
# adapted from: https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/metrics.py

import itertools
import os
import numpy as np
import xlsxwriter
from skimage import measure
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import jaccard_score

from visualization import segment_bars, plot_video_gestures

def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i+1)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)
    
    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def evaluate(dataset, split, exp_name, base_data_folder, gt_path, additional_relaxed_eval=False, parition="test"):
    print("Evaluate dataset {} in split {} for single stamp supervision".format(dataset, split))

    recog_path = "./results/" + dataset + f"/{exp_name}/split_" + split + '/'
    # recog_path = "./results/" + dataset + f"/{exp_name}/split_" + split + '_certain-v2/'

    ground_truth_path = gt_path
    file_list = base_data_folder + dataset + f"/splits/{parition}.split" + split + ".bundle"

    list_of_videos = read_file(file_list).split('\n')[:-1]

    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

    file_name = './result/' + f"{exp_name}_{dataset}_{split}_{parition}" + '.xlsx'
    os.makedirs('./result/', exist_ok=True)

    visualization_folder = './visualization/' + f"{exp_name}_{dataset}_{split}_{parition}" + '/'
    os.makedirs(visualization_folder, exist_ok=True)

    workbook = xlsxwriter.Workbook(file_name)
    worksheet = workbook.add_worksheet()
    metrics = ['F1@10', 'F1@25', 'F1@50', 'Edit', 'Acc']
    row = 0
    col = 0
    for m in range(len(metrics)):
        worksheet.write(row, col, metrics[m])
        col += 1

    row += 1
    col = 0

    all_gt = []
    all_recog = []

    correct = 0
    total = 0
    edit = 0

    for vid in list_of_videos:
        gt_file = ground_truth_path + vid
        gt_content = read_file(gt_file).split('\n')[0:-1]
        
        recog_file = recog_path + vid.split('.')[0]
        recog_content = read_file(recog_file).split('\n')[1].split()

        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == recog_content[i]:
                correct += 1
        
        edit += edit_score(recog_content, gt_content)

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1

        all_gt.append(gt_content)
        all_recog.append(recog_content)
    
    metric_values = [] 

    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s]+fp[s])
        recall = tp[s] / float(tp[s]+fn[s])
    
        f1 = 2.0 * (precision*recall) / (precision+recall)
        f1 = np.nan_to_num(f1)*100
        print('F1@%0.2f: %.4f' % (overlap[s], f1))

        worksheet.write(row, col, round(f1, 4))
        metric_values.append(round(f1, 4))
        col += 1

    edit = (1.0 * edit) / len(list_of_videos)
    acc = 100 * float(correct) / total
    

    worksheet.write(row, col, round(edit, 4))
    worksheet.write(row, col + 1, round(acc, 4))

    metric_values.append(round(edit, 4))
    metric_values.append(round(acc, 4))


    mapping_dict = create_mapping_dict(list(itertools.chain(*all_gt)))
    all_gt = [np.vectorize(mapping_dict.get)(l) for l in all_gt]
    all_recog = [np.vectorize(mapping_dict.get)(p) for p in all_recog]


    # add prec, recall, f1 and jaccard
    # additional_metrics = "per-frame-precision, per-frame-recall, per-frame-f1, per-frame-jaccard".split(", ")
    additional_metrics = "macro-precision, macro-recall, macro-f1, macro-jaccard, micro-acc".split(", ")
    res_mean = []
    res_std = []

    # use sklearn precision_recall_fscore_support
    for sub_labels, sub_preds in zip(all_gt, all_recog):
        len_diff = abs(len(sub_labels) - len(sub_preds)) 
        if len_diff > 0 and len_diff < 3:
                # interpolate sub_preds to match sub_labels, if a small difference occurs
                sub_preds = np.interp(np.arange(len(sub_labels)), np.arange(len(sub_preds)), sub_preds)

        prec, rec, f1, _ = score(sub_labels, sub_preds, average='macro')

        # calc micro acc
        acc = np.sum(sub_labels == sub_preds) * 100 / len(sub_labels)

        # calc jaccard usinf jaccard_score
        jacc = jaccard_score(sub_labels, sub_preds, average='macro')
        

        additional_metrics_mean = [np.mean(prec), np.mean(rec), np.mean(f1), np.mean(jacc), acc]
        # additional_metrics_std = [np.std(prec), np.std(rec), np.std(f1), np.std(jacc)]

        res_mean.append(additional_metrics_mean)
        # res_std.append(additional_metrics_std)
    
    additional_metrics_mean = np.mean(np.stack(res_mean), axis=0)
    additional_metrics_std = np.std(np.stack(res_mean), axis=0)

    for m in range(len(additional_metrics_mean)):
        worksheet.write(row, col + 2 + m, round(additional_metrics_mean[m], 4))
        metric_values.append(round(additional_metrics_mean[m], 4))
    for m in range(len(additional_metrics_std)):
        worksheet.write(row, col + 2 + len(additional_metrics_mean) + m, round(additional_metrics_std[m], 4))
        metric_values.append(round(additional_metrics_std[m], 4))
    
    metrics = metrics + [f"{m}_mean" for m in additional_metrics] + [f"{m}_std" for m in additional_metrics]



    if additional_relaxed_eval:
        additional_metrics_mean, addition_metrics_std = compute_phase_scores(all_gt, all_recog, agg="video_relaxed")
        additional_metrics = ['relaxed_acc', 'relaxed_precision', 'relaxed_recall', 'relaxed_f1', 'relaxed_jaccard']
        for m in range(len(additional_metrics)):
            worksheet.write(row, col + 2 + m, round(additional_metrics_mean[m], 4))
            metric_values.append(round(additional_metrics_mean[m], 4))
        for m in range(len(additional_metrics)):
            worksheet.write(row, col + 2 + len(additional_metrics) + m, round(addition_metrics_std[m], 4))
            metric_values.append(round(addition_metrics_std[m], 4))
        
        metrics = metrics + [f"{m}_mean" for m in additional_metrics] + [f"{m}_std" for m in additional_metrics]

    # visualize
    for i in range(len(all_gt)):
        plot_video_gestures(os.path.splitext(list_of_videos[i])[0], 
                            [all_recog[i], all_gt[i]], 
                            ["recog", "gt"], 
                            visualization_folder)



    print('Edit: %.4f' % edit)
    print("Acc: %.4f" % acc)

    workbook.close()

    metric_values = {metrics[i]: metric_values[i] for i in range(len(metrics))}

    return metric_values

def create_mapping_dict(l):
    # get uinique labels
    unique_labels = np.unique(l)
    # create mapping dict
    mapping_dict = {unique_labels[i]: i for i in range(len(unique_labels))}
    return mapping_dict



def compute_phase_scores(labels, predicts, agg="video_relaxed"):


    if agg == 'class' and len(labels) == 0: return [-1] * 7,[]
    if len(labels) == 0: return [-1] * 5, []
    preds = predicts
    if agg == 'frame':

        labels, preds = np.concatenate(labels), np.concatenate(predicts)

        scores = score(labels, preds)

        acc = np.sum(labels == preds) * 100 / len(labels)
        acc = np.around(acc, 2)

        mean = np.mean(np.vstack(scores).T, axis=0)
        mean[:-1] *= 100
        mean = np.around(mean, 2)
        mean = [acc] + mean.tolist()

        std = np.std(np.vstack(scores).T, axis=0)
        std[:-1] *= 100
        std = np.around(std, 2)
        std = std.tolist()

    elif agg == 'class':
        
        class_f1 = []
        for sub_labels, sub_preds in zip(labels, preds):
            # compute F1
            vid_score = class_metrics(sub_labels, sub_preds)
            class_f1.append(np.array(vid_score[2])*100)

        print(len(class_f1), len(class_f1[0]))
        mean = np.around(np.nanmean(class_f1, axis=0), 2).tolist()
        std = np.around(np.nanstd(class_f1, axis=0), 2).tolist()
    elif agg == 'video':
        # split labels, preds by video
        accs = []
        scores = []
        for sub_labels, sub_preds in zip(labels, preds):

            # compute acc and append
            vid_acc = np.sum(sub_labels == sub_preds) * 100 / len(sub_labels)
            accs.append(vid_acc)

            # compute F1
            vid_score = score(sub_labels, sub_preds)
            mean = np.mean(np.vstack(vid_score).T, axis=0)
            mean[:-1] *= 100
            scores.append(mean)

        # summarize
        overall_acc = np.mean(np.stack(accs))
        overall_acc = np.around(overall_acc, 2)

        overall_f1 = np.mean(np.stack(scores), axis=0)
        overall_f1 = np.around(overall_f1, 2)

        mean = [overall_acc] + overall_f1.tolist()

        std = np.std(np.stack(scores), axis=0)
        std = np.around(std, 2)
        std = [np.std(np.stack(accs))] + std.tolist()

    elif agg == 'video_relaxed':
        # split labels, preds by video
        accs = []
        scores = []
        for sub_labels, sub_preds in zip(labels, preds):

            len_diff = abs(len(sub_labels) - len(sub_preds))
            if len_diff > 0 and len_diff < 3:
                # interpolate sub_preds to match sub_labels, if a small difference occurs
                sub_preds = np.interp(np.arange(len(sub_labels)), np.arange(len(sub_preds)), sub_preds)
                

            vid_prec, vid_rec, vid_f1, vid_jacc, vid_acc = compute_phase_relaxed_scores(sub_preds,
                    sub_labels)
            accs.append(vid_acc)
            scores.append([np.nanmean(vid_prec), np.nanmean(vid_rec), np.nanmean(vid_f1), np.nanmean(vid_jacc)])

        mean = [np.mean(np.stack(accs))] + np.mean(np.stack(scores), axis=0).tolist()
        std = [np.std(np.stack(accs))] + np.std(np.stack(scores), axis=0).tolist()

    return mean, std

def class_metrics(labels, predictions, num_cls=7):
    exp_labels = np.array(range(num_cls))
    missing = [idx for idx in exp_labels if idx not in labels and idx not in predictions]
    class_score = score(labels, predictions)
    for miss in missing:
        class_score = [np.insert(np.float32(sc), miss, np.nan) for sc in class_score]

    return class_score



def compute_phase_relaxed_scores(preds, targets, boundary_size=10):
    #EVALUATE
    # A function to evaluate the performance of the phase recognition method
    # providing jaccard index, precision, and recall for each phase 
    # and accuracy over the surgery. All metrics are computed in a relaxed
    # boundary mode.
    # OUTPUT:
    #    res: the jaccard index per phase (relaxed) - NaN for non existing phase in GT
    #    prec: precision per phase (relaxed)        - NaN for non existing phase in GT
    #    rec: recall per phase (relaxed)            - NaN for non existing phase in GT
    #    acc: the accuracy over the video (relaxed)
    res, prec, rec = [], [], []
    diff = preds - targets
    updatedDiff = diff.copy()

    # obtain the true positive with relaxed boundary
    for iPhase in range(7):
        labels, num = measure.label(targets == iPhase, return_num=True)

        for iConn in range(1, num + 1):
            comp = np.argwhere(labels == iConn)
            startIdx = np.min(comp)
            endIdx = np.max(comp) + 1

            curDiff = diff[startIdx:endIdx]

            # in the case where the phase is shorter than the relaxed boundary
            t = boundary_size
            if t > len(curDiff):
                t = len(curDiff)

            # relaxed boundary
            # revised for cholec80 dataset !!!!!!!!!!!
            if iPhase == 3 or iPhase == 4: # Gallbladder dissection and packaging might jump between two phases
                curDiff[:t][curDiff[:t] == -1] = 0 # late transition

                # early transition, 5 can be predicted as 6/7 at the end > 5 followed by 6/7
                curDiff[-t:][curDiff[-t:] == 1] = 0
                curDiff[-t:][curDiff[-t:] == 2] = 0

            elif iPhase == 5 or iPhase == 6: # Gallbladder dissection might jump between two phases
                # late transition
                curDiff[:t][curDiff[:t] == -1] = 0
                curDiff[:t][curDiff[:t] == -2] = 0

                # early transition
                curDiff[-t:][curDiff[-t:] == 1] = 0
                curDiff[-t:][curDiff[-t:] == 2] = 0

            else:
                # general situation
                curDiff[:t][curDiff[:t] == -1] = 0 # late transition
                curDiff[-t:][curDiff[-t:] == 1] = 0 # early transition

            updatedDiff[startIdx:endIdx] = curDiff

    # compute jaccard index, prec, and rec per phase
    for iPhase in range(7):
        gt_num = (targets == iPhase).sum()
        if gt_num == 0:
            # no iPhase in current ground truth, assigned NaN values
            # SHOULD be excluded in the computation of mean (use nanmean)
            res.append(np.nan)
            prec.append(np.nan)
            rec.append(np.nan)
            continue

        # get all indices where pred is iPhase
        tp_and_fp = np.argwhere(preds == iPhase).flatten()
        tp_and_fn = np.argwhere(targets == iPhase).flatten()
        union = np.union1d(tp_and_fp, tp_and_fn)

        # compute tp
        tp = np.sum(updatedDiff[tp_and_fp] == 0)

        # divide by union to get jaccard
        jaccard = tp / len(union)
        jaccard = jaccard * 100

        res.append(jaccard)

        # Compute prec and rec
        prec.append(tp * 100 / len(tp_and_fp))
        rec.append(tp * 100 / len(tp_and_fn))

    # compute accuracy
    acc = sum(updatedDiff == 0) / len(targets)
    acc = acc * 100

    # compute f1
    prec = np.array(prec)
    rec = np.array(rec)
    f1 = 2 * prec * rec / (prec + rec)
    res = np.array(res)

    return prec, rec, f1, res, acc