import os
import numpy as np
import string
from scipy import stats

dataDir = '/Users/tina/Documents/UofT/Winter 2020/CSC2511/a3/data .nosync'


def Levenshtein(r, h):
    """
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : list of strings
    h : list of strings

    Returns
    -------
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    0.333 0 0 1
    >>> wer("who is there".split(), "".split())
    1.0 0 0 3
    >>> wer("".split(), "who is there".split())
    Inf 0 3 0
    """

    # define variables
    n, m = len(r), len(h)

    # distance matrix initialization
    R = np.zeros((n + 1, m + 1))
    R[0, :] = np.array(list(range(m + 1)))
    R[:, 0] = np.array(list(range(n + 1)))

    # backtrace matrix initialization
    B = np.zeros((n + 1, m + 1), dtype=object)
    B[0, :] = 'left'  # first row (0) is insertion errors
    B[:, 0] = 'up'  # first column (0) is deletion errors
    B[0, 0] = ''  # (0,0) no need to considered

    # update R and B
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            del_error = R[i - 1, j] + 1  # deletion

            sub_error = R[i - 1, j - 1]  # if words match
            if r[i - 1] != h[j - 1]:  # if words differ
                sub_error += 1

            ins_error = R[i, j - 1] + 1  # insertion

            R[i, j] = min([del_error, sub_error, ins_error])

            # update B matrix
            if R[i, j] == del_error:
                B[i, j] = 'up'
            elif R[i, j] == ins_error:
                B[i, j] = 'left'
            else:
                B[i, j] = 'up-left'

    nI, nD = 0, 0
    t = n
    s = m
    while True:
        if t <= 0 and s <= 0:
            break
        step = B[t, s]
        if step == 'up':
            nD += 1
            t = t - 1
        elif step == 'left':
            nI += 1
            s = s - 1
        elif step == 'up-left':
            t = t - 1
            s = s - 1

    wer = 100 * R[n, m] / float(n)
    nS = np.rint(R[n, m] - nI - nD).astype(int)

    return wer, nS, nI, nD


def preprocess_line(line):
    """
    Removes all punctuation (excluding []) from line,
    and sets all remaining letters to lowercase.

    Returns the whitespace-split version of line.
    """
    punc_to_remove = set(string.punctuation) - set("[]")
    return "".join([
        char.lower() if char not in punc_to_remove else " "
        for char in line
    ]).split()


if __name__ == "__main__":
    output = open("asrDiscussion.txt", 'w')
    google_errors, kaldi_errors = [], []
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)
            base_path = os.path.join(dataDir, speaker)
            ref_path = os.path.join(base_path, 'transcripts.txt')
            google_path = os.path.join(base_path, 'transcripts.Google.txt')
            kaldi_path = os.path.join(base_path, 'transcripts.Kaldi.txt')

            # read files
            ref_file = open(ref_path, 'r')
            ref_lines = ref_file.read().split('\n')
            if ref_lines[-1] == '':
                ref_lines = ref_lines[0:-1]
            ref_file.close()

            google_file = open(google_path, 'r')
            google_lines = google_file.read().split('\n')
            if google_lines[-1] == '':
                google_lines = google_lines[0:-1]
            google_file.close()

            kaldi_file = open(kaldi_path, 'r')
            kaldi_lines = kaldi_file.read().split('\n')
            if kaldi_lines[-1] == '':
                kaldiLines = kaldi_lines[0:-1]
            kaldi_file.close()

            length = min(len(ref_lines), len(google_lines), len(kaldi_lines))

            # loop over each line
            for i in range(0, length):
                ref = preprocess_line(ref_lines[i])
                google = preprocess_line(google_lines[i])
                kaldi = preprocess_line(kaldi_lines[i])

                # compute error rate
                google_result = Levenshtein(ref, google)
                google_errors.append(google_result[0])

                kaldi_result = Levenshtein(ref, kaldi)
                kaldi_errors.append(kaldi_result[0])

                # write computed results into file
                output.write('{0} {1} {2} {3: 1.4f} S:{4}, I:{5}, D:{6} \n'.format(speaker, 'Google', i,
                                                                                   google_result[0], google_result[1],
                                                                                   google_result[2],
                                                                                   google_result[3]))

                output.write('{0} {1} {2} {3: 1.4f} S:{4}, I:{5}, D:{6} \n'.format(speaker, 'Kaldi', i,
                                                                                   kaldi_result[0], kaldi_result[1],
                                                                                   kaldi_result[2], kaldi_result[3]))
            output.write('\n\n')

    googleErrors = np.array(google_errors)
    kaldiErrors = np.array(kaldi_errors)

    t_value, p_value = stats.ttest_ind(googleErrors, kaldiErrors, equal_var=False)
    output.write('Google WER Average: {0: 1.4f}, Google WER Standard Deviation: {1: 1.4f}, \
        Kaldi WER Average: {2: 1.4f}, Kaldi WER Standard Deviation: {3: 1.4f}, \
        Calculate T-test for Google WER and Kaldi WER:  T-value: {4: 1.4f}  P-value: {5} \n'.format(
        np.mean(googleErrors), np.std(googleErrors), np.mean(kaldiErrors), np.std(kaldiErrors), t_value, p_value))

    output.close()
