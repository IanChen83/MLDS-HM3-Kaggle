import numpy as np
import pdb
from output48_39 import * 
in_file ="count.map"
obserstype = ["^","aa", "ae", "ah", "ao", "aw", "ax", "ay", "b", "ch", "cl", "d", "dh", "dx", "eh", "el"
               , "en", "epi", "er", "ey", "f", "g", "hh", "ih", "ix", "iy", "jh", "k", "l", "m", "ng"
               , "n", "ow", "oy", "p", "r", "sh", "sil", "s", "th", "t", "uh", "uw", "vcl", "v", "w"
               , "y", "zh", "z","$"]
dur_len = 4
mat = np.zeros((48, 48))
start_mat = np.zeros(48)
f = open(in_file,'r')
for line in f:
	token = line.split()
	for ind , val in enumerate(obserstype):
		if token[0] == val:
			head = ind
		if token[1] == val:
			tail = ind
	if head == 49 or tail == 49:
	    continue
	if head ==0 and tail != 0:
		start_mat[tail-1] = np.float32(token[2])
	if head == 0 or tail == 0:
	    continue		
	mat[head-1][tail-1] = np.float32(token[2])
row_sums = mat.sum(axis=1)
tran_mat = mat / row_sums[:, np.newaxis]
where_are_NaNs = np.isnan(tran_mat)
tran_mat[where_are_NaNs] = 0.0
tran_mat = np.log(tran_mat)
start_sum = start_mat.sum()
start_mat = start_mat/start_sum
start_mat = np.log(start_mat)
states = [i for i in range(48)]
#pdb.set_trace()
def float_convert(i):
    try: 
        return np.float32(i)
    except ValueError :
        return i
def print_dptable(V):
    print "    ",
    for i in range(len(V)): print "%7d" % i,
    print

    for y in V[0].keys():
        print "%.5s: " % y,
        for t in range(len(V)):
            print "%.7s" % ("%f" % V[t][y]),
        print
def viterbi_alg(states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}
    status = np.zeros(48)
    arr = np.array([1 for i in range(48)])
    trans_p2 = np.diag(arr)
    trans_p2 = np.log(trans_p2)
    #pdb.set_trace()
    # Initialize base cases (t == 0)
    for y in states:
        V[0][y] = start_p[y] + emit_p[y][0]
        path[y] = [y]
        status[y] = 1
    # Run Viterbi for t > 0
    for t in range(1,len(emit_p[0])):
        V.append({})
        newpath = {}
        for y in states:
            probs =[] 
            for y0 in states:
                if status[y0] >= dur_len:
                    probs.append(V[t-1][y0] + trans_p[y0][y] + emit_p[y][t])
                else:
                    probs.append(V[t-1][y0] + trans_p2[y0][y] + emit_p[y][t])
            (prob,state) = max((probs[i], i) for i in range(len(probs)))
            #(prob, state) = max([(V[t-1][y0] + trans_p[y0][y] + emit_p[y][t], y0) for y0 in states])
            #pdb.set_trace()
            V[t][y] = prob
            newpath[y] = path[state] + [y]
            if y == state:
                status[y] = status[y]+1
            else:
                status[y] = 1

        # Don't need to remember the old paths
        path = newpath
        #pdb.set_trace()
    #print_dptable(V)
    (prob, state) = max([(V[len(emit_p[0]) - 1][y], y) for y in states])
    return (prob, path[state])


test_f = open('RNN_test_softmax.txt','r')
test_ans = open('HMM_ans_1207.csv','w')
test_ans.write('Id,Prediction')
f_test = []
name = []
for line in test_f:
    input_x = line.split()
    input_x = [float_convert(i) for i in input_x]
    name_x = input_x[0].split('_')
    name.append(name_x)
    input_x.pop(0)
    #input_x.pop(-1)
    #input_x.insert(36,input_x.pop(0))
    #pdb.set_trace()
    f_test.append(input_x)
test_num = len(name)
test_c = MAP()
Y=None
m=1
f_test = np.asarray(f_test)
#f_test = np.insert(f_test,0,0.,axis = 1)
seq = np.array([f_test[0]])
#pdb.set_trace()
#obs = [np.argmax(seq)]
#pdb.set_trace()
test_index=0
#test_ans.write('Id,Prediction\n')
end = 0
while (m < test_num):
    while(name[m][0] == name[m-1][0] and name[m][1] == name[m-1][1]):
        #if m <=28:
        	#m = m+1
        	#continue
        seq = np.vstack((seq,f_test[m]))
        #obs = np.append(obs,np.array([np.argmax(f_test[m])]),axis = 0)
        m = m + 1
        #if m == 45:
        	#break
        if (m == test_num):
            end = 1
            break
    seq = seq.transpose()
    #pdb.set_trace()
    seq = np.log(seq)
    seq = seq + seq + seq + seq + seq
    print m
    probs, paths = viterbi_alg(states, start_mat, tran_mat, seq)
    count = 1
    for i in paths:
        if end == 0:
            test_ans.write(name[m-1][0])
            test_ans.write('_')
            test_ans.write(name[m-1][1])
            test_ans.write('_')
            test_ans.write(str(count))
        else:
            test_ans.write(name[m-1][0])
            test_ans.write('_')
            test_ans.write(name[m-1][1])
            test_ans.write('_')
            test_ans.write(str(count))
        test_ans.write(',')
        test_ans.write(test_c.map(int(i)))
        test_ans.write('\n')
        count = count + 1 
    print name[m-1][0]," ",name[m-1][1],"best path is" ,paths
    if end == 1:
        break;
    seq = np.array([f_test[m]])
    #pdb.set_trace()
    #obs = [np.argmax(seq)]
    m = m + 1
test_ans.close()        
