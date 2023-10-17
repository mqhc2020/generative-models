import os
import sys
import re
import subprocess
import csv
import argparse
import collections
import threading
import time
from itertools import permutations

GFLOPS_MI250 = (362.1 * 1000)/2
DEFAULT_TARGET_UTIL = 0.8

ENV_TUNING_K_0 = 'ROCBLAS_LAYER'
ENV_TUNING_V_0 = '6'
ENV_TUNING_K_1 = 'TENSILE_DB'
ENV_TUNING_V_1 = '0x8000'

krnl_cat = {
        'attention': [['attention', 'attn', 'gemm_softmax_gemm'], 0],
        'convolution' : [['igemm', 'conv', 'convolution_forward_implicit', 'naive_conv_fwd'], 0],
        'memory copy' : [['copy'], 0],
        'norm' : [['norm'], 0],
        'MIOpen': [['batched_transpose'], 0],
        'rocBLAS' : [['cijk'] ,0],
        }

#"naive_conv_fwd_nchw_half_double_half.kd",45,4955980623,110132902,29.682252163588068

# From rocBLASTER: gemm_ex
GENERIC_ROCBLAS_BENCH_RE = (
        r"./rocblas-bench -f gemm_ex"
        r" --transposeA (?P<TRANSPOSE_A>\w)"
        r" --transposeB (?P<TRANSPOSE_B>\w)"
        r" -m (?P<M>\d+)"
        r" -n (?P<N>\d+)"
        r" -k (?P<K>\d+)"
        r" --alpha (?P<ALPHA>[-+]?(?:\d*\.*\d+))"
        r" --a_type (?P<A_TYPE>\w+)"
        r" --lda (?P<LDA>\d+)"
        r" --b_type (?P<B_TYPE>\w+)"
        r" --ldb (?P<LDB>\d+)"
        r" --beta (?P<BETA>[-+]?(?:\d*\.*\d+))"
        r" --c_type (?P<C_TYPE>\w+)"
        r" --ldc (?P<LDC>\d+)"
        r" --d_type (?P<D_TYPE>\w+)"
        r" --ldd (?P<LDD>\d+)"
        r" --compute_type (?P<COMPUTE_TYPE>\w+)"
        r" --algo (?P<ALGO>\d+)"
        r" --solution_index (?P<SOLUTION_INDEX>\d+)"
        r" --flags (?P<FLAGS>\w+)"
    )

# shorter form: gemm
GENERIC_ROCBLAS_BENCH_RE_SHORT = (
        r"./rocblas-bench -f gemm"
        r" -r (?P<R>\w+)"
        r" --transposeA (?P<TRANSPOSE_A>\w)"
        r" --transposeB (?P<TRANSPOSE_B>\w)"
        r" -m (?P<M>\d+)"
        r" -n (?P<N>\d+)"
        r" -k (?P<K>\d+)"
        r" --alpha (?P<ALPHA>[-+]?(?:\d*\.*\d+))"
        r" --lda (?P<LDA>\d+)"
        r" --ldb (?P<LDB>\d+)"
        r" --beta (?P<BETA>[-+]?(?:\d*\.*\d+))"
        r" --ldc (?P<LDC>\d+)"
    )

blas_krnls = []
lines_of_file = []
amdalh_items = []

def store_blas(krnl, percent, duration):
    blas_krnls.append((krnl, percent, duration))


def filter_and_check_blas(it):
    with open('profiling/bench_kernel_filtered.log', 'w') as f_out:
        bench_count = 0
        kernel_count = 0
        bench_found = False
        kernel_found = False
        while True:
            #s = f_in.readline()
            #if not s:
            #    break
            try:
                s = next(it)
            except:
                break
            if s.find('rocblas-bench ') != -1:
                if bench_found:
                    # This is just to handle the case of two 'rocblas-bench' lines in a row
                    print('Warning: skip one rocblas-bench line that we cannot explain so far...')
                    continue
                bench_found = True
                kernel_found = False
                bench_count = bench_count + 1
                f_out.write(s)
                continue
            if s.find('Running kernel:') != -1:
                if kernel_found:
                    # This is just to handle the case of two 'rocblas-bench' lines in a row
                    print('Warning: skip one kernel line that we cannot explain so far...')
                    continue
                bench_found = False
                kernel_found = True
                kernel_count = kernel_count + 1
                f_out.write(s)
                continue

        if bench_count != kernel_count:
            print('Check failed: rocblas-bench lines and kernel name lines not paired! {} != {}'.format(bench_count, kernel_count))

def filter_and_check_blas_from_file():
    try:
        with open('profiling/bench_kernel.log', 'r') as f_in:
            it = iter(f_in.readlines())
            filter_and_check_blas(it)
    except:
        print('Cannot find the file')


def collect_bench(lines):
    ret = []
    for l in lines:
        pos = l.find('./rocblas-bench ')
        if pos != -1:
            # This is totally application-dependent: for midjourney,
            # the rocblas-bench command always appears at the end of a line.
            ret.append((l[pos:]).rstrip())
    return ret

def run_input_cmd(cmd, env):
    ll = []
    #ress = ''
    #for i in cmd:
    #    ress = ' '.join([ress, i])
    #print(ress)
    child = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    for l in iter(child.stdout.readline, b''):
        ll.append(l.decode())

    child.wait()

    #print('length =', len(ll))

    filter_and_check_blas(iter(ll))
    return ll

def run_bench(cmd, location):
    l_cmd = cmd.split()
    l_cmd[0] = l_cmd[0][2:]
    l_cmd[0] = '{}/{}'.format(location, l_cmd[0])
    child = subprocess.Popen(l_cmd, stdout=subprocess.PIPE)

    for l in iter(child.stdout.readline, b''):
        last_line = l

    child.wait()

    return last_line.decode()

def run_tuning(cmd, location):
    l_cmd = cmd.split()
    l_cmd[0] = '{}/{}'.format(location, l_cmd[0])
    child = subprocess.Popen(l_cmd, stdout=subprocess.PIPE)

    found = False

    for l in iter(child.stdout.readline, b''):
        if l.decode().find('Winner: ') == 0:
            found = True
            res = l.split()

            solution = int(res[1].decode())
            tuned = float(res[3].decode())
            break

    child.wait()

    if not found:
        print('Fail to find the \'Winner\' for', l_cmd)
        return None, None
    else:
        print(solution, tuned)
        return solution, tuned

def get_pairs(all_lines):
    for l in all_lines:
        if l.find('rocblas-bench ') != -1:
            l = l.strip()
            lines_of_file.append(l)
        elif l.find('Running kernel:') != -1:
            l = l.split()
            lines_of_file.append(l[2])
    return lines_of_file

def output_blas(blas_location, target_util):
    with open('profiling/bench_kernel.log', 'r') as f:
        lines_of_file = get_pairs(f.readlines())

        '''
        for l in f.readlines():
            if l.find('rocblas-bench ') != -1:
                l = l.strip()
                lines_of_file.append(l)
            elif l.find('Running kernel:') != -1:
                l = l.split()
                lines_of_file.append(l[2])
        '''

        print('#pairs =', len(lines_of_file))
        print('#cijk =', len(blas_krnls))

        '''
        stop_count = 0
        for s_bench, s_krnl in zip(lines_of_file[0::2], lines_of_file[1::2]):
            stop_count = stop_count + 1
            print(s_bench, s_krnl)
            if stop_count == 2:
                break
        '''

        with open('profiling/output.csv', 'w') as f_out:
            title = ['Percentage', 'Kernels', 'rocbench', 'TransA', 'TransB', 'M', 'N', 'K', 'Solution Index', 'Tuned Time (us)', 'Duration', 'Bench Time (us)', 'Tuning rate', 'Ideal Speedup<Tuning>', 'Gflops', 'Util rate', 'Ideal Speedup<Util>']
            writer = csv.writer(f_out)
            writer.writerow(title)
            for i, stat in enumerate(blas_krnls):
                for s_bench, s_krnl in zip(lines_of_file[0::2], lines_of_file[1::2]):
                    if stat[0].find(s_krnl) != -1:
                        print(i, stat[0])
                        print('\t', s_bench)
                        print('\t', stat[1])
                        mres = re.match(GENERIC_ROCBLAS_BENCH_RE, s_bench)
                        if mres != None:
                            # rocblas-bench -f gemm_ex --transposeA T --transposeB N -m 10240 -n 1536 -k 1280 --alpha 1 --a_type f16_r --lda 1280 --b_type f16_r --ldb 1280 --beta 1 --c_type f16_r --ldc 10240 --d_type f16_r --ldd 10240 --compute_type f32_r --algo 0 --solution_index 0 --flags 0
                            bres = run_bench(s_bench, blas_location)
                            bres = bres.strip()

                            # rocblas-example-user-driven-tuning gemm_ex T N 10240 1536 1280
                            s_tuning = 'rocblas-example-user-driven-tuning gemm_ex {} {} {} {} {}'.format(
                                    mres.group('TRANSPOSE_A'),
                                    mres.group('TRANSPOSE_B'),
                                    mres.group('M'),
                                    mres.group('N'),
                                    mres.group('K')
                                    )

                            solution_index, tuned_time = run_tuning(s_tuning, blas_location)

                            percentage = float(stat[1])/100
                            duration = float(stat[2])/1000 # us
                            bench_time = float(bres.split(',')[-1])
                            tuning_rate = tuned_time/bench_time
                            ideal_speedup_1 = (float(stat[1])/100)*(tuning_rate)

                            gflops = float(bres.split(',')[-2])
                            util_rate = gflops/GFLOPS_MI250
                            ideal_speedup_2 = (float(stat[1])/100)*(util_rate/target_util)

                            csv_line = [
                                percentage,
                                stat[0],
                                s_bench,
                                mres.group('TRANSPOSE_A'),
                                mres.group('TRANSPOSE_B'),
                                mres.group('M'),
                                mres.group('N'),
                                mres.group('K'),
                                solution_index,
                                tuned_time,
                                duration,
                                bench_time,
                                tuning_rate,
                                ideal_speedup_1,
                                gflops,
                                util_rate,
                                ideal_speedup_2,
                                ]
                            writer.writerow(csv_line)

                            amdalh_items.append((percentage, ideal_speedup_1, ideal_speedup_2))

                        break

def output_amdalh(target_util):
    F1 = 0
    F2 = 0
    F3 = 0
    for f1, f2, f3 in amdalh_items:
        F1 = F1 + f1
        F2 = F2 + f2
        F3 = F3 + f3
    print('Estimated speedup suppose all rocBLAS kernels are tuned : {} (based on results of rocblas-example-user-driven-tuning)'.format(1 / (1 - F1 + F2)))
    print('Estimated speedup suppose utilization of all rocBLAS kernels are optimized to {}% : {} (based on results of rocblas-bench)'.format(target_util * 100, 1 / (1 - F1 + F3)))


class RedundancyChecker:
    def __init__(self):
        self.lookup_table = []
        self.m = 0
        self.n = 0
        self.k = 0

    # Current rule: ignore TransposeA and TransposeB, consider only M, N, K
    def parse_match(self, match):
        self.m = match.group('M')
        self.n = match.group('N')
        self.k = match.group('K')
        return (self.m, self.n, self.k)

    def is_redundant(self, match):
        key = self.parse_match(match)
        for e in self.lookup_table:
            if key in permutations(iter(e), len(key)):
                print('Combination found in lookup table.')
                return True
        self.lookup_table.append(key)
        print('Combination', key, 'added.')
        return False

def output_solutions(lines, blas_location, csv_file, is_short, need_skip):

    '''
    with open('debug.log', 'w') as f:
        for i in pairs:
            if i.lower().find('running kernel') != -1:
                f.write(i)
    '''

#    pairs = get_pairs(lines)

    checker = RedundancyChecker()

    writer = csv.writer(csv_file)
    title = ['transA','transB','M','N','batch_count','K','alpha','beta','lda','ldb','ldc','input_type','output_type','compute_type','solution_index']
    writer.writerow(title)

    for s_bench in lines:
#    for s_bench, s_krnl in zip(lines_of_file[0::2], lines_of_file[1::2]):
        bench_re = GENERIC_ROCBLAS_BENCH_RE_SHORT if is_short else GENERIC_ROCBLAS_BENCH_RE
        mres = re.match(bench_re, s_bench)
        if mres == None:
            print('FATAL: can\'t parse this line \"{}\". Command format: {}.'.format(s_bench, 'GEMM' if is_short else 'GEMM_EX'))
        else:
            # GEMM_EX: rocblas-bench -f gemm_ex --transposeA T --transposeB N -m 10240 -n 1536 -k 1280 --alpha 1 --a_type f16_r --lda 1280 --b_type f16_r --ldb 1280 --beta 1 --c_type f16_r --ldc 10240 --d_type f16_r --ldd 10240 --compute_type f32_r --algo 0 --solution_index 0 --flags 0
            # GEMM: rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 3072 -n 256 -k 3072 --alpha 1 --lda 3072 --ldb 3072 --beta 0 --ldc 3072
            if need_skip:
                if checker.is_redundant(mres):
                    continue

            bres = run_bench(s_bench, blas_location)
            bres = bres.strip()

            # rocblas-example-user-driven-tuning gemm_ex T N 10240 1536 1280
            s_tuning = 'rocblas-example-user-driven-tuning {} {} {} {} {} {}'.format(
                        'gemm' if is_short else 'gemm_ex',
                        mres.group('TRANSPOSE_A'),
                        mres.group('TRANSPOSE_B'),
                        mres.group('M'),
                        mres.group('N'),
                        mres.group('K')
                        )

            print('tuning cmd:', s_tuning)

            solution_index, _ = run_tuning(s_tuning, blas_location)

            # rocBLAS solution file format:
            # transA,transB,M,N,batch_count,K,alpha,beta,lda,ldb,ldc,input_type,output_type,compute_type,solution_index
            # T,N,3072,77,1,1024,1,1,1024,1024,3072,f16_r,f16_r,f32_r,1509
            # ......
            if is_short:
                input_type = mres.group('R')
                output_type = mres.group('R')
                compute_type = mres.group('R')
            else:
                assert mres.group('A_TYPE') == mres.group('B_TYPE')
                assert mres.group('A_TYPE') == mres.group('C_TYPE')
                input_type = mres.group('A_TYPE')
                output_type = mres.group('D_TYPE')
                compute_type = mres.group('COMPUTE_TYPE')
            
            csv_line = [
                    mres.group('TRANSPOSE_A'),
                    mres.group('TRANSPOSE_B'),
                    mres.group('M'),
                    mres.group('N'),
                    1,
                    mres.group('K'),
                    mres.group('ALPHA'),
                    mres.group('BETA'),
                    mres.group('LDA'),
                    mres.group('LDB'),
                    mres.group('LDC'),
                    input_type,
                    output_type,
                    compute_type,
                    solution_index
                    ]
            writer.writerow(csv_line)

class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __eq__(self, other):
        return self.start <= other <= self.end

def main():
    parser = argparse.ArgumentParser(description='For parsing rocprof stats and also other stuff for performance tuning.')
    parser.add_argument('--blas-tuning', action="store_true", default=False)
    parser.add_argument('--stats', action='store_true', default=False,
                    dest='stats_parsing',
                    help='rocprof stats parsing')
    parser.add_argument('-p', action='store_true', default=False,
                    dest='blas_projecting',
                    help='rocBLAS projecting')
    parser.add_argument('-t', action='store_true', default=False,
                    dest='blas_tuning',
                    help='rocBLAS Tuning')
    parser.add_argument('-s', action='store_true', default=False,
                    dest='blas_tuning_slow',
                    help='rocBLAS Tuning via a slow approach')
    parser.add_argument('--prof-stats-file', dest='prof_stats_file', action='store',
                    default='',
                    help='location of rocBLAS clients')
    parser.add_argument('--csv-file', action='store',
                    default='',
                    help='csv file that contains solutions')
    parser.add_argument('--bench-log-file', action='store',
                    default='',
                    help='This is necessary because sometimes the application is not allowed to run by us directly')
    parser.add_argument('--blas-clients-location', dest='blas_location', action='store',
                    default='/dockerx/rocBLAS/build/release/clients/staging',
                    help='location of rocBLAS clients')
    parser.add_argument('--util', type=float, default=DEFAULT_TARGET_UTIL, choices=[Range(0.0, 1.0)])
    parser.add_argument('--bs', type=int, default=DEFAULT_TARGET_UTIL, choices=range(1, 1024))
    parser.add_argument('cmd_as_rest', nargs=argparse.REMAINDER)
    args = parser.parse_args()


    if args.stats_parsing:
        if args.prof_stats_file == '':
            print('Found no input file.')
            sys.exit()
        elif os.path.isfile(args.prof_stats_file) == False:
            print('Not a file:', args.prof_stats_file)
            sys.exit()

        with open(args.prof_stats_file) as f:
            count = 0
            for l in f.readlines():
                count = count + 1
                if count == 1:
                    continue

                cols = l.split('"')
                for k, v in krnl_cat.items():
                    found = False
                    for w in v[0]:
                        if cols[1].lower().find(w.lower()) != -1:
                            found = True
                            ns = l[1:].split(',')
                            v[1] = v[1] + float(ns[-1])
                            if args.blas_projecting and k == 'rocBLAS':
                                store_blas(cols[1], float(ns[-1]), ns[-2])
                                #print(cols[1])
                            break
                    if found == True:
                        break


        total = 100
        for k, v in krnl_cat.items():
            #print(k, ':', v[1])
            print(v[1], end = ',')
            total = total - float(v[1])

        #print('others', ':', total)
        print(total, end=',')
        print('\r')

        if args.blas_projecting:
            filter_and_check_blas_from_file()
            output_blas(args.blas_location, args.util)
            output_amdalh(args.util)

    if args.blas_tuning or args.blas_tuning_slow:
        if args.csv_file == '':
            print('Found no csv file.')
            sys.exit()

        with open(args.csv_file, 'w') as f_out:
            env_tuning = os.environ.copy()
            env_tuning[ENV_TUNING_K_0] = ENV_TUNING_V_0
            env_tuning[ENV_TUNING_K_1] = ENV_TUNING_V_1
            if args.blas_tuning_slow and args.bench_log_file != '':
                with open(args.bench_log_file, 'r') as f_in:
                    lines = f_in.readlines()
            else:
                lines = run_input_cmd(args.cmd_as_rest, env_tuning)
            lines = list(dict.fromkeys(lines)) # remove duplicates
            if args.blas_tuning_slow:
                lines = collect_bench(lines)
                for l in lines:
                    print(l)
            is_short = True if args.blas_tuning_slow else False
            output_solutions(lines, args.blas_location, f_out, is_short, True)



if __name__ == '__main__':
    main()

