import sys
import csv
import argparse


# 'sampling of process None : 40044.77047920227 ms'


parser = argparse.ArgumentParser(description='For every little thing.')
parser.add_argument('-s', type=int, default=0, choices=range(1,10), help='shrunk rows')
parser.add_argument('-m', action='store_true', default=False, help='sum up rows of SD XL output')
parser.add_argument('-a', action='store_true', default=False,
                    dest='append',
                    help='Append rows.')
parser.add_argument('-i', dest='input_file', action='store',
                    default='',
                    help='Input file.')
parser.add_argument('-o', dest='output_file', action='store',
                    default='',
                    help='Output file.')

args = parser.parse_args()


#print(sys.argv[1])

if args.append:

    with open(args.input_file, 'r', newline='\n') as f_in:
    #with open(sys.argv[1], 'r', newline='\n') as f_in:
        with open(args.output_file, 'a') as f_out:
        #with open('profiling/logs/all.csv', 'a') as f_out:
        #with open(sys.argv[2], 'a') as f_out:
            row = []
            writer = csv.writer(f_out)
            for l in f_in.readlines():
                if l.find('sampling of process ') != -1:
                    row.append(float(l.split()[-2]))

            writer.writerow(row)
            print('row written.')

elif args.s > 0:
    ll = [float(0) for i in range(0, 16)]

    print('s =', args.s)

    with open(args.input_file, 'r') as f_in:
        with open(args.output_file, 'w') as f_out:
            row = []
            reader = csv.reader(f_in, delimiter=',')
            writer = csv.writer(f_out, delimiter=',')
            for count, l in enumerate(reader):
                assert len(l) == 16
                for i in range(0, len(l)):
                    ll[i] = ll[i] + float(l[i])
                if (count+1) % args.s == 0:
                    for i in range(0, len(l)):
                        ll[i] = ll[i] / args.s
                    writer.writerow(ll)
                    ll = [float(0) for i in range(0,16)]

    print('Shrunk.')

elif args.m:
    batch = 0
    iters = 0
    elapsed = 0

    ll = ['', 0, float(0), float(0)]

    with open(args.input_file, 'r') as f_in:
        with open(args.output_file, 'a') as f_out:
            row = []
            reader = csv.reader(f_in, delimiter=',')
            writer = csv.writer(f_out, delimiter=',')
            for l in reader:
                assert len(l) == 4
                elapsed = elapsed + float(l[-1])
                iters = iters + int(l[-2])
                if batch == 0:
                    batch = int(l[1])
                else:
                    assert batch == int(l[1])
            # 2 samplers!!!
            ll = [batch, iters, elapsed, (elapsed/batch)/2, 2*(batch/elapsed)]
            #print('Summed up:', ll)
            writer.writerow(ll)
