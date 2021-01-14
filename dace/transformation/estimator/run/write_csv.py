import matplotlib.pyplot as plt 
from operator import add 

'''
INPUT_DIR = 'outputs'
INPUT_NAME = 'hdiff_output'
CSV_FILENAME = 'test.csv'
AGGREGATE = False 
INCLUDE_OCCUPANCY = False 
INCLUDE_EFFICIENCY = False 
MAX_I = 87
'''

INPUT_DIR = './outputs/'
INPUT_NAME = 'vadv_output'
CSV_FILENAME = 'test.csv'
AGGREGATE = False 
INCLUDE_OCCUPANCY = False 
INCLUDE_EFFICIENCY = False 
INCLUDE_THROUGHPUT = True
MAX_I = 9


size_dict = {'Mbyte': 1, 
             'Kbyte': 0.001,
             'byte' : 0.000001,
             'usecond' : 1,
             'msecond' : 1000,
             '%': 1
             }

def is_separating_line(line):
    return len(line) > 1 and all([len(e) == 0 or all(ee == '-' for ee in e) for e in line.split(' ')])



with open(CSV_FILENAME, 'w') as f:
    f.write('id, read, write, sum, throughput, runtime, registers, occupancy, read_eff, write_eff\n')
    for i in range(1,MAX_I):
        print(f"--- Current file: {i} ---")
        file_len = 0
        file_name = INPUT_DIR + '/' + INPUT_NAME + str(i) + '.txt'
        with open(file_name,'r') as current:
            for line in current:
                file_len += 1

        with open(file_name,'r') as current:
            line = current.readline()[:-1]
            counter = 0
            while counter < file_len:
                if is_separating_line(line):
                    read = current.readline()[:-1].split(' ')
                    write = current.readline()[:-1].split(' ')
                    throughput = current.readline()[:-1].split(' ')
                    duration = current.readline()[:-1].split(' ')
                    registers = current.readline()[:-1].split(' ')
                    occupancy = current.readline()[:-1].split(' ')
                    read_efficiency = current.readline()[:-1].split(' ')
                    write_efficiency = current.readline()[:-1].split(' ')


                    for _ in range(4):
                        _ = current.readline() 
                    counter += 4 

                    x = f'{i}'
                    rwd = [] 
                    for ln in [read, write, throughput, duration]:
                        quantity = None
                        # pre-multiplier 
                        for s in ln:
                            if len(s) == 0:
                                continue 
                            if s in size_dict:
                                quantity = size_dict[s]
                            if quantity is not None:
                                try:
                                    quantity *= float(s)
                                    break 
                                except ValueError:
                                    pass 
                        else:
                            raise RuntimeError("Cannot convert row!")
                        
                        rwd.append(quantity)
                    
                    x = f'{i},{rwd[0]},{rwd[1]},{rwd[0]+rwd[1]},{rwd[2]},{rwd[3]}'

                    rwd = []
                    for ln in [registers, occupancy, read_efficiency, write_efficiency]:
                        for s in ln:
                            try:
                                quantity = float(s)
                                break 
                            except ValueError:
                                pass 
                        else:
                            raise RuntimeError("Cannot convert row!")
                        rwd.append(quantity)
                        
                    x += f',{rwd[0]}, {rwd[1]}, {rwd[2]}, {rwd[3]}\n'
                    print("Line =", x )
                    f.write(x)
                line = current.readline()[:-1]
                counter += 1

# create graph 
with open(CSV_FILENAME, 'r') as f:
    if AGGREGATE:
        aggregates = dict() 
        for (i,line) in enumerate(f):
            if i == 0:
                continue
            id = int(line.split(',')[0])
            (read,write,sumrw,throughput,duration, _, occupancy, _, _) = [float(e) for e in line.split(',')[1:]]
            if id in aggregates:
                aggregates[id] = list(map(add, aggregates[id], [read, write, sumrw, duration]))
            else:
                aggregates[id] = [read, write, sumrw, duration]
        x, y = [], []
        for (r, w, s, d) in aggregates.values():
            x.append(s)
            y.append(d)
        plt.scatter(x,y)
        plt.title('Aggregate Runtime Correlation per Graph')
        plt.xlabel('DRAM Memory Movement [MB]')
        plt.ylabel('Runtime [us]')
        plt.title('Runtime Correlation')
        plt.savefig('plot.pdf')
    else:
        x, y = [], []
        for (i,line) in enumerate(f):
            if i == 0:
                continue
            print([float(e) for e in line.split(',')[1:]])
            (read,write,sumrw,throughput,duration,_,occupancy,read_eff,write_eff) = [float(e) for e in line.split(',')[1:]]
            x.append([read, write])
            if INCLUDE_OCCUPANCY:
                x[-1][0] /= occupancy 
                x[-1][1] /= occupancy 
            if INCLUDE_EFFICIENCY:
                x[-1][0] /= read_eff 
                x[-1][1] /= write_eff 
            if INCLUDE_THROUGHPUT:
                x[-1][0] /= throughput
                x[-1][0] /= throughput
            x[-1] = x[-1][0] + x[-1][1]
            y.append(duration)
        plt.scatter(x,y)
        if INCLUDE_OCCUPANCY and INCLUDE_EFFICIENCY:
            plt.xlabel('DRAM Memory Movement [MB] / (occupancy * efficiency)')
        elif INCLUDE_OCCUPANCY:
            plt.xlabel('DRAM Memory Movement [MB] / occupancy')
        elif INCLUDE_EFFICIENCY:
            plt.xlabel('DRAM Memory Movement [MB] / efficiency')
        elif INCLUDE_THROUGHPUT:
            plt.xlabel('DRAM Memory ')
        else:
            plt.xlabel('DRAM Memory Movement [MB]')

        plt.ylabel('Runtime [us]')
        plt.title('Runtime Correlation per Kernel')
        plt.savefig('plot.pdf')